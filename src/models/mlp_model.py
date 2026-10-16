"""
MLP（多层感知器）模型
=====================
基于 PyTorch 实现的 MLP 二分类器。

要点:
    1. 配置驱动的网络架构（层数、激活函数、Dropout 都可配）
    2. 早停 + 学习率调度
    3. 类别不平衡：BCE + 类别权重 / Focal Loss
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger("models.mlp")


class MLPNetwork(nn.Module):
    """
    MLP 网络结构。

    根据配置动态构建：
        Input -> [Linear -> BatchNorm -> Activation -> Dropout] x N -> Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        activation: str = "relu",
        dropout: float = 0.3,
        batch_norm: bool = True,
        output_dim: int = 1,
        output_activation: str = "sigmoid",
    ):
        super().__init__()

        # 激活函数映射
        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
        }
        Act = act_map.get(activation.lower(), nn.ReLU)

        layers = []
        prev_dim = input_dim

        # 隐藏层
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPModel(BaseModel):
    """
    MLP 模型封装。

    使用示例:
        >>> model = MLPModel(model_config.mlp)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> proba = model.predict_proba(X_test)
    """

    def __init__(self, config, name: str = "mlp"):
        super().__init__(config, name)
        self.arch_cfg = config.architecture
        self.train_cfg = config.training
        self.device = self._setup_device(config.device)

        self.network: Optional[MLPNetwork] = None
        self._train_history = {"train_loss": [], "val_loss": [], "val_auc": []}

    @staticmethod
    def _setup_device(device_str: str) -> torch.device:
        """设置设备"""
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "MLPModel":
        """训练 MLP"""
        logger.info(f"开始训练 MLP - 设备: {self.device}, 样本: {len(X_train)}")

        self._feature_names = X_train.columns.tolist()
        input_dim = X_train.shape[1]

        # 构建网络
        self.network = MLPNetwork(
            input_dim=input_dim,
            hidden_layers=list(self.arch_cfg.hidden_layers),
            activation=self.arch_cfg.activation,
            dropout=self.arch_cfg.dropout,
            batch_norm=self.arch_cfg.batch_norm,
            output_activation=self.arch_cfg.output_activation,
        ).to(self.device)

        logger.info(f"网络结构:\n{self.network}")

        # 数据加载器
        train_loader = self._create_loader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_loader(X_val, y_val, shuffle=False)

        # 损失函数（类别加权）
        criterion = self._build_loss(y_train)

        # 优化器
        optimizer = self._build_optimizer()

        # 学习率调度器
        scheduler = self._build_scheduler(optimizer)

        # 训练循环
        best_val_auc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(self.train_cfg.epochs):
            train_loss = self._train_one_epoch(train_loader, optimizer, criterion)
            val_loss, val_auc = self._validate(val_loader, criterion) if val_loader else (None, None)

            # 记录历史
            self._train_history["train_loss"].append(train_loss)
            if val_loss is not None:
                self._train_history["val_loss"].append(val_loss)
                self._train_history["val_auc"].append(val_auc)

            # 调度学习率
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    scheduler.step()

            # 日志
            log_msg = f"Epoch {epoch+1}/{self.train_cfg.epochs} - train_loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", val_loss: {val_loss:.4f}, val_auc: {val_auc:.4f}"
            logger.info(log_msg)

            # 早停
            if val_auc is not None and self.train_cfg.early_stopping.enabled:
                if val_auc > best_val_auc + self.train_cfg.early_stopping.min_delta:
                    best_val_auc = val_auc
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.train_cfg.early_stopping.patience:
                        logger.info(f"早停触发 @ epoch {epoch+1}, 最佳 val_auc={best_val_auc:.4f}")
                        break

        # 恢复最佳权重
        if best_state is not None:
            self.network.load_state_dict(best_state)

        self._is_fitted = True
        logger.info(f"MLP 训练完成 - 最佳 val_auc: {best_val_auc:.4f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        self._check_fitted()
        if self._feature_names:
            X = X[self._feature_names]

        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X.values).to(self.device)
            probs = self.network(x_tensor).cpu().numpy().flatten()
        return probs

    # -------- 内部辅助方法 --------

    def _create_loader(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        shuffle: bool = True,
    ) -> DataLoader:
        """创建 DataLoader"""
        x_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,  # 教学环境用 0 避免 fork 问题
        )

    def _build_loss(self, y_train: pd.Series) -> nn.Module:
        """构建损失函数（含类别权重）"""
        if self.train_cfg.loss == "bce":
            if self.train_cfg.class_weight == "balanced":
                # 类别加权 BCE
                pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                pos_weight_tensor = torch.tensor([pos_weight]).to(self.device)
                # BCEWithLogitsLoss 接受 logits，但我们网络已加了 sigmoid，所以用 BCELoss
                # 这里简化处理：用 BCELoss + 手动加权
                return WeightedBCELoss(pos_weight=pos_weight)
            return nn.BCELoss()
        elif self.train_cfg.loss == "focal":
            return FocalLoss()
        else:
            raise ValueError(f"不支持的损失函数: {self.train_cfg.loss}")

    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        opt_name = self.train_cfg.optimizer.lower()
        params = self.network.parameters()
        lr = self.train_cfg.learning_rate
        wd = self.train_cfg.weight_decay

        if opt_name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"不支持的优化器: {opt_name}")

    def _build_scheduler(self, optimizer: optim.Optimizer):
        """构建学习率调度器"""
        scheduler_name = self.train_cfg.scheduler
        if scheduler_name == "none":
            return None
        elif scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.train_cfg.epochs
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
        return None

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """训练一个 epoch"""
        self.network.train()
        total_loss = 0.0
        n_samples = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            output = self.network(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            n_samples += x_batch.size(0)

        return total_loss / max(n_samples, 1)

    def _validate(self, loader: DataLoader, criterion: nn.Module):
        """验证"""
        from sklearn.metrics import roc_auc_score

        self.network.eval()
        total_loss = 0.0
        n_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.network(x_batch)
                loss = criterion(output, y_batch)

                total_loss += loss.item() * x_batch.size(0)
                n_samples += x_batch.size(0)

                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        avg_loss = total_loss / max(n_samples, 1)
        auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
        return avg_loss, auc

    def save(self, file_path: str) -> None:
        """保存模型"""
        self._check_fitted()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 同时保存网络权重和元信息
        checkpoint = {
            "state_dict": self.network.state_dict(),
            "input_dim": self.network.network[0].in_features,
            "feature_names": self._feature_names,
            "config": dict(self.config) if hasattr(self.config, "__dict__") else None,
            "train_history": self._train_history,
        }
        torch.save(checkpoint, str(path))
        logger.info(f"MLP 模型已保存至 {path}")

    def load(self, file_path: str) -> "MLPModel":
        """加载模型"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self._feature_names = checkpoint["feature_names"]
        self._train_history = checkpoint.get("train_history", {})

        # 重建网络
        self.network = MLPNetwork(
            input_dim=checkpoint["input_dim"],
            hidden_layers=list(self.arch_cfg.hidden_layers),
            activation=self.arch_cfg.activation,
            dropout=self.arch_cfg.dropout,
            batch_norm=self.arch_cfg.batch_norm,
            output_activation=self.arch_cfg.output_activation,
        ).to(self.device)
        self.network.load_state_dict(checkpoint["state_dict"])

        self._is_fitted = True
        logger.info(f"MLP 模型已从 {path} 加载")
        return self


# ===== 自定义损失函数 =====
class WeightedBCELoss(nn.Module):
    """加权 BCE 损失（处理类别不平衡）"""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        loss_pos = -self.pos_weight * target * torch.log(pred)
        loss_neg = -(1 - target) * torch.log(1 - pred)
        return (loss_pos + loss_neg).mean()


class FocalLoss(nn.Module):
    """Focal Loss（聚焦难分类样本）"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        pt = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()
