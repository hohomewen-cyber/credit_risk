"""
模型融合
========
将多个基模型的预测结果按配置融合。

支持的融合方式:
    1. 加权平均（weighted_avg）
    2. 投票（voting）
    3. Stacking（元学习器）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger("models.ensemble")


class EnsembleModel(BaseModel):
    """
    模型融合。

    使用示例:
        >>> ensemble = EnsembleModel(model_config.ensemble)
        >>> ensemble.add_model("xgboost", xgb_model, weight=0.6)
        >>> ensemble.add_model("mlp", mlp_model, weight=0.4)
        >>> ensemble.fit(X_train, y_train)  # 仅 stacking 模式需要
        >>> proba = ensemble.predict_proba(X_test)
    """

    def __init__(self, config, name: str = "ensemble"):
        super().__init__(config, name)
        self.method = config.method
        self.weights_cfg = dict(config.weights) if config.method == "weighted_avg" else {}

        self.models: Dict[str, BaseModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.meta_learner = None  # stacking 用

    def add_model(self, name: str, model: BaseModel, weight: Optional[float] = None) -> None:
        """添加基模型"""
        self.models[name] = model
        if weight is None:
            weight = self.weights_cfg.get(name, 1.0)
        self.model_weights[name] = weight
        logger.info(f"已添加基模型: {name} (权重: {weight})")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val=None,
        y_val=None,
        **kwargs,
    ) -> "EnsembleModel":
        """
        融合模型训练。
        - weighted_avg / voting: 假设基模型已训练，仅做权重归一化
        - stacking: 训练元学习器
        """
        if not self.models:
            raise RuntimeError("请先调用 add_model() 添加基模型")

        # 检查基模型是否都已训练
        for name, model in self.models.items():
            if not model.is_fitted:
                raise RuntimeError(f"基模型 {name} 未训练")

        if self.method == "weighted_avg":
            self._normalize_weights()
        elif self.method == "stacking":
            self._train_meta_learner(X_train, y_train)

        self._is_fitted = True
        self._feature_names = list(self.models.values())[0].feature_names
        logger.info(f"融合模型 ({self.method}) 已就绪")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        self._check_fitted()

        # 收集所有基模型的预测
        base_preds = {}
        for name, model in self.models.items():
            preds = model.predict_proba(X)
            if preds.ndim == 2:
                preds = preds[:, 1]
            base_preds[name] = preds

        if self.method == "weighted_avg":
            return self._weighted_average(base_preds)
        elif self.method == "voting":
            return self._voting(base_preds)
        elif self.method == "stacking":
            return self._stacking_predict(base_preds)
        else:
            raise ValueError(f"未知融合方法: {self.method}")

    # -------- 各融合策略实现 --------

    def _normalize_weights(self) -> None:
        """归一化权重"""
        total = sum(self.model_weights.values())
        if total > 0:
            self.model_weights = {k: v / total for k, v in self.model_weights.items()}

    def _weighted_average(self, base_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """加权平均"""
        result = np.zeros(len(next(iter(base_preds.values()))))
        for name, preds in base_preds.items():
            weight = self.model_weights.get(name, 1.0 / len(base_preds))
            result += weight * preds
        return result

    def _voting(self, base_preds: Dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """硬投票（多数表决）转回概率"""
        votes = np.array([p >= threshold for p in base_preds.values()])
        return votes.mean(axis=0)

    def _train_meta_learner(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """训练元学习器（stacking 模式）"""
        from sklearn.linear_model import LogisticRegression

        # 用基模型预测训练集（理想情况应该用 OOF，这里简化）
        meta_features = []
        for name, model in self.models.items():
            preds = model.predict_proba(X_train)
            if preds.ndim == 2:
                preds = preds[:, 1]
            meta_features.append(preds)

        meta_X = np.column_stack(meta_features)
        self.meta_learner = LogisticRegression(max_iter=1000)
        self.meta_learner.fit(meta_X, y_train)

    def _stacking_predict(self, base_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Stacking 预测"""
        meta_X = np.column_stack(list(base_preds.values()))
        return self.meta_learner.predict_proba(meta_X)[:, 1]

    # -------- 持久化 --------

    def save(self, file_path: str) -> None:
        """保存融合配置（基模型需单独保存）"""
        self._check_fitted()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        info = {
            "method": self.method,
            "model_names": list(self.models.keys()),
            "weights": self.model_weights,
            "feature_names": self._feature_names,
        }

        # stacking: 保存 meta learner
        if self.method == "stacking" and self.meta_learner is not None:
            import joblib
            joblib.dump(self.meta_learner, path.with_suffix(".meta.pkl"))

        path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"融合配置已保存至 {path}")

    def load(self, file_path: str) -> "EnsembleModel":
        """加载融合配置"""
        path = Path(file_path)
        info = json.loads(path.read_text(encoding="utf-8"))

        self.method = info["method"]
        self.model_weights = info["weights"]
        self._feature_names = info["feature_names"]

        if self.method == "stacking":
            import joblib
            meta_path = path.with_suffix(".meta.pkl")
            if meta_path.exists():
                self.meta_learner = joblib.load(meta_path)

        self._is_fitted = True
        logger.warning("注意：基模型需要单独加载并通过 add_model() 添加")
        return self
