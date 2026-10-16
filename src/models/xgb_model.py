"""
XGBoost 模型封装
================
对 XGBClassifier 进行配置化封装，统一接口。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger("models.xgb")


class XGBModel(BaseModel):
    """
    XGBoost 二分类器封装。

    使用示例:
        >>> model = XGBModel(model_config.xgboost)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> proba = model.predict_proba(X_test)
    """

    def __init__(self, config, name: str = "xgboost"):
        super().__init__(config, name)
        self.params = dict(config.params)  # 转为普通 dict
        self.model: Optional[xgb.XGBClassifier] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "XGBModel":
        """训练 XGBoost 模型"""
        logger.info(f"开始训练 XGBoost - 训练样本: {len(X_train)}, 特征: {X_train.shape[1]}")

        # 自动调整 scale_pos_weight（如果配置中没有显式设置）
        if "scale_pos_weight" not in self.params or self.params["scale_pos_weight"] is None:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            self.params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            logger.info(f"自动设置 scale_pos_weight = {self.params['scale_pos_weight']:.2f}")

        # 构造 XGBClassifier
        self.model = xgb.XGBClassifier(**self.params)

        # 准备验证集（用于早停）
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        # 早停参数
        fit_kwargs = {"eval_set": eval_set, "verbose": False}
        if self.config.early_stopping.enabled and eval_set:
            # XGBoost 2.x 版本将早停参数放在 init 里
            self.model.set_params(early_stopping_rounds=self.config.early_stopping.rounds)

        # 训练
        self.model.fit(X_train, y_train, **fit_kwargs)

        # 记录训练历史
        if hasattr(self.model, "evals_result_"):
            self._train_history = self.model.evals_result_

        self._is_fitted = True
        self._feature_names = X_train.columns.tolist()

        # 打印训练结果
        if X_val is not None:
            from sklearn.metrics import roc_auc_score
            val_pred = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred)
            logger.info(f"XGBoost 训练完成 - 验证集 AUC: {val_auc:.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（返回正类概率，shape=(n,)）"""
        self._check_fitted()
        # 对齐特征
        if self._feature_names:
            X = X[self._feature_names]
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """
        获取特征重要性。

        参数:
            importance_type: weight/gain/cover/total_gain/total_cover
        """
        self._check_fitted()
        booster = self.model.get_booster()
        importance_dict = booster.get_score(importance_type=importance_type)

        # 标准化为 Series（包含所有特征，缺失的填 0）
        importance = pd.Series(0.0, index=self._feature_names)
        for k, v in importance_dict.items():
            if k in importance.index:
                importance[k] = v
            else:
                # XGBoost 可能用 f0/f1 等表示特征
                idx = int(k[1:]) if k.startswith("f") else None
                if idx is not None and idx < len(self._feature_names):
                    importance[self._feature_names[idx]] = v

        return importance.sort_values(ascending=False)

    def save(self, file_path: str) -> None:
        """保存模型（XGBoost 原生 JSON 格式）"""
        self._check_fitted()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save_model(str(path))

        # 单独保存特征名（XGBoost JSON 不包含）
        feature_path = path.with_suffix(".features.txt")
        feature_path.write_text("\n".join(self._feature_names), encoding="utf-8")

        logger.info(f"XGBoost 模型已保存至 {path}")

    def load(self, file_path: str) -> "XGBModel":
        """加载模型"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))

        # 加载特征名
        feature_path = path.with_suffix(".features.txt")
        if feature_path.exists():
            self._feature_names = feature_path.read_text(encoding="utf-8").splitlines()

        self._is_fitted = True
        logger.info(f"XGBoost 模型已从 {path} 加载")
        return self
