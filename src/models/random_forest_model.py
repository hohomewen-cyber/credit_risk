"""Random Forest 模型封装，遵循项目 BaseModel 统一接口。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger("models.random_forest")


class RandomForestModel(BaseModel):
    """随机森林二分类器封装。"""

    def __init__(self, config, name: str = "random_forest"):
        super().__init__(config, name)
        self.params = dict(config.params)
        self.model: Optional[RandomForestClassifier] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "RandomForestModel":
        logger.info(
            f"开始训练 Random Forest - 训练样本: {len(X_train)}, "
            f"特征: {X_train.shape[1]}"
        )
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self._feature_names = X_train.columns.tolist()
        self._is_fitted = True

        if X_val is not None and y_val is not None:
            from sklearn.metrics import roc_auc_score

            val_pred = self.predict_proba(X_val)
            logger.info(
                f"Random Forest 训练完成 - 验证集 AUC: "
                f"{roc_auc_score(y_val, val_pred):.4f}"
            )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self._feature_names:
            X = X[self._feature_names]
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Optional[pd.Series]:
        self._check_fitted()
        return pd.Series(
            self.model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)

    def save(self, file_path: str) -> None:
        self._check_fitted()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self._feature_names}, path
        )
        logger.info(f"Random Forest 模型已保存至 {path}")

    def load(self, file_path: str) -> "RandomForestModel":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        payload = joblib.load(path)
        self.model = payload["model"]
        self._feature_names = payload["feature_names"]
        self._is_fitted = True
        logger.info(f"Random Forest 模型已从 {path} 加载")
        return self
