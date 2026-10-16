"""
模型基类
========
定义所有模型的统一接口，方便后续替换、扩展、对比。

要点:
    "面向接口编程" 是降低耦合的关键。
    XGBoost、MLP、Ensemble 都继承同一基类，
    上层代码无需关心具体实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    模型抽象基类，定义统一接口。

    子类需要实现:
        - fit: 训练
        - predict_proba: 预测概率
        - save / load: 持久化
    """

    def __init__(self, config: Any, name: str = "base_model"):
        self.config = config
        self.name = name
        self._is_fitted = False
        self._feature_names: Optional[list] = None
        self._train_history: Dict = {}

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "BaseModel":
        """训练模型"""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（返回正类概率）"""
        ...

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """预测标签"""
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[:, 1]
        return (proba >= threshold).astype(int)

    @abstractmethod
    def save(self, file_path: str) -> None:
        """保存模型"""
        ...

    @abstractmethod
    def load(self, file_path: str) -> "BaseModel":
        """加载模型"""
        ...

    def get_feature_importance(self) -> Optional[pd.Series]:
        """获取特征重要性（部分模型支持）"""
        return None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> Optional[list]:
        return self._feature_names

    @property
    def train_history(self) -> Dict:
        return self._train_history

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} 模型未训练，请先调用 fit()")
