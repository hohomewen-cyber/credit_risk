"""
特征筛选器
==========
通过方差、相关性、模型重要性等多种方法筛选特征。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from src.utils.logger import get_logger

logger = get_logger("features.selector")


class FeatureSelector:
    """
    特征筛选器。

    使用示例:
        >>> selector = FeatureSelector(feature_config)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(self, config):
        self.config = config
        self.selection_cfg = config.feature_selection
        self._selected_features: Optional[List[str]] = None
        self._dropped_features: List[str] = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """拟合并筛选"""
        return self.fit(X, y).transform(X)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """学习需要保留的特征"""
        if not self.selection_cfg.enabled:
            self._selected_features = X.columns.tolist()
            return self

        current_cols = X.columns.tolist()
        logger.info(f"开始特征筛选 - 输入特征数: {len(current_cols)}")

        for method_cfg in self.selection_cfg.methods:
            method_name = method_cfg["name"]

            if method_name == "variance":
                current_cols = self._filter_by_variance(X[current_cols], method_cfg)
            elif method_name == "correlation":
                current_cols = self._filter_by_correlation(X[current_cols], method_cfg)
            elif method_name == "model_based":
                current_cols = self._filter_by_model(X[current_cols], y, method_cfg)
            else:
                logger.warning(f"未知的筛选方法: {method_name}")

            logger.info(f"  应用 [{method_name}] 后 - 剩余特征数: {len(current_cols)}")

        self._selected_features = current_cols
        self._dropped_features = [c for c in X.columns if c not in current_cols]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用筛选结果"""
        if self._selected_features is None:
            return X
        # 仅保留同时存在于训练时和当前数据中的特征
        cols = [c for c in self._selected_features if c in X.columns]
        return X[cols]

    # -------- 筛选方法 --------

    def _filter_by_variance(self, X: pd.DataFrame, cfg: Dict) -> List[str]:
        """方差筛选：剔除方差极小的特征"""
        threshold = cfg.get("threshold", 0.01)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X.fillna(0))
        return X.columns[selector.get_support()].tolist()

    def _filter_by_correlation(self, X: pd.DataFrame, cfg: Dict) -> List[str]:
        """相关性筛选：高相关特征对中保留一个"""
        threshold = cfg.get("threshold", 0.95)

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        return [c for c in X.columns if c not in to_drop]

    def _filter_by_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cfg: Dict,
    ) -> List[str]:
        """基于模型重要性筛选 Top K 特征"""
        model_name = cfg.get("model", "xgboost")
        top_k = cfg.get("top_k", 50)

        if model_name == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            raise ValueError(f"不支持的筛选模型: {model_name}")

        model.fit(X.fillna(0), y)

        importance = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importance.nlargest(top_k).index.tolist()
        return top_features

    @property
    def selected_features(self) -> List[str]:
        """获取保留的特征列表"""
        return self._selected_features or []

    @property
    def dropped_features(self) -> List[str]:
        """获取被剔除的特征列表"""
        return self._dropped_features
