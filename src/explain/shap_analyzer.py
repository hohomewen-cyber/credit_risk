"""
SHAP 模型解释
=============
基于 SHAP（SHapley Additive exPlanations）的模型解释。

要点:
    SHAP 是基于博弈论的特征归因方法，
    可以同时支持全局解释（特征重要性）和
    局部解释（单个样本的特征贡献）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

logger = get_logger("explain.shap")


@dataclass
class SHAPExplanation:
    """SHAP 解释结果"""
    base_value: float
    shap_values: np.ndarray  # (n_features,)
    feature_values: np.ndarray  # (n_features,)
    feature_names: List[str]
    prediction: float

    def get_top_features(self, top_n: int = 5) -> List[Dict]:
        """获取贡献度最大的 Top N 特征"""
        abs_shap = np.abs(self.shap_values)
        top_idx = np.argsort(-abs_shap)[:top_n]

        return [
            {
                "feature": self.feature_names[i],
                "value": self._format_value(self.feature_values[i]),
                "shap_value": float(self.shap_values[i]),
                "direction": "↑ 增加风险" if self.shap_values[i] > 0 else "↓ 降低风险",
            }
            for i in top_idx
        ]

    @staticmethod
    def _format_value(value):
        """格式化特征值为可读形式"""
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return round(float(value), 4)
        return str(value)


class SHAPAnalyzer:
    """
    SHAP 分析器，封装 TreeExplainer / DeepExplainer / KernelExplainer。

    使用示例:
        >>> analyzer = SHAPAnalyzer()
        >>> analyzer.fit(xgb_model.model, X_train.sample(1000))
        >>> # 全局解释
        >>> global_importance = analyzer.global_importance(X_test)
        >>> # 局部解释
        >>> explanation = analyzer.explain_single(X_test.iloc[0])
    """

    def __init__(self, max_background_size: int = 1000):
        self.max_background_size = max_background_size
        self.explainer: Optional[Any] = None
        self.background_data: Optional[pd.DataFrame] = None
        self.feature_names: Optional[List[str]] = None
        self.expected_value: Optional[float] = None

    def fit(
        self,
        model: Any,
        background_data: pd.DataFrame,
        explainer_type: str = "auto",
    ) -> "SHAPAnalyzer":
        """
        构建 explainer。

        参数:
            model: 训练好的模型对象（XGBoost、PyTorch 等）
            background_data: 背景数据集，用于计算期望值
            explainer_type: "auto" / "tree" / "kernel" / "deep"
        """
        # 抽样背景数据，加速计算
        if len(background_data) > self.max_background_size:
            background_data = background_data.sample(
                n=self.max_background_size, random_state=42
            )

        self.background_data = background_data
        self.feature_names = background_data.columns.tolist()

        # 自动识别 explainer 类型
        if explainer_type == "auto":
            explainer_type = self._auto_select_explainer(model)

        logger.info(f"使用 {explainer_type} explainer")

        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif explainer_type == "kernel":
            self.explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model,
                background_data,
            )
        elif explainer_type == "deep":
            # PyTorch 模型需要传入 torch.Tensor
            import torch
            background_tensor = torch.FloatTensor(background_data.values)
            self.explainer = shap.DeepExplainer(model, background_tensor)
        else:
            raise ValueError(f"不支持的 explainer 类型: {explainer_type}")

        # 缓存期望值
        self.expected_value = self._get_expected_value()
        logger.info(f"SHAP explainer 已就绪 - 期望值: {self.expected_value:.4f}")
        return self

    def explain_single(self, sample: pd.Series) -> SHAPExplanation:
        """解释单个样本"""
        if self.explainer is None:
            raise RuntimeError("请先调用 fit()")

        # 转为 DataFrame（SHAP 需要 2D 输入）
        if isinstance(sample, pd.Series):
            sample_df = sample.to_frame().T
        else:
            sample_df = sample

        # 计算 SHAP 值
        shap_values = self.explainer.shap_values(sample_df)

        # 处理多分类情况（取正类）
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # 提取第一个样本的 SHAP 值
        shap_array = shap_values[0] if shap_values.ndim == 2 else shap_values

        # 预测值
        prediction = self.expected_value + shap_array.sum()

        return SHAPExplanation(
            base_value=float(self.expected_value),
            shap_values=shap_array,
            feature_values=sample_df.iloc[0].values,
            feature_names=self.feature_names,
            prediction=float(prediction),
        )

    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """批量计算 SHAP 值"""
        if self.explainer is None:
            raise RuntimeError("请先调用 fit()")

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        return shap_values

    def global_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.Series:
        """
        计算全局特征重要性。

        SHAP 全局重要性 = 所有样本 SHAP 值绝对值的均值
        """
        shap_values = self.explain_batch(X)
        importance = np.abs(shap_values).mean(axis=0)

        result = pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return result.head(top_n)

    def _auto_select_explainer(self, model: Any) -> str:
        """根据模型类型自动选择 explainer"""
        model_class_name = model.__class__.__name__.lower()

        # 树模型
        if any(k in model_class_name for k in ["xgb", "lgbm", "cat", "tree", "forest", "gbm"]):
            return "tree"
        # PyTorch 模型
        if "module" in str(type(model).__mro__).lower() or hasattr(model, "forward"):
            return "deep"
        # 其他情况退化为 KernelExplainer
        return "kernel"

    def _get_expected_value(self) -> float:
        """获取模型基线期望值"""
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = ev[1] if len(ev) > 1 else ev[0]
        return float(ev)
