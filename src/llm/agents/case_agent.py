"""
异常案例分析 Agent
==================
对模型预测错误（尤其是高置信度错误）的样本进行 AI 诊断。

要点:
    "高置信度错误" 是模型的盲点信号，
    通过 LLM 分析这些样本能快速发现：
    - 数据质量问题
    - 特征覆盖不足
    - 业务规则漏洞
    - 模型偏差
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.llm.client import DeepSeekClient
from src.llm.prompts import PromptManager
from src.utils.logger import get_logger

logger = get_logger("llm.case_agent")


class CaseAgent:
    """
    异常案例分析 Agent。

    使用示例:
        >>> agent = CaseAgent(client, prompt_mgr, agent_config)
        >>> # 找出高置信度的预测错误
        >>> bad_cases = agent.find_anomalies(y_true, y_pred_proba, X_test)
        >>> # 分析单个案例
        >>> diagnosis = agent.analyze(case_features, true_label, pred_proba, shap_explanation)
    """

    def __init__(
        self,
        client: DeepSeekClient,
        prompt_manager: PromptManager,
        config,
    ):
        self.client = client
        self.prompts = prompt_manager
        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.max_cases = config.max_cases

    def find_anomalies(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        筛选出高置信度但预测错误的样本。

        参数:
            y_true: 真实标签 (n,)
            y_pred_proba: 预测概率 (n,)
            X: 特征 DataFrame

        返回:
            含 ['index', 'true_label', 'pred_proba', 'error_type'] 的 DataFrame
        """
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        threshold = 0.5
        y_pred = (y_pred_proba >= threshold).astype(int)

        # 错误样本掩码
        wrong_mask = (y_pred != y_true)

        # 高置信度（接近 0 或 1）
        confidence = np.abs(y_pred_proba - 0.5) * 2  # 归一化到 [0, 1]
        high_confidence_mask = confidence >= self.confidence_threshold

        # 高置信度且预测错误
        anomaly_mask = wrong_mask & high_confidence_mask

        if not anomaly_mask.any():
            logger.info("未发现高置信度的错误案例")
            return pd.DataFrame()

        # 错误类型
        error_types = np.where(
            (y_true == 0) & (y_pred == 1),
            "False Positive (误判违约)",
            np.where(
                (y_true == 1) & (y_pred == 0),
                "False Negative (漏判违约)",
                "其他",
            ),
        )

        result = pd.DataFrame({
            "index": np.where(anomaly_mask)[0],
            "true_label": y_true[anomaly_mask],
            "pred_proba": y_pred_proba[anomaly_mask],
            "confidence": confidence[anomaly_mask],
            "error_type": error_types[anomaly_mask],
        })

        # 按置信度排序，只保留 Top N
        result = result.sort_values("confidence", ascending=False).head(self.max_cases)
        logger.info(f"发现 {len(result)} 个高置信度错误案例")
        return result.reset_index(drop=True)

    def analyze(
        self,
        features: Dict[str, Any],
        true_label: int,
        probability: float,
        shap_explanation,
        stream: bool = False,
    ):
        """
        分析单个异常案例。

        参数:
            features: 样本特征
            true_label: 真实标签
            probability: 预测概率
            shap_explanation: SHAPExplanation 对象
            stream: 是否流式输出
        """
        # 构造上下文
        context = self._build_context(features, true_label, probability, shap_explanation)

        # 渲染 Prompt
        prompt = self.prompts.render("case_analysis", **context)
        system_prompt = self.prompts.get_system_role("case_analysis")

        logger.info(
            f"分析异常案例 - 真实={true_label}, 预测概率={probability:.4f}"
        )

        if stream:
            return self.client.stream_chat(prompt, system_prompt=system_prompt)
        return self.client.chat(prompt, system_prompt=system_prompt)

    def _build_context(
        self,
        features: Dict,
        true_label: int,
        probability: float,
        shap_explanation,
    ) -> Dict[str, Any]:
        """构造模板上下文"""
        # 标签文本
        true_label_text = "违约" if true_label == 1 else "正常"

        predicted_label = 1 if probability >= 0.5 else 0
        predicted_label_text = "违约" if predicted_label == 1 else "正常"

        # 置信度等级
        confidence = abs(probability - 0.5) * 2
        if confidence >= 0.85:
            confidence_level = f"非常高 ({confidence:.2%})"
        elif confidence >= 0.6:
            confidence_level = f"较高 ({confidence:.2%})"
        else:
            confidence_level = f"一般 ({confidence:.2%})"

        # SHAP Top 5 特征
        shap_features = shap_explanation.get_top_features(top_n=5)

        # 关键特征（仅展示业务可解释字段）
        display_features = self._select_display_features(features)

        return {
            "features": display_features,
            "true_label": true_label,
            "true_label_text": true_label_text,
            "predicted_label": predicted_label,
            "predicted_label_text": predicted_label_text,
            "probability": probability,
            "confidence_level": confidence_level,
            "shap_features": shap_features,
        }

    @staticmethod
    def _select_display_features(features: Dict) -> Dict:
        """筛选要展示的关键字段"""
        priority = [
            "loanAmnt", "term", "interestRate", "annualIncome",
            "dti", "employmentLength", "homeOwnership", "purpose",
            "grade", "ficoRangeLow", "delinquency_2years", "revolUtil",
        ]
        return {k: features[k] for k in priority if k in features}
