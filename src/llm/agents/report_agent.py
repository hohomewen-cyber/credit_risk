"""
风险报告生成 Agent
==================
基于模型预测和 SHAP 解释，调用 LLM 生成自然语言风险报告。

这是 "LLM + ML 协同" 的典型场景：
    - ML 负责精确的概率预测
    - SHAP 负责量化特征贡献
    - LLM 负责将技术结果翻译为业务语言
"""
from __future__ import annotations

from typing import Any, Dict, List, Generator

import numpy as np
import pandas as pd

from src.llm.client import DeepSeekClient
from src.llm.prompts import PromptManager
from src.utils.logger import get_logger

logger = get_logger("llm.report_agent")


class ReportAgent:
    """
    风险报告生成 Agent。

    使用示例:
        >>> agent = ReportAgent(client, prompt_mgr, agent_config)
        >>> report = agent.generate(
        ...     features=features_dict,
        ...     probability=0.75,
        ...     shap_explanation=explanation,
        ... )
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
        self.top_n = config.top_n_features

    def generate(
        self,
        features: Dict[str, Any],
        probability: float,
        shap_explanation,
        stream: bool = False,
        model_name: str = "XGBoost + MLP 融合模型",
    ):
        """
        生成风险报告。

        参数:
            features: 客户特征字典
            probability: 违约概率
            shap_explanation: SHAPExplanation 对象
            stream: 是否流式输出

        返回:
            报告文本（stream=False）或字符串生成器（stream=True）
        """
        # 构造输入上下文
        context = self._build_context(
            features, probability, shap_explanation, model_name=model_name
        )

        # 渲染 Prompt
        prompt = self.prompts.render("risk_report", **context)
        system_prompt = self.prompts.get_system_role("risk_report")

        logger.info(f"生成风险报告 - 违约概率={probability:.4f}")

        if stream:
            return self.client.stream_chat(
                prompt, system_prompt=system_prompt, reasoning_effort="none"
            )
        return self.client.chat(
            prompt, system_prompt=system_prompt, reasoning_effort="none"
        )

    def _build_context(
        self,
        features: Dict[str, Any],
        probability: float,
        shap_explanation,
        model_name: str = "XGBoost + MLP 融合模型",
    ) -> Dict[str, Any]:
        """构造模板渲染所需的上下文"""
        # 风险等级
        if probability < 0.3:
            risk_level = "低风险"
        elif probability < 0.7:
            risk_level = "中风险"
        else:
            risk_level = "高风险"

        # Top N 特征
        top_features = shap_explanation.get_top_features(self.top_n)

        # 简化特征展示（仅展示有业务含义的关键字段）
        display_features = self._select_display_features(features)

        return {
            "features": display_features,
            "probability": probability,
            "risk_level": risk_level,
            "shap_top_features": top_features,
            "top_n": self.top_n,
            "model_name": model_name,
        }

    @staticmethod
    def _select_display_features(features: Dict) -> Dict:
        """从所有特征中筛选出业务含义明确的字段用于展示"""
        # 优先展示这些字段
        priority_fields = [
            "loanAmnt", "term", "interestRate", "installment",
            "annualIncome", "dti", "employmentLength", "homeOwnership",
            "purpose", "grade", "ficoRangeLow", "ficoRangeHigh",
            "delinquency_2years", "openAcc", "totalAcc", "revolUtil",
        ]

        # 中文标签映射
        labels = {
            "loanAmnt": "贷款金额",
            "term": "贷款期限(年)",
            "interestRate": "利率(%)",
            "installment": "月还款额",
            "annualIncome": "年收入",
            "dti": "负债收入比",
            "employmentLength": "就业年限",
            "homeOwnership": "房屋所有权",
            "purpose": "贷款用途",
            "grade": "贷款等级",
            "ficoRangeLow": "FICO 评分下限",
            "ficoRangeHigh": "FICO 评分上限",
            "delinquency_2years": "近2年逾期次数",
            "openAcc": "未结信用额度数",
            "totalAcc": "总信用额度数",
            "revolUtil": "循环额度利用率",
        }

        result = {}
        for field in priority_fields:
            if field in features:
                label = labels.get(field, field)
                value = features[field]
                # 数值美化
                if isinstance(value, float):
                    value = round(value, 2)
                result[label] = value

        return result
