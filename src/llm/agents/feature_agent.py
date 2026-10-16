"""
特征工程建议 Agent
==================
基于数据集统计信息，让 LLM 提出特征工程建议。

要点:
    展示 LLM 作为"数据科学助手"的能力——
    给定字段和样本，LLM 能基于业务理解给出衍生特征建议，
    辅助人类工程师拓展思路。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pandas as pd

from src.llm.client import DeepSeekClient
from src.llm.prompts import PromptManager
from src.utils.logger import get_logger

logger = get_logger("llm.feature_agent")


# 字段说明（教学示例：实际项目中应从数据字典文件加载）
FIELD_DESCRIPTIONS = {
    "loanAmnt": "贷款金额（元）",
    "term": "贷款期限（年）",
    "interestRate": "贷款利率（%）",
    "installment": "分期付款金额（元）",
    "grade": "贷款等级（A-G）",
    "subGrade": "贷款等级子级",
    "employmentTitle": "就业职称（脱敏）",
    "employmentLength": "就业年限",
    "homeOwnership": "房屋所有权状况",
    "annualIncome": "年收入（元）",
    "verificationStatus": "验证状态",
    "issueDate": "贷款发放日期",
    "purpose": "贷款用途",
    "postCode": "邮编（脱敏）",
    "regionCode": "地区代码",
    "dti": "负债收入比（%）",
    "delinquency_2years": "近 2 年逾期次数",
    "ficoRangeLow": "FICO 评分下限",
    "ficoRangeHigh": "FICO 评分上限",
    "openAcc": "未结信用额度数",
    "pubRec": "公开记录数",
    "pubRecBankruptcies": "破产记录数",
    "revolBal": "循环余额（元）",
    "revolUtil": "循环额度利用率（%）",
    "totalAcc": "总信用额度数",
    "initialListStatus": "初始上市状态",
    "applicationType": "申请类型",
    "earliesCreditLine": "最早信用额度日期",
    "title": "贷款名称（脱敏）",
    "policyCode": "策略代码",
}


class FeatureAgent:
    """
    特征工程建议 Agent。

    使用示例:
        >>> agent = FeatureAgent(client, prompt_mgr, agent_config)
        >>> suggestions = agent.suggest(df=train_df, current_auc=0.74)
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

    def suggest(
        self,
        df: pd.DataFrame,
        current_auc: float = 0.7,
        current_model: str = "XGBoost",
        bottleneck: str = None,
        source_name: str = "未指定数据源",
        existing_features: List[str] | None = None,
        target_summary: str = "未提供",
    ) -> Dict[str, Any]:
        """
        生成特征工程建议。

        参数:
            df: 训练集 DataFrame
            current_auc: 当前模型 AUC
            current_model: 当前模型名称
            bottleneck: 主要瓶颈描述（可选）

        返回:
            {"suggestions": [...], "general_advice": "..."}
        """
        logger.info(f"生成特征建议 - 数据规模: {df.shape}")

        # 构造上下文
        context = self._build_context(
            df,
            current_auc,
            current_model,
            bottleneck,
            source_name,
            existing_features or [],
            target_summary,
        )

        # 渲染 Prompt
        prompt = self.prompts.render("feature_suggestion", **context)
        system_prompt = self.prompts.get_system_role("feature_suggestion")

        # 调用 LLM
        response = self.client.chat(
            prompt,
            system_prompt=system_prompt,
            reasoning_effort="none",
        )
        if not response or not response.strip():
            raise RuntimeError("LLM 返回正文为空，未生成特征建议")

        # 解析 JSON 响应
        return self._parse_response(response)

    def apply_suggestion(
        self,
        df: pd.DataFrame,
        suggestion: Dict,
    ) -> pd.DataFrame:
        """
        应用单条特征建议。

        ⚠️ 安全提示：执行 LLM 生成的代码有风险，
        实际项目中应人工审核后再执行。
        """
        code = suggestion.get("code", "")
        if not code:
            return df

        df = df.copy()
        try:
            # 受限的命名空间
            safe_globals = {"df": df, "pd": pd, "__builtins__": {}}
            exec(code, safe_globals)
            logger.info(f"已应用特征建议: {suggestion.get('name')}")
            return safe_globals["df"]
        except Exception as e:
            logger.error(f"应用特征建议失败: {e}")
            return df

    # -------- 内部辅助 --------

    def _build_context(
        self,
        df: pd.DataFrame,
        current_auc: float,
        current_model: str,
        bottleneck: str,
        source_name: str,
        existing_features: List[str],
        target_summary: str,
    ) -> Dict[str, Any]:
        """构造 Prompt 上下文"""
        sample_size = min(self.config.sample_size, len(df))

        # 字段信息
        fields = []
        for col in df.columns:
            field_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "description": FIELD_DESCRIPTIONS.get(col, "（未知字段）"),
                "missing_count": int(df[col].isna().sum()),
                "missing_ratio": float(df[col].isna().mean()),
                "unique_count": int(df[col].nunique(dropna=True)),
            }
            # 数值列增加统计信息
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if not non_null.empty:
                    stats = non_null.describe()
                    field_info["stats"] = (
                        f"均值={stats['mean']:.2f}, "
                        f"中位数={stats['50%']:.2f}, "
                        f"标准差={stats['std']:.2f}"
                    )
            fields.append(field_info)

        # 样本数据（仅前 5 行用于上下文）
        data_sample = df.head(5).to_string(max_cols=10)

        return {
            "sample_size": sample_size,
            "feature_count": df.shape[1],
            "fields": fields,
            "data_sample": data_sample,
            "current_model": current_model,
            "current_auc": current_auc,
            "bottleneck": bottleneck,
            "source_name": source_name,
            "existing_features": existing_features,
            "target_summary": target_summary,
            "max_suggestions": self.config.max_suggestions,
        }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 提取 JSON 代码块
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}")

        # 提取首个完整 JSON 对象
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("无法解析 LLM 响应为 JSON，返回原始文本")
        return {
            "suggestions": [],
            "general_advice": response,
            "raw_response": response,
        }
