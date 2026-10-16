"""
特征构造器
==========
基于配置文件生成衍生特征。

要点:
    "配置驱动 + 安全求值" 是工业界常用做法，
    避免业务规则散落在代码各处，便于维护和实验。
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("features.builder")


class FeatureBuilder:
    """
    根据配置生成衍生特征。

    使用示例:
        >>> builder = FeatureBuilder(feature_config)
        >>> df_with_new_features = builder.build(df)
    """

    def __init__(self, config):
        self.config = config
        self.derived_cfg = config.derived_features
        self._created_features: List[str] = []

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成所有配置中定义的衍生特征"""
        if not self.derived_cfg.enabled:
            return df

        df = df.copy()

        for rule in self.derived_cfg.rules:
            try:
                df = self._apply_rule(df, rule)
                self._created_features.append(rule["name"])
                logger.debug(f"创建衍生特征: {rule['name']}")
            except Exception as e:
                logger.warning(f"创建特征 {rule['name']} 失败: {e}")

        logger.info(f"成功创建 {len(self._created_features)} 个衍生特征")
        return df

    def _apply_rule(self, df: pd.DataFrame, rule: Dict) -> pd.DataFrame:
        """应用单条衍生规则"""
        name = rule["name"]
        formula = rule["formula"]

        # 跳过已存在的列
        if name in df.columns:
            logger.debug(f"特征 {name} 已存在，跳过")
            return df

        # 安全求值（仅允许特定函数和操作）
        df[name] = self._safe_eval(formula, df)
        return df

    @staticmethod
    def _safe_eval(formula: str, df: pd.DataFrame):
        """
        安全的公式求值。

        支持:
            - 列引用：df['col_name'] 或直接列名（自动包装）
            - 算术运算：+ - * /
            - pd 函数：pd.cut 等
            - np 函数：np.log, np.sqrt 等
        """
        # 受限的命名空间
        safe_globals = {
            "df": df,
            "pd": pd,
            "np": np,
            "__builtins__": {},
        }

        # 与文档声明的“直接列名（自动包装）”保持一致
        safe_locals = {col: df[col] for col in df.columns}

        # 自动将裸列名转为 df['col'] 形式（仅对配置中的字段有效）
        # 简化处理：直接 eval 公式（教学项目，假设配置可信）
        return eval(formula, safe_globals, safe_locals)

    def get_created_features(self) -> List[str]:
        """返回已创建的衍生特征列表"""
        return self._created_features.copy()
