"""
数据预处理器
============
负责缺失值填充、类别编码、数值标准化、异常值处理等基础预处理。

要点:
    1. fit/transform 分离设计（参考 sklearn 风格）
    2. 状态可保存：训练集 fit 的统计量保留下来用于推理
    3. 流水线友好：每个步骤独立可测试
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

from src.utils.logger import get_logger

logger = get_logger("data.preprocessor")


class Preprocessor:
    """
    数据预处理器。

    使用示例:
        >>> prep = Preprocessor(data_cfg, feature_cfg)
        >>> X_train_clean = prep.fit_transform(X_train)
        >>> X_test_clean = prep.transform(X_test)
    """

    def __init__(self, data_config, feature_config):
        self.data_cfg = data_config
        self.feature_cfg = feature_config

        # 状态：fit 时学到的参数
        self._numerical_imputers: Dict[str, float] = {}
        self._categorical_imputers: Dict[str, str] = {}
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scaler = None
        self._dropped_cols: list = []
        self._outlier_clips: Dict[str, tuple] = {}
        self._knn_imputers: Dict[str, tuple] = {}

        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合 + 转换"""
        return self.fit(df).transform(df)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """从训练集学习预处理参数"""
        logger.info(f"开始拟合预处理器 - 输入 shape: {df.shape}")
        df = df.copy()

        # 1. 移除高缺失率列
        self._fit_drop_high_missing(df)
        df = df.drop(columns=self._dropped_cols, errors="ignore")

        # 2. 学习缺失值填充值
        self._fit_imputers(df)

        # 3. 学习类别编码器
        self._fit_encoders(df)

        # 4. 学习异常值截断范围
        self._fit_outlier_clips(df)

        self._is_fitted = True
        logger.info(f"预处理器拟合完成 - 移除 {len(self._dropped_cols)} 列")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用学到的参数对数据进行转换"""
        if not self._is_fitted:
            raise RuntimeError("预处理器未拟合，请先调用 fit() 或 fit_transform()")

        df = df.copy()

        # 1. 移除高缺失列
        df = df.drop(columns=self._dropped_cols, errors="ignore")

        # 2. 缺失值填充
        df = self._apply_imputers(df)

        # 3. 异常值截断
        df = self._apply_outlier_clips(df)

        # 4. 类别编码
        df = self._apply_encoders(df)

        # 5. 时间特征处理（简化版）
        df = self._handle_datetime(df)

        # 6. 数值标准化
        df = self._apply_scaler(df)

        logger.info(f"transform 完成 - 输出 shape: {df.shape}")
        return df

    # -------------- 私有方法（按步骤拆分） --------------

    def _fit_drop_high_missing(self, df: pd.DataFrame) -> None:
        """识别高缺失率列"""
        threshold = self.feature_cfg.missing_value.drop_threshold
        missing_ratio = df.isnull().mean()
        self._dropped_cols = missing_ratio[missing_ratio > threshold].index.tolist()
        # 显式配置为 KNN 的列需保留，全空时由 KNN 边界逻辑回退到 0。
        strategies = self.feature_cfg.missing_value.get("strategy", {})
        knn_cols = {col for col, strategy in strategies.items() if strategy == "knn"}
        self._dropped_cols = [c for c in self._dropped_cols if c not in knn_cols]
        if self._dropped_cols:
            logger.info(f"高缺失率列将被移除: {self._dropped_cols}")

    def _fit_imputers(self, df: pd.DataFrame) -> None:
        """学习缺失值填充策略"""
        num_strategy = self.feature_cfg.missing_value.numerical_strategy
        cat_strategy = self.feature_cfg.missing_value.categorical_strategy
        column_strategies = self.feature_cfg.missing_value.get("strategy", {})

        # 数值列
        num_cols = [c for c in self.data_cfg.columns.numerical if c in df.columns]
        num_cols += [c for c in self.data_cfg.columns.anonymous if c in df.columns]
        for col in num_cols:
            strategy = column_strategies.get(col, num_strategy)
            if strategy == "knn":
                self._fit_knn_imputer(df, col, num_cols)
            elif strategy == "mean":
                self._numerical_imputers[col] = df[col].mean()
            elif strategy == "median":
                self._numerical_imputers[col] = df[col].median()
            elif strategy == "zero":
                self._numerical_imputers[col] = 0
            else:
                self._numerical_imputers[col] = df[col].median()

        # 类别列
        cat_cols = [c for c in self.data_cfg.columns.categorical if c in df.columns]
        for col in cat_cols:
            strategy = column_strategies.get(col, cat_strategy)
            if strategy == "mode":
                mode = df[col].mode()
                self._categorical_imputers[col] = mode.iloc[0] if len(mode) > 0 else "unknown"
            else:
                self._categorical_imputers[col] = "unknown"

    def _apply_imputers(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用缺失值填充"""
        for col, fill_value in self._numerical_imputers.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)

        for col, fill_value in self._categorical_imputers.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value).astype(str)

        for col, (imputer, input_cols) in self._knn_imputers.items():
            if col not in df.columns:
                continue
            knn_input = df.reindex(columns=input_cols)
            transformed = imputer.transform(knn_input)
            col_idx = input_cols.index(col)
            df[col] = transformed[:, col_idx]

        return df

    def _fit_knn_imputer(self, df: pd.DataFrame, col: str, num_cols: list) -> None:
        """为单个指定列拟合 KNN 填充器，其他数值列作为距离参考。"""
        if df[col].notna().sum() == 0:
            # KNN 无法从全空目标列学习，边界情况使用确定性 0 回退。
            self._numerical_imputers[col] = 0.0
            return

        input_cols = [
            c for c in num_cols
            if c in df.columns and (c == col or df[c].notna().any())
        ]
        if col not in input_cols:
            input_cols.append(col)
        n_neighbors = int(self.feature_cfg.missing_value.get("knn_neighbors", 5))
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputer.fit(df[input_cols])
        self._knn_imputers[col] = (imputer, input_cols)

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        """学习类别编码器"""
        cat_cols = [c for c in self.data_cfg.columns.categorical if c in df.columns]
        custom_mappings = self.feature_cfg.encoding.get("custom", {})

        for col in cat_cols:
            # 优先使用配置文件中的自定义映射
            if col in custom_mappings:
                # 自定义映射不需要 fit
                continue

            le = LabelEncoder()
            # 兼容缺失值（先填充）
            values = df[col].fillna("unknown").astype(str)
            le.fit(values)
            self._label_encoders[col] = le

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用类别编码"""
        custom_mappings = self.feature_cfg.encoding.get("custom", {})

        # 先应用自定义映射
        for col, cfg in custom_mappings.items():
            if col in df.columns and "mapping" in cfg:
                mapping = cfg["mapping"]
                # 未在映射中的值用 -1 表示
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        # 再应用学到的 LabelEncoder
        for col, le in self._label_encoders.items():
            if col in df.columns:
                values = df[col].fillna("unknown").astype(str)
                # 未见过的类别 -> 标记为 -1
                known = set(le.classes_)
                values = values.map(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(values)

        return df

    def _fit_outlier_clips(self, df: pd.DataFrame) -> None:
        """学习异常值截断阈值"""
        outlier_cfg = self.feature_cfg.outlier
        if outlier_cfg.method != "clip":
            return

        for col in outlier_cfg.apply_to:
            if col in df.columns:
                lower = df[col].quantile(outlier_cfg.lower_quantile)
                upper = df[col].quantile(outlier_cfg.upper_quantile)
                self._outlier_clips[col] = (lower, upper)

    def _apply_outlier_clips(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用异常值截断"""
        for col, (lower, upper) in self._outlier_clips.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _handle_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """简化版时间特征处理"""
        for col in self.data_cfg.columns.datetime:
            if col not in df.columns:
                continue
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
                df[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
                df = df.drop(columns=[col])
            except Exception as e:
                logger.warning(f"时间列 {col} 处理失败: {e}")
                df = df.drop(columns=[col], errors="ignore")
        return df

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """数值标准化（在所有处理后统一应用）"""
        method = self.feature_cfg.scaling.method
        if method == "none":
            return df

        # 仅对数值列（含衍生数值）做标准化
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df

        if self._scaler is None:
            if method == "standard":
                self._scaler = StandardScaler()
            elif method == "minmax":
                self._scaler = MinMaxScaler()
            elif method == "robust":
                self._scaler = RobustScaler()
            else:
                return df

            df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols])
        else:
            # 对齐列
            scaler_cols = [c for c in numeric_cols if hasattr(self._scaler, "mean_")]
            if scaler_cols:
                df[scaler_cols] = self._scaler.transform(df[scaler_cols])

        return df
