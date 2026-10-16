"""A/B 预测日志与效果分析。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class ABTestLogger:
    """持久化真实路由结果，并在有真实终态标签时计算效果指标。"""

    COLUMNS = [
        "timestamp", "user_id", "data_source", "branch", "model_version",
        "prediction", "eventual_outcome",
    ]

    def __init__(self, log_path):
        self.log_path = Path(log_path)

    def log(
        self,
        user_id,
        branch: str,
        model_version: str,
        prediction: float,
        eventual_outcome: Optional[int] = None,
        data_source: Optional[str] = None,
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_id": str(user_id),
            "data_source": data_source,
            "branch": branch,
            "model_version": model_version,
            "prediction": float(prediction),
            "eventual_outcome": (
                int(eventual_outcome) if eventual_outcome is not None else None
            ),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        records = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue
        return pd.DataFrame(records, columns=self.COLUMNS)

    def analyze(
        self, data_source: Optional[str] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """返回分组预测统计和仅基于真实终态标签的效果指标。"""
        records = self.load()
        if data_source is not None and not records.empty:
            records = records.loc[records["data_source"] == data_source].copy()
        if records.empty:
            return pd.DataFrame(), pd.DataFrame()

        summary = (
            records.groupby(["branch", "model_version"], dropna=False)
            .agg(
                样本量=("prediction", "size"),
                平均预测概率=("prediction", "mean"),
                最低预测概率=("prediction", "min"),
                最高预测概率=("prediction", "max"),
                已回收真实标签=("eventual_outcome", "count"),
            )
            .reset_index()
            .rename(columns={"branch": "分组", "model_version": "模型"})
        )

        metric_rows = []
        labeled = records.dropna(subset=["eventual_outcome"]).copy()
        for (branch, model), group in labeled.groupby(
            ["branch", "model_version"], dropna=False
        ):
            y_true = group["eventual_outcome"].astype(int).to_numpy()
            y_score = group["prediction"].astype(float).to_numpy()
            if len(group) < 2 or np.unique(y_true).size < 2:
                continue
            from src.utils.metrics import Evaluator

            result = Evaluator().evaluate(y_true, y_score)
            metric_rows.append({
                "分组": branch,
                "模型": model,
                "真实标签数": len(group),
                "AUC": result.auc,
                "KS": result.ks,
                "F1": result.f1,
                "Accuracy": result.accuracy,
            })
        return summary, pd.DataFrame(metric_rows)
