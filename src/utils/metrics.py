"""
评估指标模块
============
风控场景常用的评估指标：AUC、KS、混淆矩阵、PR/ROC 曲线等。

要点:
    KS（Kolmogorov-Smirnov）是金融风控特有的指标，
    衡量好坏样本累积分布的最大差距，越大越好。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


@dataclass
class EvalResult:
    """评估结果数据类"""
    auc: float = 0.0
    ks: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    log_loss: float = 0.0
    threshold: float = 0.5

    def to_dict(self) -> Dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"EvalResult(AUC={self.auc:.4f}, KS={self.ks:.4f}, "
            f"F1={self.f1:.4f} @ threshold={self.threshold:.2f})"
        )


class Evaluator:
    """
    模型评估器，统一计算各类指标。

    使用示例:
        >>> evaluator = Evaluator()
        >>> result = evaluator.evaluate(y_true, y_pred_proba)
        >>> print(result)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = None,
    ) -> EvalResult:
        """
        计算所有指标。

        参数:
            y_true: 真实标签 (n,)
            y_pred_proba: 预测概率 (n,) 或 (n, 2)
            threshold: 分类阈值
        """
        # 兼容二维概率输入
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        threshold = threshold if threshold is not None else self.threshold
        y_pred_label = (y_pred_proba >= threshold).astype(int)

        return EvalResult(
            auc=roc_auc_score(y_true, y_pred_proba),
            ks=self.calculate_ks(y_true, y_pred_proba),
            accuracy=accuracy_score(y_true, y_pred_label),
            precision=precision_score(y_true, y_pred_label, zero_division=0),
            recall=recall_score(y_true, y_pred_label, zero_division=0),
            f1=f1_score(y_true, y_pred_label, zero_division=0),
            log_loss=log_loss(y_true, np.clip(y_pred_proba, 1e-15, 1 - 1e-15)),
            threshold=threshold,
        )

    @staticmethod
    def calculate_ks(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        计算 KS 值（金融风控特有指标）。

        KS = max(|TPR - FPR|)
        反映模型对好坏客户的区分能力，越大越好。
        通常 KS > 0.3 算可接受，> 0.5 算优秀。
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        return float(np.max(np.abs(tpr - fpr)))

    @staticmethod
    def find_best_threshold(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        method: str = "youden",
    ) -> float:
        """
        寻找最佳分类阈值。

        参数:
            method: youden（Youden 指数）/ f1（F1 最大）
        """
        if method == "youden":
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            youden = tpr - fpr
            return float(thresholds[np.argmax(youden)])

        elif method == "f1":
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
            # 注意 thresholds 比 precisions 少一个
            return float(thresholds[np.argmax(f1_scores[:-1])])

        else:
            raise ValueError(f"不支持的方法: {method}")

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """获取混淆矩阵"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取 ROC 曲线数据点"""
        return roc_curve(y_true, y_pred_proba)

    @staticmethod
    def get_pr_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取 PR 曲线数据点"""
        return precision_recall_curve(y_true, y_pred_proba)

    @staticmethod
    def calculate_lift(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        """
        计算 Lift（提升度），用于评估营销/风控分群效果。
        """
        # 按预测概率降序排列
        sorted_indices = np.argsort(-y_pred_proba)
        y_sorted = y_true[sorted_indices]

        baseline = y_true.mean()
        bin_size = len(y_true) // n_bins

        lifts = []
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
            bin_rate = y_sorted[start:end].mean()
            lift = bin_rate / baseline if baseline > 0 else 0
            lifts.append({
                "bin": i + 1,
                "size": end - start,
                "default_rate": float(bin_rate),
                "lift": float(lift),
            })

        return {"baseline": float(baseline), "bins": lifts}
