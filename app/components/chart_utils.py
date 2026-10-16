"""
图表工具组件
============
封装常用的 Plotly 图表。
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def gauge_chart(value: float, title: str = "违约概率", max_value: float = 100) -> go.Figure:
    """风险仪表盘"""
    color = "#dc3545" if value > 70 else ("#ffc107" if value > 30 else "#28a745")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, max_value], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "lightgray",
            "steps": [
                {"range": [0, 30], "color": "#d4edda"},
                {"range": [30, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def confusion_matrix_heatmap(cm: np.ndarray, labels: List[str] = None) -> go.Figure:
    """混淆矩阵热力图"""
    labels = labels or ["正常", "违约"]

    fig = px.imshow(
        cm,
        x=labels, y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="预测", y="真实", color="数量"),
    )
    fig.update_layout(
        title="混淆矩阵",
        height=400,
        xaxis=dict(side="top"),
    )
    return fig


def roc_curve_plot(fpr, tpr, auc: float = 0.0, title: str = "ROC 曲线") -> go.Figure:
    """ROC 曲线"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC (AUC={auc:.4f})",
        line=dict(color="#0066cc", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        name="随机分类",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="假正例率 (FPR)",
        yaxis_title="真正例率 (TPR)",
        height=400,
        hovermode="x",
    )
    return fig


def pr_curve_plot(precision, recall, title: str = "PR 曲线") -> go.Figure:
    """Precision-Recall 曲线"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        name="PR 曲线",
        line=dict(color="#dc3545", width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="召回率 (Recall)",
        yaxis_title="精确率 (Precision)",
        height=400,
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 1.05]),
    )
    return fig


def feature_importance_bar(importance: pd.Series, top_n: int = 20, title: str = "特征重要性") -> go.Figure:
    """特征重要性柱状图"""
    top = importance.head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index,
        orientation="h",
        marker_color="#0066cc",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="重要性",
        yaxis_title="特征",
        height=max(400, top_n * 25),
    )
    return fig


def shap_waterfall(
    feature_names: List[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    base_value: float,
    top_n: int = 10,
) -> go.Figure:
    """SHAP 瀑布图（简化版）"""
    # 取绝对值最大的 Top N
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(-abs_shap)[:top_n]

    # 排序：负贡献在前，正贡献在后
    selected = sorted(top_idx, key=lambda i: shap_values[i])

    # 构建瀑布图数据
    measures = ["relative"] * len(selected)
    x_labels = [f"{feature_names[i]}={feature_values[i]:.2f}" for i in selected]
    y_values = [shap_values[i] for i in selected]

    # 添加 base 和 final
    measures = ["absolute"] + measures + ["total"]
    x_labels = [f"基线值 ({base_value:.3f})"] + x_labels + ["最终预测"]
    y_values = [base_value] + y_values + [None]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        textposition="outside",
        text=[f"{v:+.3f}" if v is not None else "" for v in y_values],
        connector={"line": {"color": "rgb(100, 100, 100)"}},
        increasing={"marker": {"color": "#dc3545"}},
        decreasing={"marker": {"color": "#28a745"}},
        totals={"marker": {"color": "#0066cc"}},
    ))
    fig.update_layout(
        title="SHAP 特征贡献（瀑布图）",
        height=500,
        showlegend=False,
        xaxis=dict(tickangle=-30),
    )
    return fig


def distribution_plot(df: pd.DataFrame, column: str, target_col: Optional[str] = None) -> go.Figure:
    """单特征分布图"""
    if target_col and target_col in df.columns and df[target_col].nunique() <= 5:
        fig = px.histogram(
            df, x=column, color=target_col,
            barmode="overlay",
            opacity=0.6,
            nbins=50,
            title=f"{column} 的分布（按 {target_col} 区分）",
        )
    else:
        fig = px.histogram(df, x=column, nbins=50, title=f"{column} 的分布")

    fig.update_layout(height=350)
    return fig


def correlation_heatmap(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """相关性热力图（仅取数值列）"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > top_n:
        # 仅显示与目标列相关性最高的 top_n 列
        numeric_df = numeric_df.iloc[:, :top_n]

    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="特征相关性矩阵",
    )
    fig.update_layout(height=600)
    return fig


def training_curve(train_loss: List, val_loss: List = None, val_metric: List = None) -> go.Figure:
    """训练曲线"""
    fig = go.Figure()

    epochs = list(range(1, len(train_loss) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines", name="train_loss"))

    if val_loss:
        fig.add_trace(go.Scatter(x=epochs[:len(val_loss)], y=val_loss, mode="lines", name="val_loss"))

    if val_metric:
        fig.add_trace(go.Scatter(
            x=epochs[:len(val_metric)], y=val_metric,
            mode="lines", name="val_auc",
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(title="AUC", overlaying="y", side="right"),
        )

    fig.update_layout(
        title="训练曲线",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400,
        hovermode="x unified",
    )
    return fig
