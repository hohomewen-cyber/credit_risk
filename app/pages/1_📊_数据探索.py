"""
📊 数据探索页面
================
EDA：数据概览、缺失值分析、目标分布、特征分布、相关性。
"""
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.chart_utils import distribution_plot, correlation_heatmap

st.set_page_config(page_title="数据探索", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 数据探索")
st.caption("理解你的数据是建模的第一步")


# ===== 数据加载 =====
from app.components.data_utils import (
    SOURCE_LABELS,
    available_data_sources,
    initialize_data_source_state,
    load_selected_data,
    render_analysis_context,
    set_active_data_source,
)


@st.cache_data
def load_uploaded_lending_club(file_bytes: bytes) -> pd.DataFrame:
    """让页面上传的 CSV 走与本地文件相同的 DataLoader/Adapter 流程。"""
    from src.data.loader import DataLoader
    from src.utils.config_loader import ConfigLoader

    loader = DataLoader(ConfigLoader(str(ROOT / "configs")).get("data"))
    return loader.load(BytesIO(file_bytes), source="lending_club")


# 探测可用的数据源
real_path = ROOT / "data" / "raw" / "train.csv"
sample_path = ROOT / "data" / "sample" / "sample.csv"
lending_path = ROOT / "data" / "raw" / "lending_club.csv"
real_available = real_path.exists()
sample_available = sample_path.exists()
lending_available = lending_path.exists()

with st.expander("🏦 Lending Club 数据源 Adapter", expanded=not lending_available):
    st.markdown(
        "可直接上传 Lending Club CSV；数据会通过 "
        "`DataLoader(source='lending_club')` 和 `LendingClubAdapter` "
        "映射为当前项目字段。也可以将文件保存到 "
        "`data/raw/lending_club.csv`。"
    )
    lending_upload = st.file_uploader(
        "上传 Lending Club CSV",
        type=["csv"],
        help="仅在当前页面会话中读取，不会覆盖项目现有数据。",
    )
    st.caption(
        f"固定路径状态：{'已检测到 lending_club.csv' if lending_available else '未检测到文件，可使用上方上传入口'}"
    )

uploaded_lending_df = st.session_state.get("uploaded_lending_df")
if lending_upload is not None:
    try:
        uploaded_lending_df = load_uploaded_lending_club(lending_upload.getvalue())
        st.session_state["uploaded_lending_df"] = uploaded_lending_df
        st.success(
            f"✅ Lending Club Adapter 映射成功："
            f"{len(uploaded_lending_df):,} 条数据，{uploaded_lending_df.shape[1]} 个字段"
        )
    except Exception as exc:
        st.error(f"❌ Lending Club 文件加载失败：{exc}")

if not real_available and not sample_available and uploaded_lending_df is None:
    st.error("❌ 未找到任何数据")
    st.info(
        "**方式 1**：从天池下载真实数据，放到 `data/raw/train.csv`\n\n"
        "**方式 2**：运行 `python scripts/generate_sample_data.py --n_samples 50000` 生成合成数据"
    )
    st.stop()

# ===== 全局数据源选择 =====
sources = available_data_sources(ROOT)
active_source = initialize_data_source_state(ROOT)
source_keys = list(sources)

col1, col2 = st.columns([3, 1])
with col1:
    selected_source = st.radio(
    "选择全局数据源（将同步用于KNN、训练、预测、SHAP、KS和A/B）",
        source_keys,
        index=source_keys.index(active_source),
        format_func=lambda key: sources[key],
        horizontal=True,
    )
    if selected_source != active_source:
        set_active_data_source(ROOT, selected_source)
        active_source = selected_source
with col2:
    if st.button("🔄 重新加载", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 真实数据 80 万行太大，可视化时只读前 N 行
if active_source == "tianchi":
    nrows = st.slider(
        "读取行数（真实数据较大，建议先用部分数据探索）",
        min_value=10_000, max_value=800_000, value=100_000, step=10_000,
    )
else:
    nrows = None

df, source_label = load_selected_data(ROOT, active_source, nrows=nrows)

if df.empty:
    st.error("⚠️ 数据加载失败")
    st.stop()

st.success(f"✅ 已加载 {source_label}：{len(df):,} 条数据，{df.shape[1]} 个字段")
render_analysis_context(ROOT, "XGBoost / Random Forest（A/B）")

# ===== 当前数据源真实预测 =====
if len(df) > 0:
    st.markdown("## 🎯 当前数据源模型预测")
    st.caption(
        "当前选择的数据会经过 DataLoader（Lending Club额外经过Adapter）→ 原有预处理 → "
        "原有特征工程/筛选 → XGBoost / Random Forest 预测。"
    )
    max_batch = min(2_000, len(df))
    prediction_rows = st.slider(
        "本次预测样本数", 10, max_batch, min(200, max_batch), step=10
    )
    if st.button("🚀 对当前数据源执行预测", type="primary"):
        try:
            from app.components.data_utils import (
                load_prediction_resources,
                prepare_model_features,
            )
            from src.ab_test import ABTestLogger, ABTestRouter

            resources = load_prediction_resources(str(ROOT))
            if resources is None:
                st.error("模型产物不完整，请先在模型训练页面完成训练。")
            else:
                batch = df.head(prediction_rows).copy()
                features = prepare_model_features(batch, resources)
                xgb_probability = resources["xgb_model"].predict_proba(features)
                router = ABTestRouter(
                    resources["xgb_model"],
                    resources["random_forest_model"],
                    ratio=0.5,
                )
                ab_logger = ABTestLogger(ROOT / "logs" / "ab_test_predictions.jsonl")
                rows = []
                for position, (_, record) in enumerate(batch.iterrows()):
                    user_id = record.get("id", position)
                    branch, probability = router.route(
                        user_id, features.iloc[[position]]
                    )
                    model_name = "XGBoost" if branch == "A" else "Random Forest"
                    outcome = record.get("isDefault")
                    actual = None if pd.isna(outcome) else int(outcome)
                    routed_probability = float(probability[0])
                    ab_logger.log(
                        user_id=f"{active_source}:{user_id}",
                        branch=branch,
                        model_version=model_name,
                        prediction=routed_probability,
                        eventual_outcome=actual,
                        data_source=active_source,
                    )
                    rows.append({
                        "用户ID": user_id,
                        "分组": branch,
                        "模型": model_name,
                        "A/B预测概率": routed_probability,
                        "XGBoost预测概率": float(xgb_probability[position]),
                        "真实终态标签": actual,
                    })

                prediction_df = pd.DataFrame(rows)
                st.session_state["source_prediction_df"] = {
                    "source": active_source,
                    "data": prediction_df,
                }
                st.success(f"已完成 {len(prediction_df):,} 条真实数据预测并写入 A/B 日志。")
        except Exception as exc:
            st.error(f"Lending Club 预测失败：{exc}")

    source_prediction = st.session_state.get("source_prediction_df")
    if source_prediction and source_prediction.get("source") == active_source:
        prediction_df = source_prediction["data"]
        st.dataframe(prediction_df.head(100), use_container_width=True, hide_index=True)
        labeled_count = prediction_df["真实终态标签"].notna().sum()
        st.caption(
            f"本批次有 {labeled_count} 条真实终态标签；Current、逾期中和宽限期记录"
            "尚无最终结果，不会被伪装成正常或违约。"
        )

        from src.ab_test import ABTestLogger

        ab_summary, ab_metrics = ABTestLogger(
            ROOT / "logs" / "ab_test_predictions.jsonl"
        ).analyze(data_source=active_source)
        st.markdown("### A/B 运行统计（A = XGBoost，B = Random Forest）")
        if not ab_summary.empty:
            percentage_cols = ["平均预测概率", "最低预测概率", "最高预测概率"]
            display_summary = ab_summary.copy()
            for column in percentage_cols:
                display_summary[column] = display_summary[column].map(
                    lambda value: f"{value:.2%}"
                )
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
        if ab_metrics.empty:
            st.info("暂无足够的实际结果，不能进行 AUC、KS、F1、Accuracy 效果评价。")
        else:
            st.markdown("### 基于真实终态标签的 A/B 效果")
            st.dataframe(ab_metrics.round(4), use_container_width=True, hide_index=True)

# ===== Tab 1: 数据概览 =====
tab1, tab2, tab3, tab4 = st.tabs(["📋 数据概览", "🕳️ 缺失值分析", "🎯 目标分布", "📈 特征分析"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总样本数", f"{len(df):,}")
    col2.metric("特征数", df.shape[1])
    col3.metric("内存占用", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    if "isDefault" in df.columns:
        target_numeric = pd.to_numeric(df["isDefault"], errors="coerce")
        col4.metric(
            "终态违约率" if target_numeric.isna().any() else "违约率",
            f"{target_numeric.mean():.2%}" if target_numeric.notna().any() else "暂无终态标签",
        )

    st.markdown("### 数据样本（前 10 行）")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 字段类型分布")
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["类型", "数量"]
    dtype_counts["类型"] = dtype_counts["类型"].astype(str)
    st.dataframe(dtype_counts, use_container_width=True, hide_index=True)

    with st.expander("📊 数值特征统计"):
        st.dataframe(df.describe().T, use_container_width=True)

with tab2:
    st.markdown("### 缺失值统计")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("🎉 数据无缺失！")
    else:
        missing_df = pd.DataFrame({
            "字段": missing.index,
            "缺失数": missing.values,
            "缺失率": (missing.values / len(df) * 100).round(2),
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        # 可视化
        import plotly.express as px
        fig = px.bar(
            missing_df,
            x="字段",
            y="缺失率",
            title="各字段缺失率（%）",
            color="缺失率",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if "isDefault" not in df.columns:
        st.warning("数据中没有 isDefault 字段，无法分析目标分布")
    else:
        col1, col2 = st.columns(2)
        with col1:
            import plotly.express as px
            target_dist = df["isDefault"].value_counts().reset_index()
            target_dist.columns = ["isDefault", "数量"]
            target_dist["isDefault"] = target_dist["isDefault"].map({0: "正常", 1: "违约"})

            fig = px.pie(
                target_dist,
                values="数量",
                names="isDefault",
                title="目标变量分布",
                color_discrete_sequence=["#28a745", "#dc3545"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 类别不平衡分析")
            n_pos = (df["isDefault"] == 1).sum()
            n_neg = (df["isDefault"] == 0).sum()

            st.metric("正常样本数", f"{n_neg:,}")
            st.metric("违约样本数", f"{n_pos:,}")
            st.metric("不平衡比例", f"1 : {n_neg/n_pos:.2f}" if n_pos else "暂无违约终态")

            if n_pos and n_neg / n_pos > 3:
                st.warning(
                    "⚠️ 数据存在类别不平衡，建议在模型训练时启用：\n"
                    "- XGBoost: `scale_pos_weight`\n"
                    "- MLP: 加权损失函数 / Focal Loss"
                )

with tab4:
    st.markdown("### 单特征分布")

    col1, col2 = st.columns([1, 3])
    with col1:
        # 仅展示数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "isDefault" in numeric_cols:
            numeric_cols.remove("isDefault")
        if "id" in numeric_cols:
            numeric_cols.remove("id")

        selected_col = st.selectbox("选择特征", numeric_cols)

    with col2:
        if selected_col:
            plot_target = (
                "isDefault"
                if "isDefault" in df.columns and df["isDefault"].notna().all()
                else None
            )
            fig = distribution_plot(df, selected_col, target_col=plot_target)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### 特征相关性")
    if st.button("📊 生成相关性矩阵", help="对前 15 个数值特征计算相关性"):
        fig = correlation_heatmap(df, top_n=15)
        st.plotly_chart(fig, use_container_width=True)
