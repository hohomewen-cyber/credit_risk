"""
🔍 模型解释页面
================
SHAP 全局/局部解释。
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.chart_utils import feature_importance_bar

st.set_page_config(page_title="模型解释", page_icon="🔍", layout="wide")
render_sidebar()

st.title("🔍 模型解释 (SHAP)")
st.caption("理解模型为什么这么判 - 全局重要性 + 局部解释")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))

from app.components.data_utils import (
    available_data_sources,
    data_source_version,
    initialize_data_source_state,
    load_prediction_resources,
    load_selected_data,
    prepare_model_features,
    render_analysis_context,
)

active_source = initialize_data_source_state(ROOT)
render_analysis_context(ROOT, "XGBoost")


# ===== 加载模型与数据 =====
@st.cache_resource
def load_resources(source_key: str, source_version: float):
    from src.explain.shap_analyzer import SHAPAnalyzer

    prediction_resources = load_prediction_resources(str(ROOT))
    if prediction_resources is None:
        return None
    df, source_label = load_selected_data(ROOT, source_key, nrows=1500)
    if df.empty:
        return None
    target = configs.get("data").target_column
    X_processed = prepare_model_features(df, prediction_resources)

    # 构建 SHAP（用前 1000 条作为背景）
    analyzer = SHAPAnalyzer(max_background_size=500)
    analyzer.fit(prediction_resources["xgb_model"].model, X_processed.head(500))

    return {
        "model": prediction_resources["xgb_model"],
        "X": X_processed,
        "y": df[target].reset_index(drop=True) if target in df.columns else None,
        "analyzer": analyzer,
        "source_label": source_label,
    }


resources = load_resources(active_source, data_source_version(ROOT, active_source))
if resources is None:
    st.warning("⚠️ 模型或数据未就绪，请先完成训练")
    st.stop()

# ===== Tabs =====
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 全局解释",
    "🎯 单样本解释",
    "📈 模型自带重要性",
    "📡 模型漂移监控",
])

with tab1:
    st.markdown("## 全局特征重要性")
    st.markdown("""
    SHAP 全局重要性 = 所有样本 SHAP 值绝对值的均值，
    比 XGBoost 自带的 `feature_importance_` 更准确。
    """)

    n_samples = st.slider("分析样本数", 50, 1000, 200, step=50, help="样本越多越准，但耗时增加")

    if st.button("🔬 计算全局 SHAP 值", type="primary"):
        with st.spinner(f"计算 {n_samples} 个样本的 SHAP 值..."):
            X_sample = resources["X"].sample(min(n_samples, len(resources["X"])), random_state=42)
            importance = resources["analyzer"].global_importance(X_sample, top_n=20)

        st.success("✅ 计算完成")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = feature_importance_bar(importance, top_n=20, title="SHAP 全局特征重要性")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(
                pd.DataFrame({
                    "特征": importance.index,
                    "SHAP 重要性": importance.values.round(4),
                }),
                use_container_width=True,
                hide_index=True,
            )

with tab2:
    st.markdown("## 单样本 SHAP 解释")

    sample_idx = st.number_input(
        "选择样本索引",
        0,
        len(resources["X"]) - 1,
        0,
        help="从测试集中选择一个样本进行解释"
    )

    if st.button("🔍 解释此样本", type="primary"):
        sample = resources["X"].iloc[sample_idx]

        with st.spinner("计算 SHAP 值..."):
            explanation = resources["analyzer"].explain_single(sample)

        # 预测信息
        proba = resources["model"].predict_proba(sample.to_frame().T)[0]
        true_label = resources["y"].iloc[sample_idx] if resources["y"] is not None else None

        col1, col2, col3 = st.columns(3)
        col1.metric("预测违约概率", f"{proba:.2%}")
        col2.metric(
            "真实标签",
            f"{int(true_label)}" if true_label is not None and not pd.isna(true_label) else "未知",
        )
        col3.metric("Top1 影响特征", explanation.get_top_features(1)[0]["feature"])

        # 瀑布图
        from app.components.chart_utils import shap_waterfall
        fig = shap_waterfall(
            explanation.feature_names,
            explanation.feature_values,
            explanation.shap_values,
            explanation.base_value,
            top_n=10,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top 特征表
        st.markdown("### Top 10 特征贡献")
        st.dataframe(
            pd.DataFrame(explanation.get_top_features(top_n=10)),
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    st.markdown("## 模型自带特征重要性")
    st.markdown("XGBoost 自带的 gain 重要性")

    importance_type = st.selectbox(
        "重要性类型",
        ["gain", "weight", "cover", "total_gain", "total_cover"],
        help="gain: 平均增益（推荐）；weight: 使用次数；cover: 覆盖样本数",
    )

    importance = resources["model"].get_feature_importance(importance_type=importance_type)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = feature_importance_bar(importance, top_n=20, title=f"XGBoost {importance_type} 重要性")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            pd.DataFrame({
                "特征": importance.index[:20],
                "重要性": importance.values[:20].round(4),
            }),
            use_container_width=True,
            hide_index=True,
        )

with tab4:
    st.markdown("## KS 特征漂移检测")
    st.markdown(
        "选择基准数据、待检测数据和特征后，点击按钮才会按开发指导书第7.4节执行KS检验。"
    )
    source_options = available_data_sources(ROOT)
    source_keys = list(source_options)
    default_baseline = "sample" if "sample" in source_options else source_keys[0]
    default_new = active_source if active_source in source_options else source_keys[-1]

    c_source1, c_source2 = st.columns(2)
    with c_source1:
        baseline_source = st.selectbox(
            "基准数据源",
            source_keys,
            index=source_keys.index(default_baseline),
            format_func=lambda key: source_options[key],
            key="ks_baseline_source",
        )
    with c_source2:
        new_source = st.selectbox(
            "待检测数据源",
            source_keys,
            index=source_keys.index(default_new),
            format_func=lambda key: source_options[key],
            key="ks_new_source",
        )

    feature_options = resources["X"].columns.tolist()
    selected_feature = st.selectbox(
        "检测特征", feature_options, key="ks_selected_feature"
    )
    drift_threshold = st.slider(
        "KS 阈值", 0.01, 0.50, 0.10, step=0.01, key="ks_threshold"
    )
    signature = (
        baseline_source, new_source, selected_feature, round(drift_threshold, 4)
    )
    stored_ks = st.session_state.get("ks_result")
    if stored_ks and stored_ks.get("signature") != signature:
        st.session_state.pop("ks_result", None)
        stored_ks = None

    if st.button("📡 执行漂移检测", type="primary"):
        from src.monitoring import DriftDetector

        prediction_resources = load_prediction_resources(str(ROOT))
        baseline_raw, baseline_label = load_selected_data(
            ROOT, baseline_source, nrows=1500
        )
        new_raw, new_label = load_selected_data(ROOT, new_source, nrows=1500)
        baseline = prepare_model_features(baseline_raw, prediction_resources)
        new_data = prepare_model_features(new_raw, prediction_resources)

        if baseline_source == new_source:
            combined = baseline
            split_at = len(combined) // 2
            baseline = combined.iloc[:split_at]
            new_data = combined.iloc[split_at:]
            baseline_label = f"{baseline_label}（前半）"
            new_label = f"{new_label}（后半）"

        selected_result = DriftDetector(
            baseline[[selected_feature]]
        ).evaluate(new_data[[selected_feature]], threshold=drift_threshold)[selected_feature]
        st.session_state["ks_result"] = {
            "signature": signature,
            "baseline_label": baseline_label,
            "new_label": new_label,
            "baseline_count": len(baseline),
            "new_count": len(new_data),
            "feature": selected_feature,
            "threshold": drift_threshold,
            "ks_stat": selected_result["ks_stat"],
            "p_value": selected_result["p_value"],
            "is_drift": selected_result["is_drift"],
            "baseline_values": baseline[selected_feature].dropna(),
            "new_values": new_data[selected_feature].dropna(),
        }
        stored_ks = st.session_state["ks_result"]

    if stored_ks:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("基准样本", stored_ks["baseline_count"])
        col2.metric("新数据样本", stored_ks["new_count"])
        col3.metric("KS statistic", f"{stored_ks['ks_stat']:.4f}")
        col4.metric("p-value", f"{stored_ks['p_value']:.6g}")
        status = "⚠️ 发生漂移" if stored_ks["is_drift"] else "✅ 未发现漂移"
        st.info(
            f"检测特征：**{stored_ks['feature']}**｜阈值：{stored_ks['threshold']:.2f}｜"
            f"状态：**{status}**"
        )
        if stored_ks["is_drift"]:
            st.warning(
                "该特征在两组数据中的分布差异超过当前阈值，需要关注模型在新数据上的稳定性；"
                "该结果不等同于模型已经失效。"
            )
        else:
            st.success("该特征的分布差异未超过当前阈值。")

        import plotly.express as px
        comparison = pd.concat([
            pd.DataFrame({
                "数值": stored_ks["baseline_values"],
                "样本": stored_ks["baseline_label"],
            }),
            pd.DataFrame({
                "数值": stored_ks["new_values"],
                "样本": stored_ks["new_label"],
            }),
        ], ignore_index=True)
        fig = px.histogram(
            comparison,
            x="数值",
            color="样本",
            barmode="overlay",
            histnorm="probability density",
            opacity=0.55,
            title=f"{stored_ks['feature']} 分布对比",
        )
        st.plotly_chart(fig, use_container_width=True)
