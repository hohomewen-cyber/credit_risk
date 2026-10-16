"""
🔧 特征工程页面
================
展示预处理流程、衍生特征构造、LLM 特征建议。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar

st.set_page_config(page_title="特征工程", page_icon="🔧", layout="wide")
render_sidebar()

st.title("🔧 特征工程")
st.caption("特征工程决定模型上限")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))


# ===== 数据加载（使用数据探索页选中的全局数据源） =====
from app.components.data_utils import (
    SOURCE_LABELS,
    initialize_data_source_state,
    load_active_data_or_stop,
    render_analysis_context,
)

active_source = initialize_data_source_state(ROOT)
render_analysis_context(ROOT, "XGBoost")
df = load_active_data_or_stop(ROOT)
source_label = SOURCE_LABELS.get(active_source, active_source)
data_cfg = configs.get("data")
feature_cfg = configs.get("feature")
target = data_cfg.target_column
id_col = data_cfg.id_column

# ===== 当前数据源的真实特征概况 =====
configured_features = [
    *data_cfg.columns.numerical,
    *data_cfg.columns.categorical,
    *data_cfg.columns.datetime,
    *data_cfg.columns.anonymous,
]
X_current = df.drop(columns=[target, id_col], errors="ignore").reindex(
    columns=configured_features
)
available_features = [name for name in X_current if X_current[name].notna().any()]
missing_total = int(X_current.isna().sum().sum())
missing_ratio = float(X_current.isna().mean().mean()) if len(X_current) else 0.0
labeled_count = int(df[target].notna().sum()) if target in df.columns else 0

st.markdown("### 📊 当前数据的特征概况")
p1, p2, p3, p4 = st.columns(4)
p1.metric("当前数源", source_label)
p2.metric("可用标准特征", f"{len(available_features)} / {len(configured_features)}")
p3.metric("特征缺失率", f"{missing_ratio:.1%}")
p4.metric("可用真实标签", f"{labeled_count:,} / {len(df):,}")

if labeled_count < len(df):
    st.info(
        f"{source_label} 中有 {len(df) - labeled_count:,} 条记录尚无最终还款结果；"
        "它们可用于预测和分布分析，但不会被伪造标签用于特征筛选或训练。"
    )

missing_by_feature = X_current.isna().sum().sort_values(ascending=False)
missing_by_feature = missing_by_feature[missing_by_feature > 0]
with st.expander("查看当前数源缺失最多的特征"):
    if missing_by_feature.empty:
        st.success("当前标准特征没有缺失值。")
    else:
        st.dataframe(pd.DataFrame({
            "特征": missing_by_feature.index,
            "缺失数": missing_by_feature.values,
            "缺失率": [f"{value / len(X_current):.1%}" for value in missing_by_feature.values],
        }).head(15), use_container_width=True, hide_index=True)

# ===== Tabs =====
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ 预处理流程",
    "🔬 衍生特征",
    "✂️ 特征筛选",
    "🤖 LLM 特征建议",
])

with tab1:
    st.markdown("## 数据预处理流程")
    st.markdown("以下步骤定义在 `configs/feature_config.yaml` 中：")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 缺失值处理")
        st.code(f"""
数值列策略: {feature_cfg.missing_value.numerical_strategy}
类别列策略: {feature_cfg.missing_value.categorical_strategy}
丢弃阈值: {feature_cfg.missing_value.drop_threshold}
按列策略: {dict(feature_cfg.missing_value.get('strategy', {}))}
KNN 邻居数: {feature_cfg.missing_value.get('knn_neighbors', 5)}
        """, language="yaml")

        st.markdown("### 异常值处理")
        outlier = feature_cfg.outlier
        st.code(f"""
方法: {outlier.method}
下分位: {outlier.lower_quantile}
上分位: {outlier.upper_quantile}
        """, language="yaml")

    with col2:
        st.markdown("### 类别编码")
        st.code(f"""
默认方法: {feature_cfg.encoding.method}
自定义映射: {len(feature_cfg.encoding.get('custom', {}))} 个字段
        """, language="yaml")

        st.markdown("### 数值标准化")
        st.code(f"""
方法: {feature_cfg.scaling.method}
        """, language="yaml")

    # 演示预处理效果
    st.divider()
    st.markdown("### 🔥 实时演示：对前 1000 条数据应用预处理")

    if st.button("🚀 执行预处理", type="primary"):
        from src.data.preprocessor import Preprocessor

        with st.spinner("处理中..."):
            sample_df = df.head(1000).copy()
            from src.data.adapters import LendingClubAdapter

            X = sample_df.drop(columns=[target, id_col], errors="ignore")
            X = X.reindex(columns=LendingClubAdapter.PROJECT_FEATURES)
            knn_features = [
                name for name, strategy in
                feature_cfg.missing_value.get("strategy", {}).items()
                if strategy == "knn" and name in X.columns
            ]
            missing_before = int(X.isna().sum().sum())
            knn_before = {
                name: int(X[name].isna().sum()) for name in knn_features
            }

            prep = Preprocessor(configs.get("data"), feature_cfg)
            X_processed = prep.fit_transform(X)
            missing_after = int(X_processed.isna().sum().sum())
            knn_after = {
                name: int(X_processed[name].isna().sum())
                for name in knn_features if name in X_processed.columns
            }

            st.session_state["knn_result"] = {
                "source": active_source,
                "source_label": source_label,
                "before": sum(knn_before.values()),
                "after": sum(knn_after.values()),
                "overall_before": missing_before,
                "overall_after": missing_after,
                "features": knn_features,
                "feature_before": knn_before,
                "feature_after": knn_after,
                "neighbors": int(feature_cfg.missing_value.get("knn_neighbors", 5)),
                "raw_preview": X.head(),
                "processed_preview": X_processed.head(),
                "raw_shape": X.shape,
                "processed_shape": X_processed.shape,
            }

    knn_result = st.session_state.get("knn_result")
    if knn_result and knn_result.get("source") == active_source:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**原始数据**")
            st.dataframe(knn_result["raw_preview"], use_container_width=True)
            st.caption(
                f"shape: {knn_result['raw_shape']}, 全部字段缺失值: "
                f"{knn_result['overall_before']}"
            )

        with col2:
            st.markdown("**处理后数据**")
            st.dataframe(knn_result["processed_preview"], use_container_width=True)
            st.caption(
                f"shape: {knn_result['processed_shape']}, 全部字段缺失值: "
                f"{knn_result['overall_after']}"
            )

        st.markdown("### KNN 缺失值填充结果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KNN字段填充前缺失", knn_result["before"])
        c2.metric("KNN字段填充后缺失", knn_result["after"])
        c3.metric("处理特征数", len(knn_result["features"]))
        c4.metric("处理状态", "完成" if knn_result["after"] == 0 else "仍有缺失")
        st.info(
            f"当前数据源：{knn_result['source_label']}。"
            f"使用策略：KNN（{knn_result['neighbors']} 个邻居）；"
            f"处理特征：{', '.join(knn_result['features']) or '当前配置无KNN字段'}。"
        )
        if knn_result["before"] == 0:
            st.success("当前数源的 KNN 目标字段无缺失，本次没有需要补全的记录。")
        elif knn_result["after"] == 0:
            st.success(
                f"已根据相似申请人的其他数值特征，补全 "
                f"{knn_result['before']} 个缺失值；填充后无剩余缺失。"
            )
        if knn_result["features"]:
            st.dataframe(pd.DataFrame([
                {
                    "特征": name,
                    "填充前缺失": knn_result["feature_before"].get(name, 0),
                    "填充后缺失": knn_result["feature_after"].get(name, 0),
                    "状态": "已完成" if knn_result["feature_after"].get(name, 0) == 0 else "仍有缺失",
                }
                for name in knn_result["features"]
            ]), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("## 衍生特征构造")
    st.markdown("基于业务理解构造新特征，常常能显著提升模型表现。")

    rules = feature_cfg.derived_features.rules

    st.markdown(f"### 当前配置了 **{len(rules)}** 条衍生规则：")

    for i, rule in enumerate(rules):
        with st.expander(f"📐 规则 {i+1}：{rule['name']}"):
            st.markdown(f"**业务含义**: {rule['description']}")
            st.code(rule["formula"], language="python")

    st.divider()

    if st.button("🔬 应用所有衍生规则", type="primary"):
        from src.features.builder import FeatureBuilder

        with st.spinner("构造衍生特征..."):
            sample_df = X_current.head(1000).copy()
            builder = FeatureBuilder(feature_cfg)
            df_new = builder.build(sample_df)
            new_features = builder.get_created_features()
            details = []
            for name in new_features:
                non_null = int(df_new[name].notna().sum())
                details.append({
                    "衍生特征": name,
                    "可计算样本": non_null,
                    "无法计算样本": len(df_new) - non_null,
                    "状态": "可用" if non_null else "当前数源缺少所需原字段",
                })
            st.session_state["derived_feature_result"] = {
                "source": active_source,
                "source_label": source_label,
                "features": new_features,
                "details": details,
                "preview": df_new[new_features].head() if new_features else pd.DataFrame(),
            }

    derived_result = st.session_state.get("derived_feature_result")
    if derived_result and derived_result.get("source") == active_source:
        st.success(
            f"已在 {derived_result['source_label']} 上实际执行 "
            f"{len(derived_result['features'])} 条衍生规则。"
        )
        st.dataframe(pd.DataFrame(derived_result["details"]), use_container_width=True, hide_index=True)
        if not derived_result["preview"].empty:
            with st.expander("查看衍生特征样例"):
                st.dataframe(derived_result["preview"], use_container_width=True)

with tab3:
    st.markdown("## 特征筛选")
    st.markdown("通过多种方法移除冗余、低价值特征。")

    selection_cfg = feature_cfg.feature_selection

    st.markdown(f"**启用状态**: {'✅ 已启用' if selection_cfg.enabled else '❌ 已禁用'}")

    if selection_cfg.enabled:
        st.markdown("### 筛选方法")
        for method in selection_cfg.methods:
            st.markdown(f"#### 📌 {method['name']}")
            st.json(dict(method))

        if st.button("✂️ 在当前数据源上执行特征筛选", type="primary"):
            from src.data.preprocessor import Preprocessor
            from src.features.builder import FeatureBuilder
            from src.features.selector import FeatureSelector

            labeled = df.loc[df[target].notna()].head(2000).copy()
            if len(labeled) < 20 or labeled[target].nunique() < 2:
                st.error("当前数源缺少足够的真实二分类标签，无法执行基于模型的特征筛选。")
            else:
                with st.spinner("正在按原有预处理和筛选流程计算..."):
                    raw_x = labeled.drop(columns=[target, id_col], errors="ignore").reindex(
                        columns=configured_features
                    )
                    processed = Preprocessor(data_cfg, feature_cfg).fit_transform(raw_x)
                    built = FeatureBuilder(feature_cfg).build(processed)
                    selector = FeatureSelector(feature_cfg)
                    selected = selector.fit_transform(built, labeled[target].astype(int))
                    st.session_state["feature_selection_result"] = {
                        "source": active_source,
                        "source_label": source_label,
                        "input_count": built.shape[1],
                        "selected": selector.selected_features,
                        "dropped": selector.dropped_features,
                        "sample_count": len(labeled),
                    }

        selection_result = st.session_state.get("feature_selection_result")
        if selection_result and selection_result.get("source") == active_source:
            s1, s2, s3 = st.columns(3)
            s1.metric("参与筛选样本", selection_result["sample_count"])
            s2.metric("筛选前特征", selection_result["input_count"])
            s3.metric("最终保留特征", len(selection_result["selected"]))
            st.success(
                f"筛选结果来自 {selection_result['source_label']}，"
                "数量和特征名称会随当前数据源变化。"
            )
            with st.expander("查看保留与移除的特征"):
                st.markdown("**保留特征**")
                st.write(", ".join(selection_result["selected"]) or "无")
                st.markdown("**移除特征**")
                st.write(", ".join(selection_result["dropped"]) or "无")

with tab4:
    st.markdown("## 🤖 LLM 智能特征建议")
    st.markdown(
        f"让 DeepSeek 基于 **{source_label}** 的真实字段、缺失情况和标签覆盖，"
        "用面向信贷审批人员的语言提出建议。"
    )

    if st.button("💡 生成特征建议", type="primary", help="约 10-30 秒"):
        try:
            from src.llm.client import DeepSeekClient
            from src.llm.prompts import PromptManager
            from src.llm.agents.feature_agent import FeatureAgent

            llm_cfg = configs.get("llm")

            with st.spinner("LLM 思考中..."):
                client = DeepSeekClient(llm_cfg)
                pm = PromptManager(llm_cfg)
                agent = FeatureAgent(client, pm, llm_cfg.agents.feature_agent)

                eval_path = ROOT / "data" / "models" / "latest" / "eval_results.json"
                eval_results = (
                    json.loads(eval_path.read_text(encoding="utf-8"))
                    if eval_path.exists() else {}
                )
                meta_path = ROOT / "data" / "models" / "latest" / "meta.json"
                model_source = None
                if meta_path.exists():
                    model_source = json.loads(meta_path.read_text(encoding="utf-8")).get("data_source")
                current_auc = (
                    eval_results.get("xgboost", {}).get("auc", 0.0)
                    if model_source == source_label else "当前数源尚无对应训练结果"
                )
                top_missing = ", ".join(
                    f"{name}({count / len(X_current):.1%})"
                    for name, count in missing_by_feature.head(5).items()
                ) or "无明显缺失"
                target_summary = (
                    f"{labeled_count}/{len(df)} 条有真实终态标签"
                    if target in df.columns else "当前数源无标签列"
                )
                result = agent.suggest(
                    df=df.head(1000),
                    current_auc=current_auc,
                    current_model="XGBoost",
                    source_name=source_label,
                    existing_features=[rule["name"] for rule in rules],
                    target_summary=target_summary,
                    bottleneck=(
                        f"标准特征可用 {len(available_features)}/{len(configured_features)}；"
                        f"缺失最多的字段：{top_missing}"
                    ),
                )
                st.session_state["feature_llm_result"] = {
                    "source": active_source,
                    "source_label": source_label,
                    "revision": 2,
                    "result": result,
                }

            st.success("✅ 已生成建议")

        except Exception as e:
            st.error(f"调用失败: {e}")
            st.caption("请检查 .env 中的 DEEPSEEK_API_KEY 配置")

    llm_state = st.session_state.get("feature_llm_result")
    if (
        llm_state
        and llm_state.get("source") == active_source
        and llm_state.get("revision") == 2
    ):
        result = llm_state["result"]
        suggestions = result.get("suggestions", [])
        if suggestions:
            st.success(
                f"以下 {len(suggestions)} 条建议仅针对当前数据源："
                f"{llm_state['source_label']}。切换数源后必须重新生成。"
            )
            st.markdown(f"### 💡 共 {len(suggestions)} 条可评审建议")
            for i, sug in enumerate(suggestions):
                title = sug.get("business_meaning") or sug.get("name", "未命名")
                with st.expander(f"建议 {i+1}：{title}"):
                    st.markdown(f"**建议特征**：`{sug.get('name', '未命名')}`")
                    st.markdown(f"**为什么值得关注**：{sug.get('business_meaning', '未说明')}")
                    st.markdown(f"**怎样从现有数据得到**：{sug.get('construction', '未说明')}")
                    st.markdown(f"**可能帮助识别**：{sug.get('expected_impact', '未说明')}")
                    if sug.get("code"):
                        st.caption("技术人员参考代码")
                        st.code(sug["code"], language="python")
        else:
            st.info("未解析到结构化建议")
        if result.get("general_advice"):
            st.markdown("### 整体建议")
            st.info(result["general_advice"])
