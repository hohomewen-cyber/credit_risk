"""
💬 LLM 智能助手页面
====================
聊天界面，支持四种模式：
- 风险报告
- 特征建议
- 异常分析
- 自由对话
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar

st.set_page_config(page_title="LLM 智能助手", page_icon="💬", layout="wide")
render_sidebar()

st.title("💬 LLM 智能助手")
st.caption("基于 DeepSeek 大模型的风控领域 AI 助手")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))

from app.components.data_utils import (
    SOURCE_LABELS,
    initialize_data_source_state,
    load_prediction_resources,
    load_selected_data,
    prepare_model_features,
    render_analysis_context,
)

active_source = initialize_data_source_state(ROOT)
render_analysis_context(ROOT, "XGBoost")
source_df, source_label = load_selected_data(ROOT, active_source)
if source_df.empty:
    st.error("当前数据源不可用，请先在数据探索页面重新选择。")
    st.stop()
target_column = configs.get("data").target_column
true_labels = (
    pd.to_numeric(source_df[target_column], errors="coerce").dropna()
    if target_column in source_df.columns else pd.Series(dtype=float)
)


# ===== 检查 API Key =====
import os
api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key.startswith("sk-") or api_key == "sk-your-api-key-here":
    st.error("⚠️ 未配置 DeepSeek API Key")
    st.info("""
    **配置步骤：**
    1. 访问 https://platform.deepseek.com 注册账号
    2. 生成 API Key
    3. 在项目根目录的 `.env` 文件中填入：
       ```
       DEEPSEEK_API_KEY=sk-your-real-key-here
       ```
    4. 重启 Streamlit 应用
    """)
    st.stop()


# ===== 初始化客户端 =====
@st.cache_resource
def get_llm_client():
    from src.llm.client import DeepSeekClient
    from src.llm.prompts import PromptManager
    return DeepSeekClient(configs.get("llm")), PromptManager(configs.get("llm"))


client, prompt_mgr = get_llm_client()


# ===== 模式选择 =====
mode = st.radio(
    "选择对话模式",
    ["💬 自由对话", "📝 风险报告", "💡 特征建议", "🔬 异常案例分析"],
    horizontal=True,
)

st.divider()


# ===== 模式：自由对话 =====
if mode == "💬 自由对话":
    st.markdown("##### 💡 你可以问我关于数据、模型、风控业务的任何问题")

    # 初始化对话历史
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # 渲染历史
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if user_input := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 准备上下文
        try:
            # 加载当前数据源和与其匹配的模型状态作为上下文
            latest_dir = ROOT / "data" / "models" / "latest"
            model_info = {}
            dataset_info = {
                "source": source_label,
                "n_samples": len(source_df),
                "n_features": source_df.drop(
                    columns=[target_column, configs.get("data").id_column],
                    errors="ignore",
                ).shape[1],
                "labeled_samples": len(true_labels),
                "default_rate": (
                    f"{true_labels.mean():.2%}" if len(true_labels) else "无真实终态标签"
                ),
                "missing_rate": f"{source_df.isna().mean().mean():.2%}",
            }

            eval_path = latest_dir / "eval_results.json"
            meta_path = latest_dir / "meta.json"
            model_source_matches = False
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                model_source_matches = meta.get("data_source") == source_label
            if eval_path.exists() and model_source_matches:
                eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
                model_info = {
                    "xgb_auc": round(eval_data.get("xgboost", {}).get("auc", 0), 4),
                    "mlp_auc": round(eval_data.get("mlp", {}).get("auc", 0), 4),
                    "ensemble_auc": round(eval_data.get("ensemble", {}).get("auc", 0), 4),
                }

            # 渲染 Prompt
            prompt = prompt_mgr.render(
                "chat_assistant",
                user_question=user_input,
                model_info=model_info if model_info else None,
                dataset_info=dataset_info,
            )
            system_prompt = prompt_mgr.get_system_role("chat_assistant")

            # 历史对话（最多保留最近 N 轮）
            max_history = configs.get("llm").agents.chat_agent.max_history_turns
            history = []
            for h in st.session_state.chat_messages[-max_history * 2:-1]:
                history.append({"role": h["role"], "content": h["content"]})

            # 流式输出
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                for chunk in client.stream_chat(
                    prompt,
                    system_prompt=system_prompt,
                    history=history,
                    reasoning_effort="none",
                ):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                if not full_response.strip():
                    raise RuntimeError("LLM 返回正文为空")
                placeholder.markdown(full_response)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"调用失败: {e}")

    # 清空按钮
    if st.session_state.chat_messages:
        if st.button("🗑️ 清空对话"):
            st.session_state.chat_messages = []
            st.rerun()


# ===== 模式：风险报告 =====
elif mode == "📝 风险报告":
    st.info("ℹ️ 完整的风险报告功能在 **🎯 单笔预测** 页面，本处提供快速演示。")
    st.markdown("##### 输入贷款信息，让 LLM 生成风险评估报告")

    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt = st.number_input("贷款金额", 1000, 500000, 30000)
        annual_income = st.number_input("年收入", 10000, 1000000, 80000)
    with col2:
        interest_rate = st.slider("利率(%)", 5.0, 30.0, 13.5)
        dti = st.slider("DTI(%)", 0.0, 60.0, 18.0)
    with col3:
        fico = st.slider("FICO 分数", 580, 850, 690)
        # 简单计算"违约概率"用于演示
        proba = min(max(0.05 + (interest_rate - 5) * 0.02 + dti * 0.005 - (fico - 580) * 0.001, 0.0), 0.95)
        st.metric("模拟违约概率", f"{proba:.2%}")

    if st.button("🤖 生成简化报告", type="primary"):
        prompt = f"""请基于以下信贷申请信息，生成一份简短的风险评估意见（300字以内）：

- 贷款金额: ¥{loan_amnt:,}
- 利率: {interest_rate}%
- 年收入: ¥{annual_income:,}
- 负债收入比: {dti}%
- FICO 分数: {fico}
- 模型预测违约概率: {proba:.2%}

请给出: 1) 风险等级 2) 主要担忧点 3) 审批建议
"""
        try:
            with st.spinner("生成中..."):
                placeholder = st.empty()
                full = ""
                for chunk in client.stream_chat(prompt, reasoning_effort="none"):
                    full += chunk
                    placeholder.markdown(full + "▌")
                if not full.strip():
                    raise RuntimeError("LLM 返回正文为空")
                placeholder.markdown(full)
        except Exception as e:
            st.error(f"调用失败: {e}")


# ===== 模式：特征建议 =====
elif mode == "💡 特征建议":
    st.markdown("##### 基于当前真实数据质量生成特征建议")
    st.caption(
        f"当前使用：{source_label}｜{len(source_df):,} 行｜"
        f"缺失率 {source_df.isna().mean().mean():.2%}｜真实标签 {len(true_labels):,} 条"
    )

    if st.button("🚀 生成建议", type="primary"):
        try:
            from src.llm.agents.feature_agent import FeatureAgent
            agent = FeatureAgent(client, prompt_mgr, configs.get("llm").agents.feature_agent)

            with st.spinner("AI 思考中..."):
                eval_path = ROOT / "data" / "models" / "latest" / "eval_results.json"
                meta_path = ROOT / "data" / "models" / "latest" / "meta.json"
                eval_results = {}
                if eval_path.exists() and meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("data_source") == source_label:
                        eval_results = json.loads(eval_path.read_text(encoding="utf-8"))
                configured_rules = [
                    rule.name
                    for rule in configs.get("feature").derived_features.rules
                ]
                missing = source_df.isna().mean().sort_values(ascending=False).head(5)
                result = agent.suggest(
                    df=source_df,
                    current_auc=(
                        eval_results.get("xgboost", {}).get("auc", "当前数据源尚无训练评估")
                    ),
                    current_model="XGBoost",
                    source_name=source_label,
                    existing_features=configured_rules,
                    target_summary=(
                        f"真实终态标签 {len(true_labels)} 条，违约率 {true_labels.mean():.2%}"
                        if len(true_labels) else "没有真实终态标签，不能评价模型效果"
                    ),
                    bottleneck="缺失最多：" + "、".join(
                        f"{name} {ratio:.1%}" for name, ratio in missing.items()
                    ),
                )
                st.session_state["llm_feature_result"] = {
                    "source": active_source, "result": result,
                }

            st.success(f"已基于 {source_label} 生成建议")
            suggestions = result.get("suggestions", [])
            if suggestions:
                for i, sug in enumerate(suggestions):
                    with st.expander(
                        f"💡 建议 {i+1}: "
                        f"{sug.get('business_meaning') or sug.get('name', '未命名')}"
                    ):
                        st.markdown(f"**建议特征**：`{sug.get('name', '未命名')}`")
                        st.markdown(f"**为什么值得关注**：{sug.get('business_meaning', '未说明')}")
                        st.markdown(f"**怎样从现有数据得到**：{sug.get('construction', '未说明')}")
                        st.markdown(f"**可能帮助识别**：{sug.get('expected_impact', '未说明')}")

            if "general_advice" in result:
                st.info(f"**整体建议**: {result['general_advice']}")
        except Exception as e:
            st.error(f"调用失败: {e}")


# ===== 模式：异常分析 =====
elif mode == "🔬 异常案例分析":
    st.markdown("##### 分析模型预测错误的高置信度案例")
    st.caption(
        f"从 {source_label} 的真实终态标签中查找预测错误；"
        "没有真实标签时不会构造或伪造异常案例。"
    )

    if st.button("🔬 查找并分析真实异常案例", type="primary"):
        try:
            if len(true_labels) < 2 or true_labels.nunique() < 2:
                raise RuntimeError("当前数据源缺少足够的两类真实终态标签，不能分析预测错误")
            resources = load_prediction_resources(str(ROOT))
            if resources is None:
                raise RuntimeError("模型产物不完整，请先完成模型训练")
            labeled_mask = pd.to_numeric(
                source_df[target_column], errors="coerce"
            ).notna()
            labeled_raw = source_df.loc[labeled_mask].reset_index(drop=True)
            y_true = pd.to_numeric(
                labeled_raw[target_column], errors="raise"
            ).astype(int).to_numpy()
            X = prepare_model_features(labeled_raw, resources)
            probabilities = resources["xgb_model"].predict_proba(X)

            from src.explain.shap_analyzer import SHAPAnalyzer
            from src.llm.agents.case_agent import CaseAgent

            case_agent = CaseAgent(
                client, prompt_mgr, configs.get("llm").agents.case_agent
            )
            anomalies = case_agent.find_anomalies(y_true, probabilities, X)
            if anomalies.empty:
                st.session_state["llm_case_result"] = {
                    "source": active_source,
                    "message": "当前抽样数据中未发现达到配置置信度阈值的错误案例。",
                }
            else:
                case = anomalies.iloc[0]
                row_position = int(case["index"])
                analyzer = SHAPAnalyzer(max_background_size=500)
                analyzer.fit(resources["xgb_model"].model, X.head(500))
                explanation = analyzer.explain_single(X.iloc[row_position])
                with st.spinner("正在解释真实异常案例..."):
                    diagnosis = case_agent.analyze(
                        features=labeled_raw.iloc[row_position].to_dict(),
                        true_label=int(case["true_label"]),
                        probability=float(case["pred_proba"]),
                        shap_explanation=explanation,
                    )
                st.session_state["llm_case_result"] = {
                    "source": active_source,
                    "message": diagnosis,
                    "case": case.to_dict(),
                }
        except Exception as e:
            st.error(f"调用失败: {e}")

    case_state = st.session_state.get("llm_case_result")
    if case_state and case_state.get("source") == active_source:
        case = case_state.get("case")
        if case:
            c1, c2, c3 = st.columns(3)
            c1.metric("真实结果", "违约" if int(case["true_label"]) else "正常")
            c2.metric("模型违约概率", f"{case['pred_proba']:.2%}")
            c3.metric("错误类型", case["error_type"])
        st.markdown(case_state["message"])
