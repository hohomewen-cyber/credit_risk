"""
🎯 单笔预测页面
================
表单输入 -> 模型预测 -> SHAP 解释 -> LLM 风险报告
本页面是系统最具 Demo 效果的部分，建议答辩时重点演示。
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
from app.components.chart_utils import gauge_chart, shap_waterfall

st.set_page_config(page_title="单笔预测", page_icon="🎯", layout="wide")
render_sidebar()

st.title("🎯 单笔贷款风险评估")
st.caption("模拟一个完整的贷款审批流程：录入信息 → 模型预测 → SHAP 解释 → LLM 报告")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))

from app.components.data_utils import (
    data_source_version,
    initialize_data_source_state,
    load_prediction_resources,
    load_selected_data,
    prepare_model_features,
    render_analysis_context,
)

active_source = initialize_data_source_state(ROOT)
render_analysis_context(ROOT, "XGBoost（主预测）/ Random Forest（A/B B组）")
if (
    "prediction" in st.session_state
    and st.session_state["prediction"].get("source") != active_source
):
    for stale_key in ["prediction", "llm_report", "compliance_result"]:
        st.session_state.pop(stale_key, None)


# ===== 报告生成辅助函数 =====
def _build_complete_report(
    input_data: dict,
    proba: float,
    risk_level: str,
    recommendation: str,
    explanation,
    llm_report: str,
) -> str:
    """
    把申请人信息、模型预测、SHAP 解释、LLM 分析整合成一份完整的 Markdown 报告。
    用户下载后可以直接归档或打印。
    """
    # 字段中文映射
    field_labels = {
        "loanAmnt": "贷款金额", "term": "贷款期限（年）", "interestRate": "利率（%）",
        "installment": "月供", "grade": "贷款等级", "subGrade": "贷款子等级",
        "employmentLength": "就业年限", "homeOwnership": "房屋所有权",
        "annualIncome": "年收入", "verificationStatus": "验证状态",
        "purpose": "贷款用途代码", "dti": "负债收入比（%）",
        "delinquency_2years": "近2年逾期次数", "ficoRangeLow": "FICO 下限",
        "ficoRangeHigh": "FICO 上限", "openAcc": "未结信用额度数",
        "revolUtil": "循环额度利用率（%）",
    }
    home_map = {0: "租房", 1: "按揭", 2: "自有", 3: "其他"}
    verify_map = {0: "未验证", 1: "源已验证", 2: "已验证"}

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 🏦 智能信贷风险评估报告",
        "",
        f"**生成时间**：{timestamp}",
        f"**报告编号**：RPT-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
        "",
        "---",
        "",
        "## 一、申请人信息",
        "",
        "| 字段 | 数值 |",
        "|------|------|",
    ]
    for k, label in field_labels.items():
        if k not in input_data:
            continue
        v = input_data[k]
        if k == "homeOwnership":
            v = home_map.get(v, v)
        elif k == "verificationStatus":
            v = verify_map.get(v, v)
        elif isinstance(v, float):
            v = f"{v:,.2f}" if v >= 1 else f"{v:.4f}"
        lines.append(f"| {label} | {v} |")

    lines += [
        "",
        "## 二、模型预测结果",
        "",
        f"- **违约概率**：{proba * 100:.2f}%",
        f"- **风险等级**：**{risk_level}**",
        f"- **审批建议**：{recommendation}",
        "",
    ]

    if explanation is not None:
        try:
            top_features = explanation.get_top_features(top_n=10)
            lines += [
                "## 三、SHAP 关键特征贡献（Top 10）",
                "",
                "| 排名 | 特征 | 取值 | SHAP 值 | 影响方向 |",
                "|------|------|------|---------|----------|",
            ]
            for i, feat in enumerate(top_features, 1):
                name = feat.get("feature", "")
                value = feat.get("value", "")
                shap_val = feat.get("shap_value", 0)
                direction = "↑ 推高违约风险" if shap_val > 0 else "↓ 降低违约风险"
                lines.append(f"| {i} | {name} | {value} | {shap_val:+.4f} | {direction} |")
            lines.append("")
        except Exception:
            pass

    lines += [
        "## 四、LLM 智能分析报告",
        "",
        llm_report,
        "",
        "---",
        "",
        "> **免责声明**：本报告由智能信贷风控系统自动生成，仅供风控分析参考，不构成最终审批决定。"
        "实际审批应结合人工复核及合规审查。",
        "",
    ]
    return "\n".join(lines)


# ===== 加载模型 =====
@st.cache_resource
def load_xgb_model():
    """加载已训练的 XGBoost 模型"""
    from src.models.xgb_model import XGBModel

    model_path = ROOT / "data" / "models" / "latest" / "xgb_model.json"
    if not model_path.exists():
        return None

    model = XGBModel(configs.get("model").xgboost)
    model.load(str(model_path))
    return model


@st.cache_resource
def load_random_forest_model():
    """加载用于 A/B 测试 B 组的 Random Forest 模型。"""
    from src.models.random_forest_model import RandomForestModel

    model_path = ROOT / "data" / "models" / "latest" / "random_forest_model.joblib"
    if not model_path.exists():
        return None
    model = RandomForestModel(configs.get("model").random_forest)
    model.load(str(model_path))
    return model


@st.cache_resource
def load_preprocessor():
    """加载预处理器"""
    path = ROOT / "data" / "models" / "latest" / "preprocessor.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_feature_selector():
    """加载特征筛选器"""
    path = ROOT / "data" / "models" / "latest" / "feature_selector.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_shap_analyzer(_model, source_key: str, source_version: float):
    """加载/构建 SHAP 分析器（缓存）"""
    from src.explain.shap_analyzer import SHAPAnalyzer
    df, _ = load_selected_data(ROOT, source_key, nrows=1000)
    prediction_resources = load_prediction_resources(str(ROOT))
    if df.empty or prediction_resources is None:
        return None
    X_processed = prepare_model_features(df, prediction_resources)

    # 构建 SHAP
    analyzer = SHAPAnalyzer()
    analyzer.fit(_model.model, X_processed.head(500))
    return analyzer


# ===== 检查模型状态 =====
xgb_model = load_xgb_model()
random_forest_model = load_random_forest_model()
prep = load_preprocessor()
selector = load_feature_selector()

if xgb_model is None or prep is None:
    st.warning("⚠️ 模型未训练，请先在 **🤖 模型训练** 页面完成训练")
    st.stop()

# ===== 输入表单 =====
st.markdown("## 📝 申请人信息")

with st.form("loan_application"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 💰 贷款信息")
        application_id = st.number_input("申请编号", 1, 999999999, 10001)
        loan_amnt = st.number_input("贷款金额（元）", 1000, 500000, 30000, step=1000)
        term = st.selectbox("贷款期限（年）", [3, 5])
        interest_rate = st.slider("利率（%）", 5.0, 30.0, 13.5, step=0.1)
        grade = st.selectbox("贷款等级", ["A", "B", "C", "D", "E", "F", "G"], index=2)
        purpose = st.number_input("贷款用途代码", 0, 13, 0, help="0=债务整合, 2=信用卡 等")

    with col2:
        st.markdown("##### 👤 个人信息")
        annual_income = st.number_input("年收入（元）", 10000, 1000000, 80000, step=5000)
        employment_length = st.selectbox(
            "就业年限",
            ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
             "6 years", "7 years", "8 years", "9 years", "10+ years"],
            index=4,
        )
        home_ownership = st.selectbox(
            "房屋所有权",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "租房", 1: "按揭", 2: "自有", 3: "其他"}[x],
            index=1,
        )
        verification = st.selectbox(
            "验证状态",
            options=[0, 1, 2],
            format_func=lambda x: {0: "未验证", 1: "源已验证", 2: "已验证"}[x],
            index=1,
        )

    with col3:
        st.markdown("##### 📊 信用信息")
        dti = st.slider("负债收入比（%）", 0.0, 60.0, 18.0, step=0.5)
        fico_low = st.slider("FICO 分数下限", 580, 850, 690, step=5)
        delinq = st.number_input("近 2 年逾期次数", 0, 20, 0)
        open_acc = st.number_input("未结信用额度数", 0, 100, 11)
        revol_util = st.slider("循环额度利用率（%）", 0.0, 100.0, 50.0, step=1.0)

    submitted = st.form_submit_button("🚀 开始风险评估", type="primary", use_container_width=True)


# ===== 表单提交：跑预测并写入 session_state =====
if submitted:
    # 构造完整的输入数据（其他字段用默认值）
    input_data = {
        "loanAmnt": loan_amnt,
        "term": term,
        "interestRate": interest_rate,
        "installment": loan_amnt * (interest_rate / 1200) / (1 - (1 + interest_rate / 1200) ** (-term * 12)),
        "grade": grade,
        "subGrade": grade + "3",
        "employmentTitle": 100000.0,
        "employmentLength": employment_length,
        "homeOwnership": home_ownership,
        "annualIncome": annual_income,
        "verificationStatus": verification,
        "issueDate": "2019-06-01",
        "purpose": purpose,
        "postCode": 100.0,
        "regionCode": 10,
        "dti": dti,
        "delinquency_2years": delinq,
        "ficoRangeLow": fico_low,
        "ficoRangeHigh": fico_low + 4,
        "openAcc": open_acc,
        "pubRec": 0.0,
        "pubRecBankruptcies": 0.0,
        "revolBal": 8000.0,
        "revolUtil": revol_util,
        "totalAcc": 25.0,
        "initialListStatus": 0,
        "applicationType": 0,
        "earliesCreditLine": "Jan-2010",
        "title": 0.0,
        "policyCode": 1.0,
    }
    # 匿名特征
    for i in range(15):
        input_data[f"n{i}"] = 0.0

    df_input = pd.DataFrame([input_data])

    with st.spinner("🔮 模型推理中..."):
        # 1-3. 使用与批量数据相同的标准字段顺序执行预处理、特征工程和筛选。
        # 直接把表单字段交给旧预处理器会因列顺序不同触发 sklearn 校验错误。
        prediction_resources = load_prediction_resources(str(ROOT))
        if prediction_resources is None:
            st.error("模型产物不完整，请先完成模型训练。")
            st.stop()
        X = prepare_model_features(df_input, prediction_resources)

        # 4. 预测
        proba = xgb_model.predict_proba(X)[0]
        prob_pct = proba * 100

        # 5. SHAP 解释
        analyzer = load_shap_analyzer(
            xgb_model,
            active_source,
            data_source_version(ROOT, active_source),
        )
        explanation = analyzer.explain_single(X.iloc[0]) if analyzer else None

        # 6. 风险分级
        if prob_pct < 30:
            risk_level, risk_color, recommendation = "低风险", "green", "✅ 建议通过"
        elif prob_pct < 70:
            risk_level, risk_color, recommendation = "中风险", "orange", "⚠️ 人工复核"
        else:
            risk_level, risk_color, recommendation = "高风险", "red", "❌ 建议拒绝"

        # 7.5 A/B 测试：A 组使用原 XGBoost，B 组使用新 Random Forest。
        ab_result = None
        if random_forest_model is not None:
            from src.ab_test import ABTestLogger, ABTestRouter

            branch, branch_proba = ABTestRouter(
                xgb_model, random_forest_model, ratio=0.5
            ).route(application_id, X)
            ab_result = {
                "application_id": int(application_id),
                "branch": branch,
                "model": "XGBoost" if branch == "A" else "Random Forest",
                "probability": float(branch_proba[0]),
            }
            ABTestLogger(ROOT / "logs" / "ab_test_predictions.jsonl").log(
                user_id=application_id,
                branch=branch,
                model_version=ab_result["model"],
                prediction=ab_result["probability"],
                eventual_outcome=None,
                data_source=active_source,
            )

    # 关键：把预测结果写到 session_state，确保跨 rerun 持久化
    st.session_state["prediction"] = {
        "input_data": input_data,
        "proba": proba,
        "prob_pct": prob_pct,
        "explanation": explanation,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "ab_result": ab_result,
        "source": active_source,
    }
    # 提交新预测时清掉旧报告，避免新预测看到旧报告
    st.session_state.pop("llm_report", None)
    st.session_state.pop("compliance_result", None)


# ===== 展示结果（从 session_state 读，不依赖 submitted） =====
if "prediction" in st.session_state:
    pred = st.session_state["prediction"]
    input_data = pred["input_data"]
    proba = pred["proba"]
    prob_pct = pred["prob_pct"]
    explanation = pred["explanation"]
    risk_level = pred["risk_level"]
    risk_color = pred["risk_color"]
    recommendation = pred["recommendation"]
    ab_result = pred.get("ab_result")

    st.divider()

    # 顶部操作条：清除按钮（重新评估时用）
    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.markdown("## 📊 评估结果")
    with col_clear:
        if st.button("🗑️ 清除结果", use_container_width=True):
            st.session_state.pop("prediction", None)
            st.session_state.pop("llm_report", None)
            st.session_state.pop("compliance_result", None)
            st.rerun()

    # 三栏：仪表盘、关键指标、SHAP
    col1, col2, col3 = st.columns([1.2, 1, 2])

    with col1:
        fig_gauge = gauge_chart(prob_pct, title="违约概率")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; height: 100%;'>
            <h4>风险等级</h4>
            <h2 style='color: {risk_color};'>{risk_level}</h2>
            <hr>
            <h4>建议</h4>
            <h3>{recommendation}</h3>
            <hr>
            <p style='font-size: 0.85em; color: gray;'>
                违约概率: {prob_pct:.2f}%<br>
                建议月供: ¥{input_data['installment']:,.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if explanation:
            fig_waterfall = shap_waterfall(
                explanation.feature_names,
                explanation.feature_values,
                explanation.shap_values,
                explanation.base_value,
                top_n=8,
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
        else:
            st.info("SHAP 分析器加载失败")

    # ===== Top 特征贡献表 =====
    if explanation:
        st.markdown("### 📋 关键特征贡献")
        top_features = explanation.get_top_features(top_n=10)
        df_top = pd.DataFrame(top_features)
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    # ===== A/B 测试结果 =====
    st.markdown("### 🧪 A/B 模型路由")
    st.caption("A 组 = XGBoost；B 组 = Random Forest。路由使用开发指导书规定的一致性哈希。")
    if ab_result is None:
        st.info("尚未生成 Random Forest 产物，请先到模型训练页重新训练。")
    else:
        col_ab1, col_ab2, col_ab3 = st.columns(3)
        col_ab1.metric("一致性哈希分组", ab_result["branch"])
        col_ab2.metric("实际调用模型", ab_result["model"])
        col_ab3.metric("A/B 输出违约概率", f"{ab_result['probability']:.2%}")
        st.caption(
            f"申请编号 {ab_result['application_id']} 使用 MD5 一致性哈希路由；"
            "同一编号每次进入同一分组。"
        )

        from src.ab_test import ABTestLogger

        ab_summary, ab_metrics = ABTestLogger(
            ROOT / "logs" / "ab_test_predictions.jsonl"
        ).analyze(data_source=active_source)
        if not ab_summary.empty:
            display_summary = ab_summary.copy()
            for column in ["平均预测概率", "最低预测概率", "最高预测概率"]:
                display_summary[column] = display_summary[column].map(
                    lambda value: f"{value:.2%}"
                )
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
        if ab_metrics.empty:
            st.info("暂无足够的实际结果，不能进行 AUC、KS、F1、Accuracy 效果评价。")
        else:
            st.markdown("#### 基于真实终态标签的 A/B 效果")
            st.dataframe(ab_metrics.round(4), use_container_width=True, hide_index=True)

    # ===== 政策合规检查 Agent =====
    st.divider()
    st.markdown("## ⚖️ 政策合规检查")
    if st.button("⚖️ 调用合规 Agent", use_container_width=True):
        try:
            from src.llm.client import DeepSeekClient
            from src.llm.prompts import PromptManager
            from src.llm.agents.compliance_agent import ComplianceAgent

            llm_cfg = configs.get("llm")
            agent = ComplianceAgent(
                DeepSeekClient(llm_cfg),
                PromptManager(llm_cfg),
                llm_cfg.agents.compliance,
            )
            with st.spinner("正在逐条检查合规要求..."):
                st.session_state["compliance_result"] = agent.check(
                    features=input_data,
                    decision=recommendation,
                    interest_rate=input_data["interestRate"],
                )
        except Exception as e:
            st.error(f"合规检查失败: {e}")

    if "compliance_result" in st.session_state:
        compliance = st.session_state["compliance_result"]
        violations = compliance.get("violations") or []
        warnings = compliance.get("warnings") or []
        recommendations = compliance.get("recommendations") or []

        st.markdown("### 检查结果")
        if compliance.get("is_compliant"):
            st.success("✅ 合规 Agent 判定：通过")
        else:
            st.warning("⚠️ 合规 Agent 判定：存在违规或需复核项")

        col_violation, col_warning = st.columns(2)
        with col_violation:
            st.markdown("#### 🚫 违规事项")
            if violations:
                for index, item in enumerate(violations, start=1):
                    if isinstance(item, dict):
                        rule = item.get("rule", "未注明规则")
                        reason = item.get("reason", "未注明原因")
                        st.error(f"**{index}. {rule}**\n\n{reason}")
                    else:
                        st.error(f"{index}. {item}")
            else:
                st.success("未发现明确违规事项")

        with col_warning:
            st.markdown("#### ⚠️ 需要注意")
            if warnings:
                for index, item in enumerate(warnings, start=1):
                    st.warning(f"{index}. {item}")
            else:
                st.info("没有额外警告")

        st.markdown("#### 💡 处理建议")
        if recommendations:
            for index, item in enumerate(recommendations, start=1):
                st.markdown(f"{index}. {item}")
        else:
            st.info("暂无补充处理建议")

    # ===== LLM 风险报告 =====
    st.divider()
    st.markdown("## 📝 智能风险评估报告")

    # 已经有报告就直接显示，没有就给生成按钮
    if "llm_report" not in st.session_state:
        col_gen, col_hint = st.columns([1, 3])
        with col_gen:
            generate_clicked = st.button(
                "🤖 调用 LLM 生成报告",
                type="primary",
                use_container_width=True,
                help="约 5-15 秒",
            )
        with col_hint:
            st.caption("💡 报告由 DeepSeek 基于上方预测结果和 SHAP 解释生成，含审批建议、风险因素分析、补充建议等。")

        if generate_clicked:
            try:
                from src.llm.client import DeepSeekClient
                from src.llm.prompts import PromptManager
                from src.llm.agents.report_agent import ReportAgent

                llm_cfg = configs.get("llm")
                client = DeepSeekClient(llm_cfg)
                pm = PromptManager(llm_cfg)
                agent = ReportAgent(client, pm, llm_cfg.agents.report_agent)

                # 流式输出
                report_placeholder = st.empty()
                full_report = ""

                with st.spinner("LLM 分析中..."):
                    stream = agent.generate(
                        features=input_data,
                        probability=proba,
                        shap_explanation=explanation,
                        stream=True,
                        model_name="XGBoost",
                    )
                    for chunk in stream:
                        full_report += chunk
                        report_placeholder.markdown(full_report + "▌")
                    report_placeholder.markdown(full_report)

                required_sections = [
                    "风险摘要", "关键风险因素", "保护因素", "审批建议", "补充建议", "免责声明",
                ]
                missing_sections = [
                    section for section in required_sections if section not in full_report
                ]
                if not full_report.strip():
                    raise RuntimeError("LLM 返回正文为空，未生成风险报告")
                if missing_sections:
                    raise RuntimeError(
                        "LLM 报告缺少原模板规定章节：" + "、".join(missing_sections)
                    )

                # 写入 session_state，下次 rerun 直接显示，不会回到生成按钮
                st.session_state["llm_report"] = full_report
                st.rerun()  # 立即刷新，让下载按钮和报告进入"已生成"状态

            except Exception as e:
                st.error(f"LLM 调用失败: {e}")
                st.caption("请检查 .env 中的 DEEPSEEK_API_KEY")
    else:
        # 已生成：展示报告 + 下载/重新生成按钮
        full_report = st.session_state["llm_report"]

        # 顶部操作条
        col_dl1, col_dl2, col_regen = st.columns([1, 1, 1])

        # 拼一个"完整版"报告（含输入/预测/SHAP/LLM 全部信息）
        complete_report = _build_complete_report(
            input_data=input_data,
            proba=proba,
            risk_level=risk_level,
            recommendation=recommendation,
            explanation=explanation,
            llm_report=full_report,
        )

        with col_dl1:
            st.download_button(
                "💾 下载完整报告（推荐）",
                data=complete_report.encode("utf-8"),
                file_name=f"风险评估报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True,
                help="包含申请人信息、模型预测、SHAP 解释、LLM 分析的完整 Markdown 报告",
            )
        with col_dl2:
            st.download_button(
                "📄 仅下载 LLM 报告",
                data=full_report.encode("utf-8"),
                file_name=f"LLM报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_regen:
            if st.button("🔄 重新生成报告", use_container_width=True):
                st.session_state.pop("llm_report", None)
                st.rerun()

        # 页面展示与下载使用同一份完整报告，确保模型、SHAP 与 LLM 章节齐全。
        with st.expander("📖 查看完整报告内容", expanded=True):
            st.markdown(complete_report)
