"""
侧边栏组件
==========
所有页面共享的侧边栏。
"""
import os
from pathlib import Path

import streamlit as st


def render_sidebar():
    """渲染统一的侧边栏"""
    # Streamlit Community Cloud 的根级 Secret 同步为环境变量，供原有
    # ConfigLoader/DeepSeekClient 继续按开发指导书约定读取；值不会显示或写盘。
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass

    with st.sidebar:
        st.markdown("## 💳 风控系统")
        st.caption("机器学习课程综合实战")

        st.divider()

        # 模型状态
        st.markdown("### 📊 系统状态")
        ROOT = Path(__file__).parent.parent.parent
        model_dir = ROOT / "data" / "models" / "latest"
        source_labels = {
            "tianchi": "真实天池数据",
            "sample": "合成示例数据",
            "lending_club": "Lending Club 真实数据",
            "lending_club_upload": "本次上传的 Lending Club 数据",
        }
        active_source = st.session_state.get("active_data_source")
        if active_source in source_labels:
            st.caption(f"📦 当前分析数据：{source_labels[active_source]}")

        if model_dir.exists():
            import json
            meta_path = model_dir / "meta.json"
            eval_path = model_dir / "eval_results.json"

            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                st.caption(f"📅 训练于: {meta.get('timestamp', '')[:19]}")
                st.caption(f"🧾 模型训练数据：{meta.get('data_source', '未记录')}")
                st.caption(f"📦 模型训练样本: {meta.get('train_samples', 0):,}")
                st.caption(f"🔢 特征数: {len(meta.get('feature_names', []))}")

            if eval_path.exists():
                eval_results = json.loads(eval_path.read_text(encoding="utf-8"))
                if "ensemble" in eval_results:
                    auc = eval_results["ensemble"].get("auc", 0)
                    st.metric("🎯 已保存模型 Ensemble AUC", f"{auc:.4f}")
                elif "xgboost" in eval_results:
                    auc = eval_results["xgboost"].get("auc", 0)
                    st.metric("🎯 已保存模型 XGBoost AUC", f"{auc:.4f}")
        else:
            st.warning("⚠️ 模型未训练")
            st.caption("运行: `python scripts/train_pipeline.py --use-sample`")

        st.divider()

        # LLM 状态
        st.markdown("### 🤖 LLM 状态")
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key.startswith("sk-") and api_key != "sk-your-api-key-here":
            st.success("✅ DeepSeek 已配置")
        else:
            st.warning("⚠️ 未配置 API Key")
            st.caption("在 `.env` 中填入 `DEEPSEEK_API_KEY`")

        st.divider()

        # 帮助
        with st.expander("📖 使用帮助"):
            st.markdown("""
            **页面说明：**
            - 数据探索：了解数据
            - 特征工程：处理特征
            - 模型训练：训练 + 评估
            - 单笔预测：贷款审批演示
            - 模型解释：SHAP 分析
            - LLM 助手：智能对话
            - 系统配置：修改参数
            """)

        st.caption("📚 v1.0.0 · 仅用于学习研究")
