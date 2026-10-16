"""
智能信贷风控分析系统 - Streamlit 主入口
======================================
运行: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

# 添加项目根到 sys.path（Streamlit 多页面应用必需）
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ===== 页面配置 =====
st.set_page_config(
    page_title="智能信贷风控分析系统",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "## 智能信贷风控分析系统\n机器学习课程综合实战项目"
    }
)

# ===== 加载样式 =====
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ===== 全局初始化（带缓存） =====
@st.cache_resource
def init_system():
    """初始化系统（仅在首次访问时执行）"""
    from src.utils.config_loader import ConfigLoader
    return ConfigLoader(str(ROOT / "configs"))


configs = init_system()
st.session_state["configs"] = configs

# ===== 侧边栏 =====
from app.components.sidebar import render_sidebar
render_sidebar()

# ===== 主页面 =====
st.title("💳 智能信贷风控分析系统")
st.markdown("##### 基于 XGBoost + PyTorch MLP + SHAP + DeepSeek LLM 的端到端风控解决方案")

# 系统状态指示
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

# 检查模型状态
model_dir = ROOT / "data" / "models" / "latest"
has_model = model_dir.exists() and any(model_dir.glob("*.json"))

# 检查数据状态
sample_data = ROOT / "data" / "sample" / "sample.csv"
real_data = ROOT / "data" / "raw" / "train.csv"
lending_data = ROOT / "data" / "raw" / "lending_club.csv"
has_data = sample_data.exists() or real_data.exists() or lending_data.exists()

# 检查 LLM 配置
import os
has_api_key = os.environ.get("DEEPSEEK_API_KEY", "").startswith("sk-") and \
              os.environ.get("DEEPSEEK_API_KEY") != "sk-your-api-key-here"

with status_col1:
    st.metric(
        "📦 数据状态",
        "已就绪" if has_data else "未就绪",
        delta="✅ OK" if has_data else "⚠️ 需要数据",
        delta_color="normal" if has_data else "inverse",
    )
with status_col2:
    st.metric(
        "🤖 模型状态",
        "已训练" if has_model else "未训练",
        delta="✅ OK" if has_model else "⚠️ 需训练",
        delta_color="normal" if has_model else "inverse",
    )
with status_col3:
    st.metric(
        "🔑 LLM API",
        "已配置" if has_api_key else "未配置",
        delta="✅ OK" if has_api_key else "⚠️ 需 API Key",
        delta_color="normal" if has_api_key else "inverse",
    )
with status_col4:
    st.metric("⚙️ 系统版本", "v1.0.0", delta="学习版")

st.divider()

# ===== 项目介绍 =====
st.markdown("""
## 🎯 项目简介

本系统是一个**完整的金融风控分析平台**，模拟互联网金融公司的贷款审批流程，
集成了机器学习课程涵盖的全部核心技术。

### 系统能力

| 模块 | 技术栈 | 应用场景 |
|------|--------|---------|
| 数据探索 | Pandas + Plotly | 数据质量分析、特征分布可视化 |
| 特征工程 | Sklearn | 缺失值/异常值/编码/标准化 |
| 模型训练 | XGBoost + PyTorch MLP | 双模型融合，指标以实际训练结果为准 |
| 超参优化 | Optuna | 自动搜索最优超参 |
| 模型解释 | SHAP | 全局/局部特征贡献分析 |
| 智能助手 | DeepSeek LLM | 风险报告 / 特征建议 / 案例诊断 |

### 数据集

> **天池"零基础入门金融风控-贷款违约预测"**
> 来自某信贷平台 80 万条贷款记录，47 列特征，预测用户违约概率。
""")

st.divider()

# ===== 快速导航 =====
st.markdown("## 🚀 快速开始")

quick_col1, quick_col2 = st.columns(2)

with quick_col1:
    st.markdown("""
    ### 第一次使用？
    1. **生成示例数据**
       ```bash
       python scripts/generate_sample_data.py --n_samples 10000
       ```
    2. **训练模型**
       ```bash
       python scripts/train_pipeline.py --use-sample
       ```
    3. **配置 LLM**
       在 `.env` 中填入你的 DeepSeek API Key
    """)

with quick_col2:
    st.markdown("""
    ### 使用本系统
    通过左侧导航栏访问各功能模块：
    - 📊 **数据探索**：理解你的数据
    - 🔧 **特征工程**：可视化特征处理
    - 🤖 **模型训练**：交互式训练和评估
    - 🎯 **单笔预测**：贷款审批演示
    - 🔍 **模型解释**：SHAP 分析
    - 💬 **LLM 助手**：智能问答
    - ⚙️ **系统配置**：在线修改配置
    """)

st.divider()

# ===== 学习资源 =====
with st.expander("📚 学习资源 / 项目文档"):
    st.markdown("""
    ### 课程章节对应

    - **第 1-2 章** XGBoost：详见 `src/models/xgb_model.py` 和模型训练页面
    - **第 3 章** 超参优化 + PyTorch MLP：详见 `src/tuning/` 和 `src/models/mlp_model.py`
    - **第 4 章** LLM API 应用：详见 `src/llm/` 和 LLM 助手页面

    ### 推荐阅读顺序

    1. 先阅读 `README.md` 了解整体架构
    2. 通读 `configs/` 下所有 YAML 配置（理解项目的"参数面板"）
    3. 跟着 `src/pipeline/orchestrator.py` 看完整训练流程
    4. 重点研读 `src/llm/` 下的 LLM 集成代码

    ### 学习要点

    - 📐 **配置驱动**：所有参数外部化，无硬编码
    - 🧩 **模块解耦**：每个模块独立可测试可替换
    - 🔌 **统一接口**：模型、Agent 都遵循相同接口规范
    - 📝 **可解释性**：从 SHAP 到 LLM 报告，让模型决策透明
    """)

# ===== 页脚 =====
st.markdown("""
---
<div style='text-align: center; color: gray; font-size: 0.85em;'>
    💳 智能信贷风控分析系统 · 学习版 v1.0.0 · 仅用于学习研究<br>
    本系统生成的报告和分析结果不构成任何金融建议
</div>
""", unsafe_allow_html=True)
