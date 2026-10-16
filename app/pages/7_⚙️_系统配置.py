"""
⚙️ 系统配置页面
================
在线查看、修改 YAML 配置（教学演示用）。
"""
import sys
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar

st.set_page_config(page_title="系统配置", page_icon="⚙️", layout="wide")
render_sidebar()

st.title("⚙️ 系统配置")
st.caption("查看和编辑系统的所有 YAML 配置文件")

st.warning("⚠️ **教学演示页面** - 修改配置会影响后续训练，请谨慎操作。生产环境建议通过版本控制管理配置。")

# ===== 配置文件列表 =====
config_dir = ROOT / "configs"
config_files = sorted(config_dir.glob("*.yaml"))

if not config_files:
    st.error(f"未找到配置文件: {config_dir}")
    st.stop()

selected_file = st.selectbox(
    "选择配置文件",
    config_files,
    format_func=lambda x: x.name,
)

# ===== 显示文件内容 =====
content = selected_file.read_text(encoding="utf-8")

tab1, tab2 = st.tabs(["📝 编辑模式", "🌳 树形视图"])

with tab1:
    st.markdown(f"### 文件: `configs/{selected_file.name}`")

    edited = st.text_area(
        "配置内容",
        value=content,
        height=600,
        key=f"editor_{selected_file.name}",
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ 验证 YAML", use_container_width=True):
            try:
                yaml.safe_load(edited)
                st.success("✅ YAML 格式正确")
            except yaml.YAMLError as e:
                st.error(f"❌ YAML 格式错误: {e}")

    with col2:
        if st.button("💾 保存", type="primary", use_container_width=True):
            try:
                # 验证后保存
                yaml.safe_load(edited)
                selected_file.write_text(edited, encoding="utf-8")
                st.success(f"✅ 已保存到 {selected_file.name}")
                st.balloons()
                # 清缓存以重新加载
                st.cache_resource.clear()
            except yaml.YAMLError as e:
                st.error(f"❌ YAML 格式错误，未保存: {e}")
            except Exception as e:
                st.error(f"❌ 保存失败: {e}")

with tab2:
    st.markdown(f"### 文件: `configs/{selected_file.name}` (树形视图)")
    try:
        data = yaml.safe_load(content)
        st.json(data, expanded=True)
    except yaml.YAMLError as e:
        st.error(f"YAML 解析失败: {e}")

# ===== Prompt 模板 =====
st.divider()
st.markdown("## 📜 Prompt 模板")

prompts_dir = config_dir / "prompts"
if prompts_dir.exists():
    prompt_files = sorted(prompts_dir.glob("*.j2"))

    if prompt_files:
        selected_prompt = st.selectbox(
            "选择 Prompt 模板",
            prompt_files,
            format_func=lambda x: x.name,
        )

        prompt_content = selected_prompt.read_text(encoding="utf-8")
        st.text_area(
            "模板内容",
            value=prompt_content,
            height=400,
            key=f"prompt_{selected_prompt.name}",
            help="Jinja2 模板：{{ variable }} 表示插值，{% for %} 等表示控制结构",
        )

# ===== 环境变量 =====
st.divider()
st.markdown("## 🔐 环境变量")

import os

env_vars = {
    "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
    "DEEPSEEK_BASE_URL": os.environ.get("DEEPSEEK_BASE_URL", ""),
    "DEEPSEEK_MODEL": os.environ.get("DEEPSEEK_MODEL", ""),
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", ""),
}

env_data = []
for key, value in env_vars.items():
    if "KEY" in key and value:
        # 脱敏显示
        masked = value[:6] + "..." + value[-4:] if len(value) > 12 else "***"
        env_data.append({"变量名": key, "值": masked})
    else:
        env_data.append({"变量名": key, "值": value or "(未设置)"})

import pandas as pd
st.dataframe(pd.DataFrame(env_data), use_container_width=True, hide_index=True)

st.caption("💡 修改环境变量请编辑项目根目录的 `.env` 文件后重启应用")
