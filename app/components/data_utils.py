"""
数据加载工具
============
所有 Streamlit 页面共用的数据加载函数，统一回退逻辑：
    1. 优先用真实数据 data/raw/train.csv（天池下载）
    2. 其次用合成数据 data/sample/sample.csv（脚本生成）
    3. 都没有时给出明确的提示
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import pandas as pd
import streamlit as st
import joblib


SOURCE_LABELS = {
    "tianchi": "真实天池数据",
    "sample": "合成示例数据",
    "lending_club": "Lending Club 真实数据",
    "lending_club_upload": "本次上传的 Lending Club 数据",
}


def available_data_sources(root: Path) -> Dict[str, str]:
    """返回当前实际存在、可被所有页面使用的数据源。"""
    sources: Dict[str, str] = {}
    if (root / "data" / "raw" / "train.csv").exists():
        sources["tianchi"] = SOURCE_LABELS["tianchi"]
    if (root / "data" / "sample" / "sample.csv").exists():
        sources["sample"] = SOURCE_LABELS["sample"]
    if (root / "data" / "raw" / "lending_club.csv").exists():
        sources["lending_club"] = SOURCE_LABELS["lending_club"]
    if "uploaded_lending_df" in st.session_state:
        sources["lending_club_upload"] = SOURCE_LABELS["lending_club_upload"]
    return sources


def initialize_data_source_state(root: Path) -> str:
    """初始化并校验跨页面共享的数据源状态。"""
    sources = available_data_sources(root)
    if not sources:
        return ""
    selected = st.session_state.get("active_data_source")
    if selected not in sources:
        selected = next(iter(sources))
        st.session_state["active_data_source"] = selected
    st.session_state.setdefault("active_model", "XGBoost")
    return selected


def set_active_data_source(root: Path, source_key: str) -> None:
    """切换全局数据源，并清除所有依赖旧数据的页面结果。"""
    if source_key not in available_data_sources(root):
        raise ValueError(f"不可用的数据源: {source_key}")
    if st.session_state.get("active_data_source") == source_key:
        return
    st.session_state["active_data_source"] = source_key
    st.session_state["data_source_revision"] = (
        int(st.session_state.get("data_source_revision", 0)) + 1
    )
    for key in [
        "lending_prediction_df", "source_prediction_df", "prediction",
        "llm_report", "compliance_result", "knn_result",
        "feature_llm_result", "ks_result",
        "derived_feature_result", "feature_selection_result", "latest_result",
        "chat_messages", "llm_quick_report", "llm_case_result",
    ]:
        st.session_state.pop(key, None)


def data_source_version(root: Path, source_key: str) -> float:
    """缓存版本：文件发生变化时强制重新读取。"""
    paths = {
        "tianchi": root / "data" / "raw" / "train.csv",
        "sample": root / "data" / "sample" / "sample.csv",
        "lending_club": root / "data" / "raw" / "lending_club.csv",
    }
    path = paths.get(source_key)
    return path.stat().st_mtime if path and path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_source_file(
    root_str: str,
    source_key: str,
    nrows: Optional[int],
    source_version: float,
) -> pd.DataFrame:
    """按数据源读取；source_key 与文件版本共同组成缓存键。"""
    from src.data.loader import DataLoader
    from src.utils.config_loader import ConfigLoader

    root = Path(root_str)
    paths = {
        "tianchi": root / "data" / "raw" / "train.csv",
        "sample": root / "data" / "sample" / "sample.csv",
        "lending_club": root / "data" / "raw" / "lending_club.csv",
    }
    if source_key not in paths or not paths[source_key].exists():
        return pd.DataFrame()
    loader = DataLoader(ConfigLoader(str(root / "configs")).get("data"))
    source_format = "lending_club" if source_key == "lending_club" else "tianchi"
    loaded = loader.load(str(paths[source_key]), source=source_format)
    return loaded.head(nrows).copy() if nrows else loaded


def load_selected_data(
    root: Path,
    source_key: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """读取指定或全局选中的数据源。"""
    source_key = source_key or initialize_data_source_state(root)
    if not source_key:
        return pd.DataFrame(), None
    if source_key == "lending_club_upload":
        uploaded = st.session_state.get("uploaded_lending_df", pd.DataFrame())
        data = uploaded.head(nrows).copy() if nrows else uploaded.copy()
    else:
        data = _load_source_file(
            str(root), source_key, nrows, data_source_version(root, source_key)
        )
    return data, SOURCE_LABELS.get(source_key, source_key)


def render_analysis_context(root: Path, model_name: Optional[str] = None) -> None:
    """统一显示当前数据源与当前模型。"""
    source_key = initialize_data_source_state(root)
    source_label = SOURCE_LABELS.get(source_key, "未选择")
    model_label = model_name or st.session_state.get("active_model", "XGBoost")
    col_source, col_model = st.columns(2)
    col_source.info(f"📦 当前数据源：**{source_label}**")
    col_model.info(f"🤖 当前模型：**{model_label}**")


def load_active_data_or_stop(root: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """页面读取全局数据源；失败时给出统一提示并停止。"""
    df, source = load_selected_data(root, nrows=nrows)
    if df.empty:
        st.error("当前数据源不可用，请先在数据探索页面重新选择。")
        st.stop()
    st.success(f"已加载 **{source}**：{len(df):,} 行 × {df.shape[1]} 列")
    return df


def find_data_file(
    root: Path,
    prefer: Literal["auto", "real", "sample"] = "auto",
) -> Tuple[Optional[Path], Optional[str]]:
    """
    查找可用的数据文件。

    参数
    ----
    root: 项目根目录
    prefer: "real"/"sample"/"auto"
        - real:   只用真实数据，没有就返回 None
        - sample: 只用合成数据，没有就返回 None
        - auto:   优先真实数据，回退到合成数据

    返回
    ----
    (path, source_label)：找到的文件路径和数据来源标签（"真实数据"/"合成数据"）
    没找到则返回 (None, None)
    """
    real_path = root / "data" / "raw" / "train.csv"
    sample_path = root / "data" / "sample" / "sample.csv"

    if prefer == "real":
        return (real_path, "真实数据") if real_path.exists() else (None, None)
    if prefer == "sample":
        return (sample_path, "合成数据") if sample_path.exists() else (None, None)

    # auto
    if real_path.exists():
        return real_path, "真实数据"
    if sample_path.exists():
        return sample_path, "合成数据"
    return None, None


@st.cache_data(show_spinner=False)
def load_data(
    root_str: str,
    prefer: str = "auto",
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    加载数据。被 @st.cache_data 装饰，同样的参数只读盘一次。

    注意 root 用 str 传入是因为 cache_data 不能哈希 Path 对象。

    参数
    ----
    root_str: 项目根目录字符串
    prefer: "auto"/"real"/"sample"
    nrows:  可选限制读取行数（大文件预览时用）

    返回
    ----
    (DataFrame, source_label)。找不到时返回 (空DF, None)。
    """
    root = Path(root_str)
    path, source = find_data_file(root, prefer)  # type: ignore[arg-type]
    if path is None:
        return pd.DataFrame(), None
    df = pd.read_csv(path, nrows=nrows)
    return df, source


def render_data_status_or_stop(
    root: Path,
    prefer: Literal["auto", "real", "sample"] = "auto",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    页面级辅助：尝试加载数据，加载成功显示状态条；失败给出明确提示并 st.stop()。
    返回成功加载的 DataFrame。
    """
    df, source = load_data(str(root), prefer=prefer, nrows=nrows)

    if df.empty:
        st.error("❌ 未找到任何可用数据")
        st.markdown(
            "请通过以下任一方式准备数据：\n\n"
            "**方式 1：使用真实天池数据**（推荐）\n"
            "```bash\n"
            "# 1. 从 https://tianchi.aliyun.com/competition/entrance/531830 报名下载\n"
            "# 2. 把 train.csv 放到下面这个位置\n"
            f"#    {root / 'data' / 'raw' / 'train.csv'}\n"
            "```\n\n"
            "**方式 2：生成合成数据**（快速测试）\n"
            "```bash\n"
            "python scripts/generate_sample_data.py --n_samples 50000\n"
            "```"
        )
        st.stop()

    # 顶栏显示数据来源
    n_rows = len(df)
    if source == "真实数据":
        st.success(f"📦 已加载 **真实天池数据**：{n_rows:,} 行 × {df.shape[1]} 列")
    else:
        st.info(f"🧪 已加载 **合成示例数据**：{n_rows:,} 行 × {df.shape[1]} 列（来自 generate_sample_data.py）")

    return df


@st.cache_resource(show_spinner=False)
def load_prediction_resources(root_str: str):
    """加载原有预处理、筛选器和 A/B 两个已训练模型。"""
    from src.models.random_forest_model import RandomForestModel
    from src.models.xgb_model import XGBModel
    from src.utils.config_loader import ConfigLoader

    root = Path(root_str)
    latest = root / "data" / "models" / "latest"
    required = [
        latest / "preprocessor.pkl",
        latest / "xgb_model.json",
        latest / "random_forest_model.joblib",
    ]
    if any(not path.exists() for path in required):
        return None

    configs = ConfigLoader(str(root / "configs"))
    selector_path = latest / "feature_selector.pkl"
    xgb_model = XGBModel(configs.get("model").xgboost).load(
        str(latest / "xgb_model.json")
    )
    rf_model = RandomForestModel(configs.get("model").random_forest).load(
        str(latest / "random_forest_model.joblib")
    )
    return {
        "configs": configs,
        "preprocessor": joblib.load(latest / "preprocessor.pkl"),
        "selector": joblib.load(selector_path) if selector_path.exists() else None,
        "xgb_model": xgb_model,
        "random_forest_model": rf_model,
    }


def prepare_model_features(df: pd.DataFrame, resources) -> pd.DataFrame:
    """让外部数据严格经过原有预处理、特征构建和特征筛选。"""
    from src.features.builder import FeatureBuilder

    configs = resources["configs"]
    data_cfg = configs.get("data")
    target = data_cfg.target_column
    id_col = data_cfg.id_column
    raw_features = df.drop(columns=[target, id_col], errors="ignore")
    # 预处理器是在 data_config 的 numerical → categorical → datetime → anonymous
    # 顺序上拟合的。Adapter 的字段集合相同但排列不同，不能直接用于 sklearn transform。
    training_order = [
        *data_cfg.columns.numerical,
        *data_cfg.columns.categorical,
        *data_cfg.columns.datetime,
        *data_cfg.columns.anonymous,
    ]
    raw_features = raw_features.reindex(columns=training_order)
    processed = resources["preprocessor"].transform(raw_features)
    processed = FeatureBuilder(configs.get("feature")).build(processed)
    if resources["selector"] is not None:
        processed = resources["selector"].transform(processed)
    return processed
