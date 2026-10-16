"""
配置加载器
=========
统一加载和管理所有 YAML 配置文件，支持环境变量插值。

要点:
    1. 配置驱动：所有参数外部化，无硬编码
    2. 单例模式：避免重复加载
    3. 环境变量插值：敏感信息通过 .env 注入
    4. 配置校验：及早发现配置错误
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """
    配置加载器（单例）。

    使用示例:
        >>> loader = ConfigLoader("./configs")
        >>> data_cfg = loader.get("data")
        >>> model_cfg = loader.get("model")
        >>> # 直接访问嵌套字段
        >>> lr = loader.get("model").xgboost.params.learning_rate
    """

    _instance: Optional["ConfigLoader"] = None
    _configs: Dict[str, "AttrDict"] = {}

    def __new__(cls, config_dir: str = "./configs"):
        # 单例：第一次调用初始化，之后返回同一实例
        if cls._instance is None:
            instance = super().__new__(cls)
            try:
                instance._initialize(config_dir)
            except Exception:
                # 初始化失败时，不要把半初始化的实例保留为单例，
                # 否则后续调用会拿到一个 _configs 为空的"假"实例。
                cls._instance = None
                raise
            cls._instance = instance
        return cls._instance

    def _initialize(self, config_dir: str) -> None:
        """初始化：加载 .env 和所有 YAML 配置"""
        # 加载环境变量
        load_dotenv()

        # 解析为绝对路径，便于错误提示
        self.config_dir = Path(config_dir).resolve()
        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"配置目录不存在: {self.config_dir}\n"
                f"提示: 请确认你在项目根目录下运行命令，"
                f"或通过 --config-dir 参数指定正确的配置目录。"
            )
        if not self.config_dir.is_dir():
            raise NotADirectoryError(
                f"配置路径不是一个目录: {self.config_dir}\n"
                f"--config-dir 参数应该传 configs/ 目录的路径，"
                f"而不是某个具体 YAML 文件的路径。\n"
                f"正确示例: --config-dir ./configs\n"
                f"错误示例: --config-dir ./configs/pipeline_config.yaml"
            )

        # 自动加载目录下所有 YAML 文件
        yaml_files = list(self.config_dir.glob("*.yaml"))
        if not yaml_files:
            # 给出明确的诊断信息，不要让上层因为空配置而抛模糊的 KeyError
            try:
                files_in_dir = [f.name for f in self.config_dir.iterdir()]
            except OSError:
                files_in_dir = ["<无法读取目录内容>"]
            raise FileNotFoundError(
                f"配置目录 {self.config_dir} 下没有找到任何 *.yaml 文件。\n"
                f"目录现有内容: {files_in_dir}\n"
                f"提示:\n"
                f"  1. 确认你在项目根目录运行（cwd 应包含 configs/、src/、app/ 等子目录）\n"
                f"  2. 确认 configs/ 目录中存在 data_config.yaml 等 5 个 YAML 文件\n"
                f"  3. Windows 用户：检查文件扩展名是否被系统隐藏成了 .yaml.txt"
            )

        for yaml_file in yaml_files:
            config_name = yaml_file.stem.replace("_config", "")
            self._configs[config_name] = self._load_yaml(yaml_file)

    def _load_yaml(self, file_path: Path) -> "AttrDict":
        """加载单个 YAML 文件，并进行环境变量插值"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 环境变量插值: ${VAR_NAME} 或 ${VAR_NAME:-default_value}
        content = self._interpolate_env_vars(content)

        data = yaml.safe_load(content)
        return AttrDict(data) if data else AttrDict({})

    @staticmethod
    def _interpolate_env_vars(content: str) -> str:
        """
        替换 YAML 中的环境变量占位符。

        支持两种语法:
            ${VAR_NAME}              -> 不存在则保持原文
            ${VAR_NAME:-default}     -> 不存在则使用 default
        """
        pattern = re.compile(r"\$\{([^}^{]+)\}")

        def replacer(match):
            expr = match.group(1)
            if ":-" in expr:
                var_name, default = expr.split(":-", 1)
                return os.environ.get(var_name.strip(), default.strip())
            return os.environ.get(expr.strip(), match.group(0))

        return pattern.sub(replacer, content)

    def get(self, name: str) -> "AttrDict":
        """获取指定配置"""
        if name not in self._configs:
            available = list(self._configs.keys())
            raise KeyError(f"配置 '{name}' 不存在，可用配置: {available}")
        return self._configs[name]

    def reload(self) -> None:
        """热重载所有配置（开发场景使用）"""
        self._configs = {}
        self._initialize(str(self.config_dir))

    def __repr__(self) -> str:
        return f"ConfigLoader(dir={self.config_dir}, configs={list(self._configs.keys())})"


class AttrDict(dict):
    """
    支持属性访问的字典。

    使用示例:
        >>> d = AttrDict({"a": {"b": 1}})
        >>> d.a.b   # -> 1
        >>> d["a"]["b"]   # -> 1（保留字典原生访问方式）
    """

    def __init__(self, data: Optional[Dict] = None):
        super().__init__()
        if data:
            for key, value in data.items():
                self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        """递归包装嵌套的 dict/list"""
        if isinstance(value, dict):
            return AttrDict(value)
        elif isinstance(value, list):
            return [AttrDict._wrap(item) for item in value]
        return value

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        raise AttributeError(f"配置项 '{key}' 不存在")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)


# ===== 便捷函数 =====
def load_configs(config_dir: str = "./configs") -> ConfigLoader:
    """便捷函数：加载所有配置"""
    return ConfigLoader(config_dir)


if __name__ == "__main__":
    # 自测
    loader = ConfigLoader("./configs")
    print("已加载的配置:", list(loader._configs.keys()))
    print("XGBoost 学习率:", loader.get("model").xgboost.params.learning_rate)
    print("DeepSeek 模型:", loader.get("llm").deepseek.model)
