"""
Prompt 管理器
=============
集中管理所有 Prompt 模板，使用 Jinja2 实现变量插值。

要点:
    Prompt 工程是 LLM 应用的核心，
    将 Prompt 与代码分离便于版本管理、AB 测试、迭代优化。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, Template

from src.utils.logger import get_logger

logger = get_logger("llm.prompts")


class PromptManager:
    """
    Prompt 模板管理器。

    使用示例:
        >>> pm = PromptManager(llm_config)
        >>> prompt = pm.render("risk_report", features=..., probability=...)
        >>> system = pm.get_system_role("risk_report")
    """

    def __init__(self, config):
        self.config = config
        self.templates_dir = Path(config.prompts.templates_dir)
        self.prompts_cfg = config.prompts

        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Prompt 模板目录不存在: {self.templates_dir}")

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # 缓存模板
        self._template_cache: Dict[str, Template] = {}

    def render(self, template_name: str, **kwargs) -> str:
        """
        渲染指定模板。

        参数:
            template_name: 模板名称（对应 llm_config.yaml 中的配置键）
            **kwargs: 模板变量
        """
        cfg = getattr(self.prompts_cfg, template_name, None)
        if cfg is None:
            raise KeyError(f"未配置的 Prompt 模板: {template_name}")

        template_file = cfg.template_file
        template = self._get_template(template_file)

        try:
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"渲染模板 {template_name} 失败: {e}")
            raise

    def get_system_role(self, template_name: str) -> str:
        """获取指定 Prompt 对应的 system role"""
        cfg = getattr(self.prompts_cfg, template_name, None)
        if cfg is None:
            return ""
        return cfg.get("system_role", "")

    def _get_template(self, template_file: str) -> Template:
        """获取模板对象（带缓存）"""
        if template_file not in self._template_cache:
            self._template_cache[template_file] = self.env.get_template(template_file)
        return self._template_cache[template_file]

    def reload(self) -> None:
        """重新加载所有模板（开发用）"""
        self._template_cache.clear()
