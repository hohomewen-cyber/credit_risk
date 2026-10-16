"""
DeepSeek API 客户端
====================
封装 DeepSeek 大模型 API 调用，支持流式输出、重试、限流等。

要点:
    1. 使用 OpenAI SDK（DeepSeek 兼容 OpenAI 接口）
    2. 重试机制（tenacity）
    3. 流式输出（Streamlit 实时显示）
    4. 审计日志（合规要求）
    5. 异常隔离（LLM 失败不影响主流程）
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.utils.logger import get_logger

logger = get_logger("llm.client")


class DeepSeekClient:
    """
    DeepSeek API 客户端。

    使用示例:
        >>> client = DeepSeekClient(llm_config)
        >>> # 普通调用
        >>> response = client.chat("解释一下违约概率是什么")
        >>> # 流式调用
        >>> for chunk in client.stream_chat("解释一下违约概率"):
        ...     print(chunk, end="")
    """

    def __init__(self, config):
        self.config = config
        self.deepseek_cfg = config.deepseek

        # 从环境变量读取 API Key
        api_key_env = self.deepseek_cfg.api_key_env
        api_key = os.environ.get(api_key_env)
        if not api_key or api_key == "sk-your-api-key-here":
            logger.warning(
                f"未找到有效的 {api_key_env}。\n"
                f"请在 .env 中配置真实的 DeepSeek API Key。\n"
                f"申请地址: https://platform.deepseek.com"
            )

        # 初始化 OpenAI 兼容客户端
        self.client = OpenAI(
            api_key=api_key or "dummy-key",
            base_url=self.deepseek_cfg.base_url,
            timeout=self.deepseek_cfg.default_params.timeout,
        )

        self.model = self.deepseek_cfg.model
        self.default_params = dict(self.deepseek_cfg.default_params)
        self.default_params.pop("timeout", None)  # timeout 在 client 上设置

        # 限流计数器
        self._request_times: List[float] = []

        # 审计日志路径
        self.audit_log_path = None
        if hasattr(config, "safety") and config.safety.audit_log.enabled:
            self.audit_log_path = Path(config.safety.audit_log.log_file)
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """
        非流式对话。

        参数:
            prompt: 用户消息
            system_prompt: 系统消息（可选）
            history: 历史对话 [{"role": "user/assistant", "content": "..."}]
            **kwargs: 覆盖默认参数
        """
        self._check_rate_limit()

        messages = self._build_messages(prompt, system_prompt, history)
        params = {**self.default_params, **kwargs}

        # 安全过滤
        prompt = self._sanitize_input(prompt)

        try:
            t0 = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                **params,
            )
            duration = time.time() - t0

            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else {}

            self._log_audit(prompt, content, duration, usage)
            logger.info(f"LLM 调用完成 - 耗时 {duration:.2f}s, tokens={usage}")
            return content

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def stream_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式对话（用于实时显示）。

        使用示例:
            >>> for chunk in client.stream_chat("..."):
            ...     print(chunk, end="", flush=True)
        """
        self._check_rate_limit()

        messages = self._build_messages(prompt, system_prompt, history)
        params = {**self.default_params, **kwargs}

        prompt = self._sanitize_input(prompt)
        full_response = ""
        t0 = time.time()

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **params,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_response += delta.content
                    yield delta.content

            duration = time.time() - t0
            self._log_audit(prompt, full_response, duration, {"stream": True})
            logger.info(f"LLM 流式调用完成 - 耗时 {duration:.2f}s")

        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}")
            yield f"\n\n[错误]: {str(e)}"

    # -------- 内部辅助方法 --------

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """构建符合 OpenAI 格式的消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _check_rate_limit(self) -> None:
        """简单的限流检查（每分钟请求数）"""
        if not self.deepseek_cfg.rate_limit.enabled:
            return

        now = time.time()
        # 仅保留 60 秒内的请求记录
        self._request_times = [t for t in self._request_times if now - t < 60]

        max_rpm = self.deepseek_cfg.rate_limit.requests_per_minute
        if len(self._request_times) >= max_rpm:
            wait_time = 60 - (now - self._request_times[0])
            logger.warning(f"达到限流阈值 ({max_rpm}/min)，等待 {wait_time:.1f}s")
            time.sleep(max(wait_time, 1))

        self._request_times.append(now)

    def _sanitize_input(self, text: str) -> str:
        """敏感信息过滤"""
        if not hasattr(self.config, "safety") or not self.config.safety.content_filter.enabled:
            return text

        import re
        for pattern in self.config.safety.content_filter.block_patterns:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
        return text

    def _log_audit(
        self,
        prompt: str,
        response: str,
        duration: float,
        usage: Dict,
    ) -> None:
        """审计日志（合规需要）"""
        if self.audit_log_path is None:
            return

        # 控制日志大小
        log_response = self.config.safety.audit_log.log_response if hasattr(self.config, "safety") else False

        record = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "duration_sec": round(duration, 2),
            "usage": usage,
            "prompt_preview": prompt[:200],
            "response_preview": response[:200] if log_response else None,
        }
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"审计日志写入失败: {e}")

    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.chat("hello", max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
