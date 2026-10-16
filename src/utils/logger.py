"""
日志工具
========
基于 loguru 的统一日志接口，支持文件 + 控制台双输出，按日期切分。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger


def get_logger(
    name: str = "credit_risk",
    log_dir: str = "./logs",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "14 days",
):
    """
    获取配置好的 logger 实例。

    参数:
        name: logger 名称（用于区分模块）
        log_dir: 日志保存目录
        level: 日志级别 DEBUG/INFO/WARNING/ERROR
        rotation: 日志切分规则
        retention: 日志保留时间
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 移除默认 handler，避免重复输出
    _loguru_logger.remove()

    # 控制台输出（彩色）
    _loguru_logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    # 文件输出（按日期切分）
    _loguru_logger.add(
        log_path / f"{name}_{{time:YYYY-MM-DD}}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,  # 多进程安全
    )

    return _loguru_logger.bind(name=name)


# 默认 logger 实例（导入即可用）
default_logger = get_logger()
