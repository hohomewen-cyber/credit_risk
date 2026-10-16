#!/usr/bin/env python
"""
训练流水线脚本
==============
一键执行完整的训练流程。

使用:
    # 使用示例数据快速训练（用于调试）
    python scripts/train_pipeline.py --use-sample

    # 使用真实数据 + 启用超参优化（生产模式）
    python scripts/train_pipeline.py --tune

    # 仅训练 XGBoost（跳过 MLP）
    python scripts/train_pipeline.py --use-sample --skip-mlp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根目录（脚本所在目录的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 添加项目根目录到 PYTHONPATH
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.pipeline.orchestrator import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="智能信贷风控分析系统 - 训练流水线")
    parser.add_argument(
        "--config-dir",
        default=str(PROJECT_ROOT / "configs"),
        help="配置目录（默认指向项目根下的 configs/）",
    )
    parser.add_argument("--use-sample", action="store_true", help="使用示例数据（快速）")
    parser.add_argument("--skip-mlp", action="store_true", help="跳过 MLP 训练")
    parser.add_argument("--tune", action="store_true", help="启用超参优化（耗时）")
    args = parser.parse_args()

    logger = get_logger("scripts.train")

    # 1. 加载配置
    logger.info(f"加载配置: {args.config_dir}")
    configs = ConfigLoader(args.config_dir)

    # 2. 创建并运行 Pipeline
    pipeline = TrainingPipeline(configs)
    result = pipeline.run(
        use_sample=args.use_sample,
        skip_mlp=args.skip_mlp,
        skip_tuning=not args.tune,
    )

    # 3. 打印结果
    print("\n" + "=" * 60)
    print("📊 训练完成 - 评估结果")
    print("=" * 60)
    print(f"运行 ID:    {result['run_id']}")
    print(f"耗时:       {result['duration']:.2f}s")
    print(f"训练样本:   {result['train_samples']:,}")
    print(f"特征数:     {result['feature_count']}")
    print()

    for model_name, metrics in result["eval_results"].items():
        print(f"[{model_name.upper()}]")
        print(f"  AUC:        {metrics['auc']:.4f}")
        print(f"  KS:         {metrics['ks']:.4f}")
        print(f"  F1:         {metrics['f1']:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print()

    print("✅ 模型已保存至 data/models/latest/")
    print("👉 接下来运行: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
