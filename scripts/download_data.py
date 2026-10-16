#!/usr/bin/env python
"""
数据下载帮助脚本
================
天池数据集需要登录后从网页手动下载。
此脚本提供下载指引和数据完整性校验。

使用:
    python scripts/download_data.py
"""
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("📥 天池数据集下载指引")
    print("=" * 60)
    print()
    print("天池平台不提供数据集 API 直接下载，需要手动操作：")
    print()
    print("1️⃣  访问比赛页面：")
    print("    https://tianchi.aliyun.com/competition/entrance/531830")
    print()
    print("2️⃣  使用阿里云账号登录（无账号需先注册）")
    print()
    print("3️⃣  在 [赛题与数据] 标签页下载以下文件：")
    print("    - train.csv     (~155MB)")
    print("    - testA.csv     (~39MB)")
    print()
    print("4️⃣  将下载的文件放入项目目录：")
    print("    credit_risk_system/")
    print("    └── data/")
    print("        └── raw/")
    print("            ├── train.csv")
    print("            └── testA.csv")
    print()
    print("=" * 60)
    print("💡 替代方案：使用合成示例数据")
    print("=" * 60)
    print()
    print("如果暂时无法下载真实数据，可以先用示例数据：")
    print()
    print("    python scripts/generate_sample_data.py --n_samples 50000")
    print()
    print("示例数据字段与天池数据完全一致，可以正常跑通流程。")
    print()

    # 检查现有数据状态
    data_dir = Path("./data/raw")
    if data_dir.exists():
        files = list(data_dir.glob("*.csv"))
        if files:
            print("✅ 检测到已有数据文件：")
            for f in files:
                size_mb = f.stat().st_size / (1024**2)
                print(f"   - {f.name} ({size_mb:.2f} MB)")
        else:
            print("⚠️  data/raw/ 目录存在但无 CSV 文件")
    else:
        print("ℹ️  data/raw/ 目录尚未创建")

    print()


if __name__ == "__main__":
    main()
