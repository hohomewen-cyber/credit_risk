#!/usr/bin/env python
"""
示例数据生成器
==============
生成与天池"贷款违约预测"数据集字段一致的合成数据，
让初学者无需下载真实数据即可跑通流程。

使用:
    python scripts/generate_sample_data.py --n_samples 50000
    python scripts/generate_sample_data.py --n_samples 5000 --output data/sample/sample.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 让脚本能 import src
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_sample_data(n_samples: int = 50000, random_seed: int = 42) -> pd.DataFrame:
    """生成与天池数据集字段一致的合成数据"""
    rng = np.random.default_rng(random_seed)

    # ===== 基础字段 =====
    df = pd.DataFrame()
    df["id"] = np.arange(n_samples)

    # 贷款金额：1000 - 40000，类似真实分布
    df["loanAmnt"] = np.round(rng.lognormal(mean=9.5, sigma=0.6, size=n_samples)).clip(500, 40000)

    # 期限：3 或 5 年
    df["term"] = rng.choice([3, 5], size=n_samples, p=[0.7, 0.3])

    # 利率：5% - 30%
    df["interestRate"] = np.round(rng.normal(13, 4, n_samples).clip(5, 30), 2)

    # 月还款（基于贷款金额、期限、利率近似计算）
    monthly_rate = df["interestRate"] / 100 / 12
    n_months = df["term"] * 12
    df["installment"] = np.round(
        df["loanAmnt"] * monthly_rate * (1 + monthly_rate) ** n_months
        / ((1 + monthly_rate) ** n_months - 1),
        2,
    )

    # 贷款等级 A-G
    df["grade"] = rng.choice(["A", "B", "C", "D", "E", "F", "G"],
                              size=n_samples, p=[0.15, 0.25, 0.25, 0.18, 0.10, 0.05, 0.02])
    # 子级 A1-G5
    df["subGrade"] = df["grade"] + rng.integers(1, 6, n_samples).astype(str)

    # 就业职称（用整数编码模拟脱敏）
    df["employmentTitle"] = rng.integers(0, 250000, n_samples).astype(float)

    # 就业年限
    df["employmentLength"] = rng.choice(
        ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
         "6 years", "7 years", "8 years", "9 years", "10+ years"],
        size=n_samples,
        p=[0.10, 0.07, 0.10, 0.09, 0.07, 0.07, 0.05, 0.05, 0.05, 0.05, 0.30],
    )

    # 房屋所有权：0=租房, 1=按揭, 2=自有, 3=其他
    df["homeOwnership"] = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.40, 0.50, 0.08, 0.02])

    # 年收入
    df["annualIncome"] = np.round(
        rng.lognormal(mean=11, sigma=0.6, size=n_samples), 2
    ).clip(8000, 1500000)

    # 验证状态：0=未验证, 1=源已验证, 2=已验证
    df["verificationStatus"] = rng.choice([0, 1, 2], size=n_samples, p=[0.30, 0.40, 0.30])

    # 贷款发放日期
    df["issueDate"] = pd.to_datetime(
        rng.choice(pd.date_range("2014-01-01", "2019-12-31"), size=n_samples)
    ).strftime("%Y-%m-%d")

    # 贷款用途（0-13）
    df["purpose"] = rng.integers(0, 14, n_samples)

    # 邮编（脱敏，前 3 位）
    df["postCode"] = rng.integers(0, 1000, n_samples).astype(float)

    # 地区代码
    df["regionCode"] = rng.integers(0, 50, n_samples)

    # 负债收入比
    df["dti"] = np.round(rng.normal(18, 8, n_samples).clip(0, 60), 2)

    # 近 2 年逾期次数
    df["delinquency_2years"] = rng.poisson(0.3, n_samples).astype(float)

    # FICO 评分
    df["ficoRangeLow"] = rng.normal(690, 30, n_samples).clip(580, 850).round().astype(int)
    df["ficoRangeHigh"] = df["ficoRangeLow"] + 4

    # 未结信用额度数
    df["openAcc"] = rng.poisson(11, n_samples).astype(float)

    # 公开记录数
    df["pubRec"] = rng.poisson(0.2, n_samples).astype(float)

    # 破产记录
    df["pubRecBankruptcies"] = rng.poisson(0.1, n_samples).astype(float)

    # 循环余额
    df["revolBal"] = np.round(rng.lognormal(8.5, 1.2, n_samples), 2)

    # 循环额度利用率（0-100%）
    df["revolUtil"] = np.round(rng.beta(2, 3, n_samples) * 100, 2)

    # 总信用额度数
    df["totalAcc"] = rng.poisson(25, n_samples).astype(float) + 1

    # 初始上市状态
    df["initialListStatus"] = rng.choice([0, 1], n_samples, p=[0.6, 0.4])

    # 申请类型：0=个人, 1=共同申请
    df["applicationType"] = rng.choice([0, 1], n_samples, p=[0.95, 0.05])

    # 最早信用额度（早于 issueDate 数年）
    df["earliesCreditLine"] = (
        pd.to_datetime(df["issueDate"]) - pd.to_timedelta(rng.integers(365 * 5, 365 * 30, n_samples), unit="D")
    ).dt.strftime("%b-%Y")

    # 贷款名称（脱敏整数）
    df["title"] = rng.integers(0, 100, n_samples).astype(float)

    # 策略代码
    df["policyCode"] = 1.0

    # ===== 匿名特征 n0-n14 =====
    for i in range(15):
        df[f"n{i}"] = rng.normal(0, 1, n_samples).round(4)

    # ===== 生成目标变量 isDefault =====
    # 根据多个特征构造一个"真实"违约概率，让数据具有可学习的模式
    z = (
        -2.0  # 基线 logit
        + 0.08 * (df["interestRate"] - 13)             # 利率高 -> 风险高
        + 0.000015 * df["loanAmnt"]                    # 贷款金额大 -> 风险略高
        + 0.04 * df["dti"] / 18                        # DTI 高 -> 风险高
        - 0.008 * (df["ficoRangeLow"] - 690)           # FICO 高 -> 风险低
        - 0.02 * (df["annualIncome"] / 50000)          # 年收入高 -> 风险低
        + 0.15 * df["delinquency_2years"]              # 逾期次数 -> 风险高
        + 0.5 * (df["grade"].isin(["E", "F", "G"]))    # 低等级 -> 风险高
        + 0.3 * (df["homeOwnership"] == 0)             # 租房 -> 风险略高
        - 0.25 * (df["employmentLength"] == "10+ years")  # 长就业年限 -> 风险低
        + rng.normal(0, 0.5, n_samples)                # 加入噪声
    )

    # logistic 变换
    proba = 1 / (1 + np.exp(-z))

    # 采样
    df["isDefault"] = (rng.uniform(0, 1, n_samples) < proba).astype(int)

    # 引入一些缺失值（模拟真实数据）
    for col in ["employmentTitle", "postCode", "title", "dti", "revolUtil",
                "n0", "n5", "n9", "pubRecBankruptcies"]:
        mask = rng.uniform(0, 1, n_samples) < 0.02  # 2% 缺失
        df.loc[mask, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="生成示例风控数据集")
    parser.add_argument("--n_samples", type=int, default=50000, help="样本数")
    parser.add_argument("--output", type=str, default="./data/sample/sample.csv", help="输出路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"⏳ 正在生成 {args.n_samples} 条示例数据...")
    df = generate_sample_data(args.n_samples, args.seed)

    print(f"💾 保存至: {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8")

    # 同时生成简单的数据概览
    print("\n📊 数据概览:")
    print(f"  样本数:        {len(df):,}")
    print(f"  特征数:        {df.shape[1] - 2}")  # 排除 id 和 isDefault
    print(f"  违约率:        {df['isDefault'].mean():.2%}")
    print(f"  缺失值总数:    {df.isnull().sum().sum():,}")
    print(f"  文件大小:      {output_path.stat().st_size / (1024**2):.2f} MB")

    print("\n✅ 完成！现在可以：")
    print(f"  1. 训练模型: python scripts/train_pipeline.py --use-sample")
    print(f"  2. 启动应用: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
