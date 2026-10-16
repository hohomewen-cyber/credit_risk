"""Lending Club schema 到当前天池项目 schema 的适配器。"""
from __future__ import annotations

import numpy as np
import pandas as pd


class LendingClubAdapter:
    """把 Lending Club 字段映射到当前项目使用的字段。"""

    FIELD_MAPPING = {
        # 开发指导书示例及 Lending Club 历史导出字段
        "loan_amnt": "loanAmnt",
        "int_rate": "interestRate",
        "annual_inc": "annualIncome",
        "sub_grade": "subGrade",
        "emp_title": "employmentTitle",
        "emp_length": "employmentLength",
        "home_ownership": "homeOwnership",
        "verification_status": "verificationStatus",
        "issue_d": "issueDate",
        "zip_code": "postCode",
        "addr_state": "regionCode",
        "delinq_2yrs": "delinquency_2years",
        "fico_range_low": "ficoRangeLow",
        "fico_range_high": "ficoRangeHigh",
        "open_acc": "openAcc",
        "pub_rec": "pubRec",
        "pub_rec_bankruptcies": "pubRecBankruptcies",
        "revol_bal": "revolBal",
        "revol_util": "revolUtil",
        "total_acc": "totalAcc",
        "initial_list_status": "initialListStatus",
        "application_type": "applicationType",
        "earliest_cr_line": "earliesCreditLine",
        "policy_code": "policyCode",
        "loan_status": "isDefault",
        # OpenIntro 发布的 Lending Club 公开数据字段
        "loan_amount": "loanAmnt",
        "interest_rate": "interestRate",
        "annual_income": "annualIncome",
        "emp_title": "employmentTitle",
        "emp_length": "employmentLength",
        "homeownership": "homeOwnership",
        "verified_income": "verificationStatus",
        "issue_month": "issueDate",
        "state": "regionCode",
        "debt_to_income": "dti",
        "delinq_2y": "delinquency_2years",
        "earliest_credit_line": "earliesCreditLine",
        "total_credit_lines": "totalAcc",
        "open_credit_lines": "openAcc",
        "total_credit_utilized": "revolBal",
        "num_historical_failed_to_pay": "pubRec",
        "public_record_bankrupt": "pubRecBankruptcies",
        "loan_purpose": "purpose",
        "application_type": "applicationType",
        "initial_listing_status": "initialListStatus",
    }

    PROJECT_FEATURES = [
        "loanAmnt", "term", "interestRate", "installment", "grade",
        "subGrade", "employmentTitle", "employmentLength", "homeOwnership",
        "annualIncome", "verificationStatus", "issueDate", "purpose",
        "postCode", "regionCode", "dti", "delinquency_2years",
        "ficoRangeLow", "ficoRangeHigh", "openAcc", "pubRec",
        "pubRecBankruptcies", "revolBal", "revolUtil", "totalAcc",
        "initialListStatus", "applicationType", "earliesCreditLine", "title",
        "policyCode", *[f"n{i}" for i in range(15)],
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        original = df.copy()
        df = df.rename(columns=self.FIELD_MAPPING).copy()

        if "interestRate" in df.columns:
            df["interestRate"] = pd.to_numeric(
                df["interestRate"].astype(str).str.rstrip("%"), errors="coerce"
            )
        if "revolUtil" in df.columns:
            df["revolUtil"] = pd.to_numeric(
                df["revolUtil"].astype(str).str.rstrip("%"), errors="coerce"
            )
        if "term" in df.columns:
            term = pd.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )
            # Lending Club 通常以 36/60 months 表示，项目使用 3/5 年。
            df["term"] = term.where(term <= 10, term / 12)

        if "employmentLength" in df.columns:
            df["employmentLength"] = df["employmentLength"].map(
                self._normalize_employment_length
            )

        if "homeOwnership" in df.columns:
            ownership_mapping = {
                "rent": 0, "mortgage": 1, "own": 2, "other": 3,
            }
            normalized = df["homeOwnership"].astype(str).str.lower()
            df["homeOwnership"] = normalized.map(ownership_mapping).where(
                normalized.isin(ownership_mapping), df["homeOwnership"]
            )

        if "verificationStatus" in df.columns:
            verification_mapping = {
                "not verified": 0, "source verified": 1, "verified": 2,
            }
            normalized = df["verificationStatus"].astype(str).str.lower()
            df["verificationStatus"] = normalized.map(verification_mapping).where(
                normalized.isin(verification_mapping), df["verificationStatus"]
            )

        if "applicationType" in df.columns:
            application_mapping = {"individual": 0, "joint": 1}
            normalized = df["applicationType"].astype(str).str.lower()
            df["applicationType"] = normalized.map(application_mapping).where(
                normalized.isin(application_mapping), df["applicationType"]
            )

        if "initialListStatus" in df.columns:
            listing_mapping = {"whole": 0, "fractional": 1, "w": 0, "f": 1}
            normalized = df["initialListStatus"].astype(str).str.lower()
            df["initialListStatus"] = normalized.map(listing_mapping).where(
                normalized.isin(listing_mapping), df["initialListStatus"]
            )

        if "issueDate" in df.columns:
            issue_date = pd.to_datetime(df["issueDate"], errors="coerce")
            df["issueDate"] = issue_date.dt.strftime("%Y-%m-%d")

        if "earliesCreditLine" in df.columns:
            raw_credit_line = df["earliesCreditLine"]
            year = pd.to_numeric(raw_credit_line, errors="coerce")
            converted = raw_credit_line.astype(object).copy()
            numeric_mask = year.notna()
            converted.loc[numeric_mask] = (
                "Jan-" + year.loc[numeric_mask].round().astype("Int64").astype(str)
            )
            df["earliesCreditLine"] = converted

        # 公开 OpenIntro 数据没有直接的 revol_util，按其真实余额/额度计算。
        if "revolUtil" not in df.columns and {
            "total_credit_utilized", "total_credit_limit"
        }.issubset(original.columns):
            utilized = pd.to_numeric(original["total_credit_utilized"], errors="coerce")
            limit = pd.to_numeric(original["total_credit_limit"], errors="coerce")
            df["revolUtil"] = (utilized / limit.replace(0, np.nan)) * 100

        if "isDefault" in df.columns:
            terminal_status = {
                "fully paid": 0,
                "charged off": 1,
                "default": 1,
            }
            status = df["isDefault"].astype(str).str.strip().str.lower()
            df["isDefault"] = status.map(terminal_status).astype("Float64")

        if "id" not in df.columns:
            df.insert(0, "id", np.arange(len(df), dtype=int))

        # 外部数据没有的项目字段保持为缺失值，交给原预处理器的既有策略处理。
        for column in self.PROJECT_FEATURES:
            if column not in df.columns:
                df[column] = np.nan

        ordered = ["id", *self.PROJECT_FEATURES]
        if "isDefault" in df.columns:
            ordered.append("isDefault")
        remaining = [column for column in df.columns if column not in ordered]
        df = df[ordered + remaining]
        return df

    @staticmethod
    def _normalize_employment_length(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, str) and "year" in value.lower():
            return value
        try:
            years = int(float(value))
        except (TypeError, ValueError):
            return value
        if years <= 0:
            return "< 1 year"
        if years >= 10:
            return "10+ years"
        return f"{years} year" if years == 1 else f"{years} years"
