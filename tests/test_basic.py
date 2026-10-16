"""
基础冒烟测试
============
确保各模块基本可用。

运行: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def configs():
    """全局配置"""
    from src.utils.config_loader import ConfigLoader
    return ConfigLoader(str(ROOT / "configs"))


@pytest.fixture(scope="module")
def sample_df():
    """合成测试数据"""
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "id": np.arange(n),
        "loanAmnt": rng.lognormal(9.5, 0.5, n),
        "term": rng.choice([3, 5], n),
        "interestRate": rng.uniform(5, 25, n),
        "installment": rng.uniform(100, 2000, n),
        "grade": rng.choice(["A", "B", "C", "D"], n),
        "subGrade": rng.choice(["A1", "B2"], n),
        "employmentTitle": rng.uniform(0, 100, n),
        "employmentLength": rng.choice(["< 1 year", "1 year", "10+ years"], n),
        "homeOwnership": rng.choice([0, 1, 2], n),
        "annualIncome": rng.lognormal(11, 0.5, n),
        "verificationStatus": rng.choice([0, 1, 2], n),
        "issueDate": ["2019-01-01"] * n,
        "purpose": rng.integers(0, 14, n),
        "postCode": rng.uniform(0, 1000, n),
        "regionCode": rng.integers(0, 50, n),
        "dti": rng.uniform(0, 50, n),
        "delinquency_2years": rng.poisson(0.3, n).astype(float),
        "ficoRangeLow": rng.integers(580, 850, n),
        "ficoRangeHigh": rng.integers(580, 850, n),
        "openAcc": rng.poisson(11, n).astype(float),
        "pubRec": rng.poisson(0.2, n).astype(float),
        "pubRecBankruptcies": rng.poisson(0.1, n).astype(float),
        "revolBal": rng.lognormal(8, 1, n),
        "revolUtil": rng.uniform(0, 100, n),
        "totalAcc": rng.poisson(25, n).astype(float),
        "initialListStatus": rng.choice([0, 1], n),
        "applicationType": rng.choice([0, 1], n),
        "earliesCreditLine": ["Jan-2010"] * n,
        "title": rng.uniform(0, 100, n),
        "policyCode": np.ones(n),
        "isDefault": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })
    # 加几个匿名字段
    for i in range(15):
        df[f"n{i}"] = rng.normal(0, 1, n)
    return df


# ===== 测试 ConfigLoader =====
class TestConfig:
    def test_load_all_configs(self, configs):
        assert configs.get("data") is not None
        assert configs.get("model") is not None
        assert configs.get("feature") is not None
        assert configs.get("llm") is not None

    def test_attr_access(self, configs):
        # 应支持点访问
        lr = configs.get("model").xgboost.params.learning_rate
        assert isinstance(lr, (int, float))


# ===== 测试评估指标 =====
class TestMetrics:
    def test_evaluator(self):
        from src.utils.metrics import Evaluator
        rng = np.random.default_rng(0)
        y_true = rng.choice([0, 1], 1000)
        y_pred = rng.uniform(0, 1, 1000)

        evaluator = Evaluator()
        result = evaluator.evaluate(y_true, y_pred)

        assert 0 <= result.auc <= 1
        assert 0 <= result.ks <= 1
        assert 0 <= result.f1 <= 1


# ===== 测试预处理 =====
class TestPreprocessor:
    def test_fit_transform(self, configs, sample_df):
        from src.data.preprocessor import Preprocessor

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        X_processed = prep.fit_transform(X)

        # 处理后不应有缺失值
        assert X_processed.isnull().sum().sum() == 0


# ===== 测试 XGBoost =====
class TestXGB:
    def test_fit_predict(self, configs, sample_df):
        from src.models.xgb_model import XGBModel
        from src.data.preprocessor import Preprocessor

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        # 减少 estimators 加快测试
        configs.get("model").xgboost.params.n_estimators = 50
        configs.get("model").xgboost.early_stopping.enabled = False

        model = XGBModel(configs.get("model").xgboost)
        model.fit(X_processed, y)

        proba = model.predict_proba(X_processed)
        assert proba.shape == (len(y),)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_save_load(self, configs, sample_df, tmp_path):
        from src.models.xgb_model import XGBModel
        from src.data.preprocessor import Preprocessor

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        configs.get("model").xgboost.params.n_estimators = 50
        configs.get("model").xgboost.early_stopping.enabled = False

        model = XGBModel(configs.get("model").xgboost)
        model.fit(X_processed, y)

        save_path = tmp_path / "test_model.json"
        model.save(str(save_path))
        assert save_path.exists()

        model2 = XGBModel(configs.get("model").xgboost)
        model2.load(str(save_path))
        proba1 = model.predict_proba(X_processed.head(10))
        proba2 = model2.predict_proba(X_processed.head(10))
        assert np.allclose(proba1, proba2, rtol=1e-5)


# ===== 测试 Feature Builder =====
class TestFeatureBuilder:
    def test_build(self, configs, sample_df):
        from src.features.builder import FeatureBuilder

        builder = FeatureBuilder(configs.get("feature"))
        df_new = builder.build(sample_df)

        # 至少应该多出一些衍生特征
        assert df_new.shape[1] >= sample_df.shape[1]


# ===== 测试 Prompt Manager =====
class TestPromptManager:
    def test_render(self, configs):
        from src.llm.prompts import PromptManager

        pm = PromptManager(configs.get("llm"))
        # 简单测试 chat_assistant 模板
        prompt = pm.render("chat_assistant", user_question="什么是 AUC？")
        assert "什么是 AUC？" in prompt
        assert len(prompt) > 50

    def test_system_role(self, configs):
        from src.llm.prompts import PromptManager
        pm = PromptManager(configs.get("llm"))
        role = pm.get_system_role("risk_report")
        assert isinstance(role, str)
        assert len(role) > 0


# ===== 测试 SHAP =====
class TestSHAP:
    def test_explain(self, configs, sample_df):
        from src.models.xgb_model import XGBModel
        from src.data.preprocessor import Preprocessor
        from src.explain.shap_analyzer import SHAPAnalyzer

        prep = Preprocessor(configs.get("data"), configs.get("feature"))
        X = sample_df.drop(columns=["isDefault", "id"])
        y = sample_df["isDefault"]
        X_processed = prep.fit_transform(X)

        configs.get("model").xgboost.params.n_estimators = 30
        configs.get("model").xgboost.early_stopping.enabled = False

        model = XGBModel(configs.get("model").xgboost)
        model.fit(X_processed, y)

        analyzer = SHAPAnalyzer(max_background_size=50)
        analyzer.fit(model.model, X_processed.head(50))

        # 解释单样本
        explanation = analyzer.explain_single(X_processed.iloc[0])
        assert explanation.shap_values is not None
        assert len(explanation.feature_names) == X_processed.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
