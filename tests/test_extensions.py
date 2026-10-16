"""开发指导书第七部扩展功能测试。"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def configs():
    from src.utils.config_loader import ConfigLoader

    return ConfigLoader(str(ROOT / "configs"))


class TestRandomForest:
    def test_fit_predict_save_load(self, configs, tmp_path):
        from src.models.random_forest_model import RandomForestModel

        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
        y = pd.Series((X["a"] + X["b"] > 0).astype(int))
        cfg = copy.deepcopy(configs.get("model").random_forest)
        cfg.params.n_estimators = 20

        model = RandomForestModel(cfg).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (120,)
        assert ((proba >= 0) & (proba <= 1)).all()

        path = tmp_path / "rf.joblib"
        model.save(str(path))
        loaded = RandomForestModel(cfg).load(str(path))
        assert np.allclose(proba, loaded.predict_proba(X))


class TestLendingClubAdapter:
    def test_adapter_and_loader(self, configs, tmp_path):
        from src.data.loader import DataLoader

        source = pd.DataFrame({
            "loan_amnt": [10000],
            "int_rate": ["12.5%"],
            "annual_inc": [80000],
            "term": ["36 months"],
            "revol_util": ["48.2%"],
        })
        path = tmp_path / "lending_club.csv"
        source.to_csv(path, index=False)
        result = DataLoader(configs.get("data")).load(
            str(path), source="lending_club"
        )
        assert result.loc[0, "loanAmnt"] == 10000
        assert result.loc[0, "interestRate"] == 12.5
        assert result.loc[0, "annualIncome"] == 80000
        assert result.loc[0, "term"] == 3
        assert result.loc[0, "revolUtil"] == 48.2

    def test_openintro_schema_and_terminal_labels(self, configs, tmp_path):
        from src.data.loader import DataLoader

        source = pd.DataFrame({
            "loan_amount": [12000, 8000, 5000],
            "interest_rate": [11.5, 8.0, 17.0],
            "annual_income": [70000, 90000, 45000],
            "term": [36, 60, 36],
            "earliest_credit_line": [2005, 2010, 2012],
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
        })
        path = tmp_path / "openintro_lending_club.csv"
        source.to_csv(path, index=False)
        result = DataLoader(configs.get("data")).load(
            str(path), source="lending_club"
        )
        assert result.loc[0, "loanAmnt"] == 12000
        assert result.loc[0, "earliesCreditLine"] == "Jan-2005"
        assert result.loc[0, "isDefault"] == 0
        assert result.loc[1, "isDefault"] == 1
        assert pd.isna(result.loc[2, "isDefault"])
        assert set(["ficoRangeLow", "n14"]).issubset(result.columns)


class TestDriftDetector:
    def test_detect_shift(self):
        from src.monitoring import DriftDetector

        rng = np.random.default_rng(42)
        baseline = pd.DataFrame({"stable": rng.normal(0, 1, 500), "shifted": rng.normal(0, 1, 500)})
        new_data = pd.DataFrame({"stable": baseline["stable"].copy(), "shifted": rng.normal(3, 1, 500)})
        result = DriftDetector(baseline).detect(new_data, threshold=0.1)
        assert "shifted" in result
        assert "stable" not in result

        details = DriftDetector(baseline).evaluate(new_data, threshold=0.1)
        assert set(details) == {"stable", "shifted"}
        assert details["shifted"]["is_drift"] is True


class TestABRouter:
    def test_consistent_hash_routes_real_models(self, configs):
        from src.ab_test import ABTestRouter
        from src.models.random_forest_model import RandomForestModel

        X = pd.DataFrame({"x": [-2, -1, 1, 2], "y": [0, 1, 0, 1]})
        labels = pd.Series([0, 0, 1, 1])
        cfg = copy.deepcopy(configs.get("model").random_forest)
        cfg.params.n_estimators = 10
        model_a = RandomForestModel(cfg, name="a").fit(X, labels)
        model_b = RandomForestModel(cfg, name="b").fit(X, labels)
        router = ABTestRouter(model_a, model_b, ratio=0.5)

        branch1, pred1 = router.route("user-001", X.head(1))
        branch2, pred2 = router.route("user-001", X.head(1))
        assert branch1 == branch2
        assert np.allclose(pred1, pred2)

    def test_prediction_log_and_labeled_analysis(self, tmp_path):
        from src.ab_test import ABTestLogger

        logger = ABTestLogger(tmp_path / "ab.jsonl")
        for index, (score, outcome) in enumerate(
            [(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)]
        ):
            logger.log(index, "A", "XGBoost", score, outcome)
        logger.log("unlabeled", "B", "Random Forest", 0.4, None)

        summary, metrics = logger.analyze()
        assert summary["样本量"].sum() == 5
        assert set(["AUC", "KS", "F1", "Accuracy"]).issubset(metrics.columns)
        assert metrics.loc[metrics["分组"] == "A", "AUC"].iloc[0] == 1.0


class TestKNNMissingValue:
    @pytest.mark.parametrize(
        "values",
        [
            [10.0, np.nan, 30.0],  # 正常缺失
            [np.nan, np.nan, np.nan],  # 全空
            [5.0, np.nan, 5.0],  # 全同
        ],
        ids=["normal", "all_empty", "all_same"],
    )
    def test_knn_boundaries(self, configs, values):
        from src.data.preprocessor import Preprocessor

        feature_cfg = copy.deepcopy(configs.get("feature"))
        feature_cfg.scaling.method = "none"
        df = pd.DataFrame({
            "dti": values,
            "annualIncome": [50000.0, 60000.0, 70000.0],
            "loanAmnt": [10000.0, 12000.0, 14000.0],
        })
        result = Preprocessor(configs.get("data"), feature_cfg).fit_transform(df)
        assert "dti" in result.columns
        assert result["dti"].isna().sum() == 0


class TestComplianceAgent:
    def test_prompt_and_json_extraction(self, configs):
        from src.llm.agents.compliance_agent import ComplianceAgent
        from src.llm.prompts import PromptManager

        pm = PromptManager(configs.get("llm"))
        prompt = pm.render(
            "compliance_check",
            features={"annualIncome": 80000},
            decision="建议通过",
            interest_rate=12.5,
            rules=ComplianceAgent.DEFAULT_RULES,
        )
        assert "annualIncome" in prompt
        assert "12.5" in prompt
        text = '```json\n{"is_compliant": true, "violations": [], "warnings": [], "recommendations": []}\n```'
        parsed = ComplianceAgent._extract_json(text)
        assert '"is_compliant": true' in parsed
