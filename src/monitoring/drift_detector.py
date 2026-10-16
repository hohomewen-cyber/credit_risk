"""KS 特征漂移检测。"""
from scipy.stats import ks_2samp


class DriftDetector:
    def __init__(self, baseline_data):
        self.baseline = baseline_data

    def detect(self, new_data, threshold=0.1):
        """对每个特征做 KS 检验。"""
        return {
            name: {"ks_stat": result["ks_stat"], "p_value": result["p_value"]}
            for name, result in self.evaluate(new_data, threshold).items()
            if result["is_drift"]
        }

    def evaluate(self, new_data, threshold=0.1):
        """返回全部可检验特征的 KS 明细，detect 的核心判定保持不变。"""
        results = {}
        for col in self.baseline.columns:
            if col not in new_data.columns:
                continue
            baseline_values = self.baseline[col].dropna()
            new_values = new_data[col].dropna()
            if baseline_values.empty or new_values.empty:
                continue
            stat, pvalue = ks_2samp(baseline_values, new_values)
            results[col] = {
                "ks_stat": float(stat),
                "p_value": float(pvalue),
                "is_drift": bool(stat > threshold),
            }
        return results
