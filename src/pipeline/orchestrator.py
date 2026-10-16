"""
训练流程编排器
==============
将数据加载、预处理、特征工程、模型训练、评估、解释串联起来。

要点:
    将完整流程封装为可重复执行的 Pipeline，
    避免 notebook 式的散乱代码，便于：
    - 一键复现完整实验
    - 自动化训练（CI/CD）
    - 参数对比（不同配置同样的流程）
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.loader import DataLoader, DataSplitter
from src.data.preprocessor import Preprocessor
from src.features.builder import FeatureBuilder
from src.features.selector import FeatureSelector
from src.models.xgb_model import XGBModel
from src.models.mlp_model import MLPModel
from src.models.random_forest_model import RandomForestModel
from src.models.ensemble import EnsembleModel
from src.tuning.optimizer import OptunaOptimizer
from src.utils.metrics import Evaluator
from src.utils.logger import get_logger

logger = get_logger("pipeline")


class TrainingPipeline:
    """
    训练流程编排器。

    使用示例:
        >>> from src.utils.config_loader import ConfigLoader
        >>> configs = ConfigLoader("./configs")
        >>> pipeline = TrainingPipeline(configs)
        >>> result = pipeline.run(use_sample=True)
    """

    def __init__(self, configs):
        self.configs = configs
        self.data_cfg = configs.get("data")
        self.feature_cfg = configs.get("feature")
        self.model_cfg = configs.get("model")
        self.pipeline_cfg = configs.get("pipeline")

        # 各步骤产出
        self.df: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[Preprocessor] = None
        self.feature_builder: Optional[FeatureBuilder] = None
        self.feature_selector: Optional[FeatureSelector] = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.xgb_model: Optional[XGBModel] = None
        self.mlp_model: Optional[MLPModel] = None
        self.random_forest_model: Optional[RandomForestModel] = None
        self.ensemble: Optional[EnsembleModel] = None

        self.eval_results: Dict[str, Any] = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(
        self,
        use_sample: bool = False,
        input_df: Optional[pd.DataFrame] = None,
        data_source_name: Optional[str] = None,
        skip_mlp: bool = False,
        skip_random_forest: bool = False,
        skip_tuning: bool = True,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        执行完整训练流程。

        参数:
            use_sample: 使用示例数据
            input_df: Web 端已选中并由 DataLoader 加载的数据
            data_source_name: 当前数据源名称（用于日志和产物元信息）
            skip_mlp: 跳过 MLP 训练（快速调试）
            skip_random_forest: 跳过 Random Forest 训练
            skip_tuning: 跳过超参优化（默认跳过，节省时间）
            progress_callback: 进度回调函数 (step_name, progress_pct)

        返回:
            训练结果字典
        """
        t_start = time.time()
        logger.info(f"=" * 60)
        logger.info(f"开始训练流程 - run_id: {self.run_id}")
        logger.info(f"=" * 60)

        steps = [
            ("加载数据", self._step_load_data, {
                "use_sample": use_sample,
                "input_df": input_df,
                "data_source_name": data_source_name,
            }),
            ("数据预处理", self._step_preprocess, {}),
            ("特征工程", self._step_feature_engineering, {}),
            ("特征筛选", self._step_feature_selection, {}),
            ("数据划分", self._step_split_data, {}),
            ("训练 XGBoost", self._step_train_xgb, {"with_tuning": not skip_tuning}),
        ]

        if not skip_random_forest:
            steps.append(("训练 Random Forest", self._step_train_random_forest, {}))

        if not skip_mlp:
            steps.append(("训练 MLP", self._step_train_mlp, {}))
            steps.append(("模型融合", self._step_ensemble, {}))

        steps.append(("评估", self._step_evaluate, {}))
        steps.append(("保存产物", self._step_save_artifacts, {}))

        total_steps = len(steps)
        for i, (step_name, step_func, step_kwargs) in enumerate(steps):
            logger.info(f"\n[{i+1}/{total_steps}] {step_name}...")
            if progress_callback:
                progress_callback(step_name, int((i / total_steps) * 100))
            try:
                step_func(**step_kwargs)
            except Exception as e:
                logger.error(f"步骤 [{step_name}] 失败: {e}", exc_info=True)
                raise

        if progress_callback:
            progress_callback("完成", 100)

        duration = time.time() - t_start
        logger.info(f"\n训练流程完成 - 总耗时 {duration:.2f}s")

        return {
            "run_id": self.run_id,
            "duration": duration,
            "eval_results": self.eval_results,
            "feature_count": self.X_train.shape[1] if self.X_train is not None else 0,
            "train_samples": len(self.X_train) if self.X_train is not None else 0,
            "data_source": getattr(self, "data_source_name", None),
        }

    # =====================================================================
    # 各步骤实现
    # =====================================================================

    def _step_load_data(
        self,
        use_sample: bool = False,
        input_df: Optional[pd.DataFrame] = None,
        data_source_name: Optional[str] = None,
    ) -> None:
        loader = DataLoader(self.data_cfg)
        if input_df is None:
            self.df = loader.load_train(use_sample=use_sample)
            self.data_source_name = "合成示例数据" if use_sample else "天池原始数据"
        else:
            self.df = input_df.copy()
            self.data_source_name = data_source_name or "Web 当前数据源"

        target = self.data_cfg.target_column
        if target not in self.df.columns:
            raise ValueError(f"当前数据源缺少标签列: {target}")

        # Lending Club 包含 Current/宽限期等尚无最终结果的记录；
        # 训练只能使用真实终态标签，不得将未知结果伪装成 0/1。
        unlabeled_count = int(self.df[target].isna().sum())
        if unlabeled_count:
            self.df = self.df.loc[self.df[target].notna()].copy()
            logger.info(f"过滤无终态标签记录: {unlabeled_count} 条")

        if self.df.empty or self.df[target].nunique(dropna=True) < 2:
            raise ValueError("当前数据源没有足够的真实二分类标签，无法训练")

        self.df[target] = self.df[target].astype(int)
        summary = loader.get_data_summary(self.df)
        logger.info(f"数据加载 [{self.data_source_name}]: {summary}")

    def _step_preprocess(self) -> None:
        # 注意：先分离标签，再 fit
        target = self.data_cfg.target_column
        self.y_full = self.df[target].copy()

        # 临时整体 fit （也可以先 split 后 fit，避免数据泄漏；
        # 但 LabelEncoder 需要看全量类别，这里简化处理）
        self.preprocessor = Preprocessor(self.data_cfg, self.feature_cfg)
        # 排除 ID 列和标签列
        df_features = self.df.drop(
            columns=[self.data_cfg.id_column, target], errors="ignore"
        )
        # Adapter 可保留外部数据源的原始字段供探索页展示，
        # 训练入口只把项目配置中的标准字段交给原预处理器。
        configured_features = [
            *self.data_cfg.columns.numerical,
            *self.data_cfg.columns.categorical,
            *self.data_cfg.columns.datetime,
            *self.data_cfg.columns.anonymous,
        ]
        df_features = df_features.reindex(columns=configured_features)
        df_processed = self.preprocessor.fit_transform(df_features)
        df_processed[target] = self.y_full.values

        self.df = df_processed
        logger.info(f"预处理后 shape: {self.df.shape}")

    def _step_feature_engineering(self) -> None:
        self.feature_builder = FeatureBuilder(self.feature_cfg)
        target = self.data_cfg.target_column
        y_tmp = self.df[target]

        df_features = self.df.drop(columns=[target])
        df_with_new = self.feature_builder.build(df_features)
        df_with_new[target] = y_tmp.values

        self.df = df_with_new
        logger.info(f"特征工程后 shape: {self.df.shape}")

    def _step_feature_selection(self) -> None:
        if not self.feature_cfg.feature_selection.enabled:
            logger.info("跳过特征筛选")
            return

        target = self.data_cfg.target_column
        y_tmp = self.df[target]
        X = self.df.drop(columns=[target])

        self.feature_selector = FeatureSelector(self.feature_cfg)
        X_selected = self.feature_selector.fit_transform(X, y_tmp)
        X_selected[target] = y_tmp.values

        self.df = X_selected
        logger.info(f"特征筛选后 shape: {self.df.shape}")

    def _step_split_data(self) -> None:
        splitter = DataSplitter(self.data_cfg)
        result = splitter.split(self.df, with_validation=True)
        (
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
        ) = result

    def _step_train_xgb(self, with_tuning: bool = False) -> None:
        if not self.model_cfg.xgboost.enabled:
            logger.info("跳过 XGBoost 训练")
            return

        # 超参优化（可选）
        if with_tuning and self.model_cfg.xgboost.tuning.enabled:
            logger.info("开始 XGBoost 超参优化...")
            optimizer = OptunaOptimizer(self.model_cfg.xgboost.tuning)
            from xgboost import XGBClassifier

            base_params = dict(self.model_cfg.xgboost.params)
            best_params = optimizer.optimize(
                model_class=XGBClassifier,
                X_train=self.X_train, y_train=self.y_train,
                X_val=self.X_val, y_val=self.y_val,
                base_params=base_params,
            )
            # 将最佳参数写回配置（运行时）
            for k, v in best_params.items():
                self.model_cfg.xgboost.params[k] = v

        self.xgb_model = XGBModel(self.model_cfg.xgboost)
        self.xgb_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

    def _step_train_mlp(self) -> None:
        if not self.model_cfg.mlp.enabled:
            logger.info("跳过 MLP 训练")
            return

        self.mlp_model = MLPModel(self.model_cfg.mlp)
        self.mlp_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

    def _step_train_random_forest(self) -> None:
        if not self.model_cfg.random_forest.enabled:
            logger.info("跳过 Random Forest 训练")
            return

        self.random_forest_model = RandomForestModel(self.model_cfg.random_forest)
        self.random_forest_model.fit(
            self.X_train, self.y_train, self.X_val, self.y_val
        )

    def _step_ensemble(self) -> None:
        if not self.model_cfg.ensemble.enabled:
            return
        if self.xgb_model is None or self.mlp_model is None:
            logger.warning("基模型不全，跳过融合")
            return

        self.ensemble = EnsembleModel(self.model_cfg.ensemble)
        self.ensemble.add_model("xgboost", self.xgb_model)
        self.ensemble.add_model("mlp", self.mlp_model)
        self.ensemble.fit(self.X_train, self.y_train)

    def _step_evaluate(self) -> None:
        evaluator = Evaluator()

        # 评估各个模型
        if self.xgb_model is not None:
            y_pred = self.xgb_model.predict_proba(self.X_test)
            self.eval_results["xgboost"] = evaluator.evaluate(self.y_test, y_pred).to_dict()
            logger.info(f"XGBoost: {self.eval_results['xgboost']}")

        if self.mlp_model is not None:
            y_pred = self.mlp_model.predict_proba(self.X_test)
            self.eval_results["mlp"] = evaluator.evaluate(self.y_test, y_pred).to_dict()
            logger.info(f"MLP: {self.eval_results['mlp']}")

        if self.random_forest_model is not None:
            y_pred = self.random_forest_model.predict_proba(self.X_test)
            self.eval_results["random_forest"] = evaluator.evaluate(
                self.y_test, y_pred
            ).to_dict()
            logger.info(f"Random Forest: {self.eval_results['random_forest']}")

        if self.ensemble is not None:
            y_pred = self.ensemble.predict_proba(self.X_test)
            self.eval_results["ensemble"] = evaluator.evaluate(self.y_test, y_pred).to_dict()
            logger.info(f"Ensemble: {self.eval_results['ensemble']}")

    def _step_save_artifacts(self) -> None:
        manager = ArtifactManager(self.model_cfg.persistence.save_dir, self.run_id)
        manager.save_pipeline(self)


class ArtifactManager:
    """
    模型产物管理器。
    管理模型、预处理器、特征筛选器、评估结果的保存与加载。
    """

    def __init__(self, base_dir: str, run_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.run_id = run_id or "latest"
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 同时维护 latest 软链接（windows 兼容性问题，改为复制）
        self.latest_dir = self.base_dir / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)

    def save_pipeline(self, pipeline: TrainingPipeline) -> None:
        """保存完整 Pipeline 产物"""
        # 1. 模型
        if pipeline.xgb_model is not None:
            pipeline.xgb_model.save(str(self.run_dir / "xgb_model.json"))
            pipeline.xgb_model.save(str(self.latest_dir / "xgb_model.json"))

        if pipeline.mlp_model is not None:
            pipeline.mlp_model.save(str(self.run_dir / "mlp_model.pth"))
            pipeline.mlp_model.save(str(self.latest_dir / "mlp_model.pth"))

        if pipeline.random_forest_model is not None:
            pipeline.random_forest_model.save(
                str(self.run_dir / "random_forest_model.joblib")
            )
            pipeline.random_forest_model.save(
                str(self.latest_dir / "random_forest_model.joblib")
            )

        if pipeline.ensemble is not None:
            pipeline.ensemble.save(str(self.run_dir / "ensemble.json"))
            pipeline.ensemble.save(str(self.latest_dir / "ensemble.json"))

        # 2. 预处理器、特征筛选器（用 joblib）
        if pipeline.preprocessor is not None:
            joblib.dump(pipeline.preprocessor, self.run_dir / "preprocessor.pkl")
            joblib.dump(pipeline.preprocessor, self.latest_dir / "preprocessor.pkl")

        if pipeline.feature_selector is not None:
            joblib.dump(pipeline.feature_selector, self.run_dir / "feature_selector.pkl")
            joblib.dump(pipeline.feature_selector, self.latest_dir / "feature_selector.pkl")

        # 3. 评估结果
        eval_path = self.run_dir / "eval_results.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(pipeline.eval_results, f, ensure_ascii=False, indent=2)
        with open(self.latest_dir / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(pipeline.eval_results, f, ensure_ascii=False, indent=2)

        # 4. 元信息
        meta = {
            "run_id": pipeline.run_id,
            "timestamp": datetime.now().isoformat(),
            "feature_names": pipeline.X_train.columns.tolist() if pipeline.X_train is not None else [],
            "train_samples": len(pipeline.X_train) if pipeline.X_train is not None else 0,
            "val_samples": len(pipeline.X_val) if pipeline.X_val is not None else 0,
            "test_samples": len(pipeline.X_test) if pipeline.X_test is not None else 0,
            "data_source": getattr(pipeline, "data_source_name", None),
        }
        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(self.latest_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"产物已保存至 {self.run_dir}")

    @classmethod
    def load_latest(cls, base_dir: str) -> Dict:
        """加载最新的训练产物"""
        latest_dir = Path(base_dir) / "latest"
        if not latest_dir.exists():
            raise FileNotFoundError(f"未找到训练产物目录: {latest_dir}")

        result = {"dir": str(latest_dir)}

        meta_path = latest_dir / "meta.json"
        if meta_path.exists():
            result["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))

        eval_path = latest_dir / "eval_results.json"
        if eval_path.exists():
            result["eval_results"] = json.loads(eval_path.read_text(encoding="utf-8"))

        return result
