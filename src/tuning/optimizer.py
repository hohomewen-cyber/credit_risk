"""
基于 Optuna 的超参优化
======================
通过 TPE（树结构 Parzen 估计器）算法在配置定义的搜索空间中寻找最优参数。

要点:
    1. 配置驱动的搜索空间
    2. 支持多种采样器（TPE/Random/Grid）
    3. 早停剪枝（避免无效 trial 浪费时间）
    4. 完整的实验追踪
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_auc_score

from src.utils.logger import get_logger

logger = get_logger("tuning.optuna")

# 关闭 Optuna 默认日志
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    通用 Optuna 超参优化器。

    使用示例:
        >>> optimizer = OptunaOptimizer(model_config.xgboost.tuning)
        >>> best_params = optimizer.optimize(
        ...     model_class=XGBClassifier,
        ...     X_train=X_train, y_train=y_train,
        ...     X_val=X_val, y_val=y_val,
        ...     base_params={"objective": "binary:logistic"}
        ... )
    """

    def __init__(self, config):
        self.config = config
        self.search_space = config.search_space
        self.n_trials = config.n_trials
        self.timeout = config.timeout
        self.direction = config.direction
        self.metric = config.metric

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None

    def optimize(
        self,
        model_class: Callable,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_params: Optional[Dict] = None,
        custom_objective: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        执行超参优化。

        参数:
            model_class: 模型构造函数（如 xgb.XGBClassifier）
            X_train, y_train: 训练集
            X_val, y_val: 验证集
            base_params: 基础参数（不参与优化）
            custom_objective: 自定义目标函数（覆盖默认逻辑）

        返回:
            最佳超参字典
        """
        base_params = base_params or {}

        # 构造目标函数
        if custom_objective is not None:
            objective_func = custom_objective
        else:
            objective_func = self._build_default_objective(
                model_class, X_train, y_train, X_val, y_val, base_params
            )

        # 创建 Study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"opt_{model_class.__name__}",
        )

        logger.info(
            f"开始超参优化 - n_trials={self.n_trials}, timeout={self.timeout}s, "
            f"direction={self.direction}"
        )

        # 执行
        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params
        logger.info(
            f"优化完成 - 最佳 {self.metric}: {self.study.best_value:.4f}, "
            f"最佳参数: {self.best_params}"
        )

        return self.best_params

    def _build_default_objective(
        self,
        model_class: Callable,
        X_train, y_train, X_val, y_val,
        base_params: Dict,
    ) -> Callable:
        """构造默认目标函数（适用于大多数 sklearn 风格模型）"""

        def objective(trial: optuna.Trial) -> float:
            # 从搜索空间采样
            params = {}
            for param_name, param_cfg in self.search_space.items():
                params[param_name] = self._suggest_param(trial, param_name, param_cfg)

            # 合并基础参数
            full_params = {**base_params, **params}

            # 创建并训练模型
            try:
                model = model_class(**full_params)
                # XGBoost 早停
                if "early_stopping_rounds" in full_params or hasattr(model, "set_params"):
                    try:
                        model.set_params(early_stopping_rounds=30)
                    except Exception:
                        pass

                fit_kwargs = {}
                # 尝试启用早停
                try:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                except TypeError:
                    model.fit(X_train, y_train)

                # 验证
                val_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, val_pred)

                return score
            except Exception as e:
                logger.warning(f"Trial 失败: {e}")
                return 0.0

        return objective

    def _suggest_param(self, trial: optuna.Trial, name: str, cfg: Dict):
        """根据配置生成参数建议值"""
        ptype = cfg.get("type", "float")

        if ptype == "int":
            return trial.suggest_int(name, cfg["low"], cfg["high"])
        elif ptype == "float":
            return trial.suggest_float(
                name,
                cfg["low"],
                cfg["high"],
                log=cfg.get("log", False),
            )
        elif ptype == "categorical":
            return trial.suggest_categorical(name, cfg["choices"])
        else:
            raise ValueError(f"不支持的参数类型: {ptype}")

    def get_history_df(self) -> pd.DataFrame:
        """获取所有 trial 的历史记录"""
        if self.study is None:
            return pd.DataFrame()
        return self.study.trials_dataframe()

    def get_param_importance(self) -> Dict[str, float]:
        """计算参数重要性（哪些参数对结果影响最大）"""
        if self.study is None:
            return {}
        try:
            return optuna.importance.get_param_importances(self.study)
        except Exception as e:
            logger.warning(f"计算参数重要性失败: {e}")
            return {}
