"""Gradient-boosted hybrid head (Groll et al. pattern, heavily regularized).

Trained ONLY on tournament-grade matches (WC + continental finals): a few
thousand rows at best, so the configuration errs hard toward shrinkage
(small trees, large min_data_in_leaf). LightGBM if available; sklearn
HistGradientBoosting as a drop-in fallback so the pipeline never hard-fails
on environment differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:  # pragma: no cover - environment dependent
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier

    _HAS_LGB = False


@dataclass
class HybridModel:
    num_leaves: int = 8
    min_data_in_leaf: int = 40
    learning_rate: float = 0.05
    n_estimators: int = 200
    feature_fraction: float = 0.8
    seed: int = 26
    feature_names: list[str] = field(default_factory=list)
    _model: object | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame | None = None,
            y_val: np.ndarray | None = None) -> "HybridModel":
        self.feature_names = list(X.columns)
        if _HAS_LGB:
            self._model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3,
                num_leaves=self.num_leaves, min_child_samples=self.min_data_in_leaf,
                learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                colsample_bytree=self.feature_fraction, random_state=self.seed,
                verbose=-1,
            )
            kwargs = {}
            if X_val is not None and y_val is not None:
                kwargs = {"eval_set": [(X_val, y_val)],
                          "callbacks": [lgb.early_stopping(25, verbose=False)]}
            self._model.fit(X, y, **kwargs)
        else:
            self._model = HistGradientBoostingClassifier(
                max_leaf_nodes=self.num_leaves, min_samples_leaf=self.min_data_in_leaf,
                learning_rate=self.learning_rate, max_iter=self.n_estimators,
                random_state=self.seed,
            )
            self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HybridModel not fitted")
        p = self._model.predict_proba(X[self.feature_names])
        # guard: some folds may miss a class; expand to 3 columns
        if p.shape[1] < 3:
            full = np.full((len(X), 3), 1e-9)
            for j, cls in enumerate(self._model.classes_):
                full[:, int(cls)] = p[:, j]
            p = full / full.sum(axis=1, keepdims=True)
        return p

    def feature_importance(self) -> dict[str, float]:
        if _HAS_LGB and self._model is not None:
            imp = self._model.feature_importances_
            return dict(zip(self.feature_names, map(float, imp)))
        return {}
