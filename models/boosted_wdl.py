from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from models.base import BaseMatchModel, MatchPrediction, prediction_from_matrix
from utils.features import RunningFeatureBuilder
from utils.poisson import matrix_from_wdl_probs


class BoostedWDLModel(BaseMatchModel):
    name = "boosted_wdl"

    def __init__(
        self,
        max_goals: int = 8,
        preferred_backend: str | None = None,
        xgboost_device: str = "cpu",
    ):
        self.max_goals = max_goals
        self.preferred_backend = preferred_backend
        self.xgboost_device = xgboost_device
        self.feature_builder = RunningFeatureBuilder()
        self.model = None
        self.backend = "sklearn_hist_gradient_boosting"

    def fit(self, matches: pd.DataFrame) -> "BoostedWDLModel":
        x, y = self.feature_builder.fit_transform(matches)
        self.model = self._make_classifier()
        self.model.fit(x, y)
        if self.backend == "xgboost" and self.xgboost_device == "cuda":
            # Training can use CUDA, but simulation inference is tiny and passes CPU pandas frames.
            # Moving inference back to CPU avoids XGBoost's mismatched-device warning.
            self.model.set_params(device="cpu")
        self.feature_builder.fit_current(matches)
        return self

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        date: Optional[pd.Timestamp] = None,
        country: Optional[str] = None,
    ) -> MatchPrediction:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        row = {
            "home_team": home_team,
            "away_team": away_team,
            "neutral": neutral,
            "tournament": tournament,
        }
        x = pd.DataFrame([self.feature_builder.features_for_match(row)], columns=self.feature_builder.feature_columns)
        probs = self.model.predict_proba(x)[0]
        aligned = np.zeros(3, dtype=float)
        for cls, prob in zip(self.model.classes_, probs):
            aligned[int(cls)] = prob
        aligned = np.clip(aligned, 1e-6, 1.0)
        aligned = aligned / aligned.sum()
        total_goals = float(np.clip((x["home_gf_ema"].iloc[0] + x["away_gf_ema"].iloc[0]) / 2.0 + 1.3, 1.8, 3.8))
        matrix = matrix_from_wdl_probs(aligned[0], aligned[1], aligned[2], max_goals=self.max_goals, total_goals=total_goals)
        home_xg = float(sum(i * matrix[i, :].sum() for i in range(matrix.shape[0])))
        away_xg = float(sum(i * matrix[:, i].sum() for i in range(matrix.shape[1])))
        return prediction_from_matrix(home_team, away_team, matrix, home_xg, away_xg, self.name)

    def _make_classifier(self):
        makers = {
            "xgboost": self._make_xgboost,
            "lightgbm": self._make_lightgbm,
            "catboost": self._make_catboost,
            "sklearn": self._make_sklearn,
        }
        if self.preferred_backend:
            try:
                return makers[self.preferred_backend]()
            except KeyError as exc:
                raise ValueError(f"Unknown boosted backend: {self.preferred_backend}") from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Requested backend '{self.preferred_backend}' is not available. "
                    "Install it in the active environment or use --model boosted for automatic fallback."
                ) from exc

        for backend in ["xgboost", "lightgbm", "catboost", "sklearn"]:
            try:
                return makers[backend]()
            except Exception:
                continue
        return self._make_sklearn()

    def _make_xgboost(self):
        from xgboost import XGBClassifier

        self.backend = "xgboost"
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=350,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2.0,
            reg_lambda=1.5,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            device=self.xgboost_device,
        )

    def _make_lightgbm(self):
        from lightgbm import LGBMClassifier

        self.backend = "lightgbm"
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=350,
            learning_rate=0.035,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )

    def _make_catboost(self):
        from catboost import CatBoostClassifier

        self.backend = "catboost"
        return CatBoostClassifier(
            iterations=350,
            depth=5,
            learning_rate=0.035,
            loss_function="MultiClass",
            verbose=False,
            random_seed=42,
        )

    def _make_sklearn(self):
        self.backend = "sklearn_hist_gradient_boosting"
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=220,
            learning_rate=0.045,
            l2_regularization=0.05,
            random_state=42,
        )
