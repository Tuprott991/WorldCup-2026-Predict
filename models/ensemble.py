from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from models.base import BaseMatchModel, MatchPrediction, prediction_from_matrix
from models.bayesian_poisson import BayesianPoissonModel
from models.boosted_wdl import BoostedWDLModel
from models.catboost_wdl import CatBoostWDLModel
from models.dixon_coles import DixonColesPoissonModel
from models.elo import EloModel
from models.xgboost_wdl import XGBoostWDLModel
from utils.features import result_label
from utils.poisson import reweight_matrix_to_outcomes


class CalibratedEnsembleModel(BaseMatchModel):
    name = "ensemble"

    def __init__(self, max_goals: int = 8, xgboost_device: str = "cpu"):
        self.max_goals = max_goals
        self.xgboost_device = xgboost_device
        self.models: dict[str, BaseMatchModel] = {}
        self.weights: dict[str, float] = {}
        self.calibrator: LogisticRegression | None = None
        self.validation_log_loss: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> "CalibratedEnsembleModel":
        matches = matches.sort_values("date").reset_index(drop=True)
        val_size = min(1500, max(400, int(len(matches) * 0.04)))
        train_idx = max(1000, len(matches) - val_size)
        train, val = matches.iloc[:train_idx], matches.iloc[train_idx:]

        val_models = self._new_models()
        val_prob_by_model = {}
        y_val = np.array([result_label(r["home_score"], r["away_score"]) for _, r in val.iterrows()])
        losses = {}
        for name, model in val_models.items():
            model.fit(train)
            probs = []
            for _, row in val.iterrows():
                pred = model.predict_match(
                    row["home_team"],
                    row["away_team"],
                    neutral=bool(row["neutral"]),
                    tournament=row.get("tournament", ""),
                    date=row["date"],
                    country=row.get("country"),
                )
                probs.append([pred.home_win, pred.draw, pred.away_win])
            probs_arr = np.array(probs)
            val_prob_by_model[name] = probs_arr
            losses[name] = float(log_loss(y_val, np.clip(probs_arr, 1e-6, 1.0), labels=[0, 1, 2]))

        stacked_rows = np.hstack([val_prob_by_model[name] for name in val_models])

        self.validation_log_loss = losses
        inv = {name: 1.0 / max(loss, 1e-6) for name, loss in losses.items()}
        total = sum(inv.values())
        self.weights = {name: value / total for name, value in inv.items()}

        try:
            self.calibrator = LogisticRegression(max_iter=1000, C=2.0)
            self.calibrator.fit(stacked_rows, y_val)
        except Exception:
            self.calibrator = None

        self.models = self._new_models()
        for model in self.models.values():
            model.fit(matches)
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
        preds = []
        stacked = []
        matrix = None
        for name, model in self.models.items():
            pred = model.predict_match(home_team, away_team, neutral, tournament, date, country)
            preds.append((name, pred))
            stacked.extend([pred.home_win, pred.draw, pred.away_win])
            weight = self.weights.get(name, 1.0 / max(len(self.models), 1))
            matrix = pred.score_matrix * weight if matrix is None else matrix + pred.score_matrix * weight

        raw = np.array(
            [
                sum(self.weights.get(name, 0.0) * pred.home_win for name, pred in preds),
                sum(self.weights.get(name, 0.0) * pred.draw for name, pred in preds),
                sum(self.weights.get(name, 0.0) * pred.away_win for name, pred in preds),
            ]
        )
        if self.calibrator is not None:
            calibrated = self.calibrator.predict_proba(np.array(stacked).reshape(1, -1))[0]
            target = np.zeros(3)
            for cls, prob in zip(self.calibrator.classes_, calibrated):
                target[int(cls)] = prob
        else:
            target = raw
        target = np.clip(target, 1e-6, 1.0)
        target = target / target.sum()
        matrix = reweight_matrix_to_outcomes(matrix / matrix.sum(), target[0], target[1], target[2])
        home_xg = float(sum(i * matrix[i, :].sum() for i in range(matrix.shape[0])))
        away_xg = float(sum(i * matrix[:, i].sum() for i in range(matrix.shape[1])))
        return prediction_from_matrix(home_team, away_team, matrix, home_xg, away_xg, self.name)

    def _new_models(self) -> dict[str, BaseMatchModel]:
        models: dict[str, BaseMatchModel] = {
            "elo": EloModel(max_goals=self.max_goals),
            "dixon_coles": DixonColesPoissonModel(max_goals=self.max_goals),
            "bayesian_poisson": BayesianPoissonModel(max_goals=self.max_goals),
        }
        try:
            import lightgbm  # noqa: F401

            models["lightgbm_wdl"] = BoostedWDLModel(max_goals=self.max_goals, preferred_backend="lightgbm")
        except Exception:
            models["boosted_wdl"] = BoostedWDLModel(max_goals=self.max_goals)
        try:
            import xgboost  # noqa: F401

            models["xgboost_wdl"] = XGBoostWDLModel(max_goals=self.max_goals, xgboost_device=self.xgboost_device)
        except Exception:
            pass
        try:
            import catboost  # noqa: F401

            models["catboost_wdl"] = CatBoostWDLModel(max_goals=self.max_goals)
        except Exception:
            pass
        return models
