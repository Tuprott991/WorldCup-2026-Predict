from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.base import BaseMatchModel, MatchPrediction, prediction_from_matrix
from utils.data import recency_weights, tournament_weight
from utils.poisson import poisson_score_matrix


class DixonColesPoissonModel(BaseMatchModel):
    name = "dixon_coles"

    def __init__(self, alpha: float = 0.001, max_iter: int = 500, rho: float = -0.08, max_goals: int = 8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.rho = rho
        self.max_goals = max_goals
        self.pipeline: Pipeline | None = None
        self.global_home_goals = 1.35
        self.global_away_goals = 1.10

    def fit(self, matches: pd.DataFrame) -> "DixonColesPoissonModel":
        rows = []
        y = []
        weights = []
        recency = recency_weights(matches["date"])
        for i, (_, row) in enumerate(matches.sort_values("date").iterrows()):
            tw = tournament_weight(row.get("tournament", ""))
            w = float(recency[i] * tw)
            rows.append(self._goal_row(row["home_team"], row["away_team"], True, bool(row["neutral"]), row["tournament"]))
            y.append(float(row["home_score"]))
            weights.append(w)
            rows.append(self._goal_row(row["away_team"], row["home_team"], False, bool(row["neutral"]), row["tournament"]))
            y.append(float(row["away_score"]))
            weights.append(w)

        x = pd.DataFrame(rows)
        categorical = ["team", "opponent"]
        numeric = ["is_home", "neutral", "tournament_weight"]
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", StandardScaler(), numeric),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("pre", pre),
                ("model", PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)),
            ]
        )
        self.pipeline.fit(x, np.array(y), model__sample_weight=np.array(weights))
        self.global_home_goals = float(matches["home_score"].mean())
        self.global_away_goals = float(matches["away_score"].mean())
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
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted.")
        home_row = pd.DataFrame([self._goal_row(home_team, away_team, True, neutral, tournament)])
        away_row = pd.DataFrame([self._goal_row(away_team, home_team, False, neutral, tournament)])
        home_lambda = float(np.clip(self.pipeline.predict(home_row)[0], 0.05, 6.0))
        away_lambda = float(np.clip(self.pipeline.predict(away_row)[0], 0.05, 6.0))
        matrix = poisson_score_matrix(home_lambda, away_lambda, max_goals=self.max_goals, rho=self.rho)
        return prediction_from_matrix(home_team, away_team, matrix, home_lambda, away_lambda, self.name)

    def _goal_row(self, team: str, opponent: str, is_home: bool, neutral: bool, tournament: str) -> dict:
        return {
            "team": team,
            "opponent": opponent,
            "is_home": int(is_home),
            "neutral": int(neutral),
            "tournament_weight": tournament_weight(tournament),
        }

