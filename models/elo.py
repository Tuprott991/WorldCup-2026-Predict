from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from models.base import BaseMatchModel, MatchPrediction, prediction_from_matrix
from utils.data import tournament_weight
from utils.poisson import poisson_score_matrix


class EloModel(BaseMatchModel):
    name = "elo"

    def __init__(self, k_factor: float = 24.0, home_advantage: float = 65.0, max_goals: int = 8):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.max_goals = max_goals
        self.ratings: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame) -> "EloModel":
        self.ratings = {}
        for _, row in matches.sort_values("date").iterrows():
            self._update(row)
        return self

    def rating(self, team: str) -> float:
        return self.ratings.get(team, 1500.0)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        date: Optional[pd.Timestamp] = None,
        country: Optional[str] = None,
    ) -> MatchPrediction:
        home_adv = 0.0 if neutral else self.home_advantage
        diff = self.rating(home_team) + home_adv - self.rating(away_team)
        strength = float(np.clip(diff / 400.0, -1.6, 1.6))
        base_home = 1.32 if not neutral else 1.24
        base_away = 1.12 if not neutral else 1.20
        home_lambda = float(np.clip(base_home * np.exp(0.58 * strength), 0.15, 5.0))
        away_lambda = float(np.clip(base_away * np.exp(-0.58 * strength), 0.15, 5.0))
        matrix = poisson_score_matrix(home_lambda, away_lambda, max_goals=self.max_goals)
        return prediction_from_matrix(home_team, away_team, matrix, home_lambda, away_lambda, self.name)

    def _update(self, row: pd.Series) -> None:
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_score"]), int(row["away_score"])
        self.ratings.setdefault(home, 1500.0)
        self.ratings.setdefault(away, 1500.0)
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        home_adv = 0.0 if bool(row["neutral"]) else self.home_advantage
        expected = 1.0 / (1.0 + 10.0 ** (-(self.ratings[home] + home_adv - self.ratings[away]) / 400.0))
        margin = max(abs(hg - ag), 1)
        k = self.k_factor * np.log1p(margin) * tournament_weight(row.get("tournament", ""))
        delta = k * (actual - expected)
        self.ratings[home] += delta
        self.ratings[away] -= delta

