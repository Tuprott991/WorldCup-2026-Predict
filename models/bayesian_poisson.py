from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from models.base import BaseMatchModel, MatchPrediction, prediction_from_matrix
from utils.data import recency_weights, tournament_weight
from utils.poisson import poisson_score_matrix


class BayesianPoissonModel(BaseMatchModel):
    """Empirical-Bayes Poisson model with shrinkage for sparse international teams."""

    name = "bayesian_poisson"

    def __init__(self, prior_matches: float = 8.0, home_advantage: float = 1.08, max_goals: int = 8):
        self.prior_matches = prior_matches
        self.home_advantage = home_advantage
        self.max_goals = max_goals
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.global_for = 1.22
        self.global_home = 1.35
        self.global_away = 1.10

    def fit(self, matches: pd.DataFrame) -> "BayesianPoissonModel":
        matches = matches.sort_values("date").copy()
        rw = recency_weights(matches["date"])
        gf = defaultdict(float)
        ga = defaultdict(float)
        games = defaultdict(float)
        home_goals = []
        away_goals = []

        for idx, (_, row) in enumerate(matches.iterrows()):
            w = float(rw[idx] * tournament_weight(row.get("tournament", "")))
            home, away = row["home_team"], row["away_team"]
            hg, ag = float(row["home_score"]), float(row["away_score"])
            gf[home] += w * hg
            ga[home] += w * ag
            games[home] += w
            gf[away] += w * ag
            ga[away] += w * hg
            games[away] += w
            home_goals.append(hg)
            away_goals.append(ag)

        self.global_home = float(np.mean(home_goals))
        self.global_away = float(np.mean(away_goals))
        self.global_for = float(np.mean(home_goals + away_goals) if isinstance(home_goals, np.ndarray) else np.mean(home_goals + away_goals))
        if not np.isfinite(self.global_for) or self.global_for <= 0:
            self.global_for = 1.22

        teams = set(games)
        for team in teams:
            g = games[team]
            for_rate = (gf[team] + self.prior_matches * self.global_for) / (g + self.prior_matches)
            against_rate = (ga[team] + self.prior_matches * self.global_for) / (g + self.prior_matches)
            self.attack[team] = float(np.clip(for_rate / self.global_for, 0.35, 2.6))
            self.defense[team] = float(np.clip(against_rate / self.global_for, 0.35, 2.6))
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
        h_att = self.attack.get(home_team, 1.0)
        a_att = self.attack.get(away_team, 1.0)
        h_def = self.defense.get(home_team, 1.0)
        a_def = self.defense.get(away_team, 1.0)
        home_boost = 1.0 if neutral else self.home_advantage
        away_boost = 1.0 / home_boost if not neutral else 1.0
        home_lambda = float(np.clip(self.global_home * h_att * a_def * home_boost, 0.08, 6.0))
        away_lambda = float(np.clip(self.global_away * a_att * h_def * away_boost, 0.08, 6.0))
        matrix = poisson_score_matrix(home_lambda, away_lambda, max_goals=self.max_goals, rho=-0.04)
        return prediction_from_matrix(home_team, away_team, matrix, home_lambda, away_lambda, self.name)

