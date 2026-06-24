from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.data import tournament_weight


LABELS = ["home_win", "draw", "away_win"]


@dataclass
class TeamState:
    games: int = 0
    elo: float = 1500.0
    ema_points: float = 1.0
    ema_gf: float = 1.2
    ema_ga: float = 1.2


class RunningFeatureBuilder:
    feature_columns = [
        "home_elo",
        "away_elo",
        "elo_diff",
        "neutral",
        "home_games",
        "away_games",
        "games_diff",
        "home_points_ema",
        "away_points_ema",
        "points_form_diff",
        "home_gf_ema",
        "away_gf_ema",
        "home_ga_ema",
        "away_ga_ema",
        "attack_form_diff",
        "defense_form_diff",
        "tournament_weight",
    ]

    def __init__(self, alpha: float = 0.18, k_factor: float = 22.0, home_advantage: float = 65.0):
        self.alpha = alpha
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.states: defaultdict[str, TeamState] = defaultdict(TeamState)

    def reset(self) -> None:
        self.states = defaultdict(TeamState)

    def features_for_match(self, row: pd.Series | dict) -> dict:
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row.get("neutral", True))
        hs = self.states[home]
        aws = self.states[away]
        tw = tournament_weight(row.get("tournament", ""))
        home_adv = 0.0 if neutral else self.home_advantage
        return {
            "home_elo": hs.elo,
            "away_elo": aws.elo,
            "elo_diff": hs.elo + home_adv - aws.elo,
            "neutral": int(neutral),
            "home_games": hs.games,
            "away_games": aws.games,
            "games_diff": hs.games - aws.games,
            "home_points_ema": hs.ema_points,
            "away_points_ema": aws.ema_points,
            "points_form_diff": hs.ema_points - aws.ema_points,
            "home_gf_ema": hs.ema_gf,
            "away_gf_ema": aws.ema_gf,
            "home_ga_ema": hs.ema_ga,
            "away_ga_ema": aws.ema_ga,
            "attack_form_diff": hs.ema_gf - aws.ema_gf,
            "defense_form_diff": aws.ema_ga - hs.ema_ga,
            "tournament_weight": tw,
        }

    def fit_transform(self, matches: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        self.reset()
        rows = []
        labels = []
        for _, row in matches.sort_values("date").iterrows():
            rows.append(self.features_for_match(row))
            labels.append(result_label(row["home_score"], row["away_score"]))
            self.update(row)
        return pd.DataFrame(rows, columns=self.feature_columns), np.array(labels, dtype=int)

    def fit_current(self, matches: pd.DataFrame) -> "RunningFeatureBuilder":
        self.reset()
        for _, row in matches.sort_values("date").iterrows():
            self.update(row)
        return self

    def update(self, row: pd.Series | dict) -> None:
        home = row["home_team"]
        away = row["away_team"]
        hg = int(row["home_score"])
        ag = int(row["away_score"])
        neutral = bool(row.get("neutral", True))
        tw = tournament_weight(row.get("tournament", ""))

        if hg > ag:
            home_result, away_result = 1.0, 0.0
            home_points, away_points = 3.0, 0.0
        elif hg < ag:
            home_result, away_result = 0.0, 1.0
            home_points, away_points = 0.0, 3.0
        else:
            home_result, away_result = 0.5, 0.5
            home_points, away_points = 1.0, 1.0

        hs = self.states[home]
        aws = self.states[away]
        home_adv = 0.0 if neutral else self.home_advantage
        expected_home = 1.0 / (1.0 + 10.0 ** (-(hs.elo + home_adv - aws.elo) / 400.0))
        margin = max(abs(hg - ag), 1)
        multiplier = np.log1p(margin) * tw
        change = self.k_factor * multiplier * (home_result - expected_home)
        hs.elo += change
        aws.elo -= change

        self._update_team(hs, hg, ag, home_points)
        self._update_team(aws, ag, hg, away_points)

    def _update_team(self, state: TeamState, gf: int, ga: int, points: float) -> None:
        if state.games == 0:
            state.ema_points = points
            state.ema_gf = float(gf)
            state.ema_ga = float(ga)
        else:
            state.ema_points = self.alpha * points + (1 - self.alpha) * state.ema_points
            state.ema_gf = self.alpha * gf + (1 - self.alpha) * state.ema_gf
            state.ema_ga = self.alpha * ga + (1 - self.alpha) * state.ema_ga
        state.games += 1


def result_label(home_score: int | float, away_score: int | float) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2

