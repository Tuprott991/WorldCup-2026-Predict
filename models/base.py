from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.poisson import outcome_probs


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    home_xg: float
    away_xg: float
    score_matrix: np.ndarray
    model: str

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
            "home_xg": self.home_xg,
            "away_xg": self.away_xg,
        }


class BaseMatchModel:
    name = "base"

    def fit(self, matches: pd.DataFrame) -> "BaseMatchModel":
        raise NotImplementedError

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        date: Optional[pd.Timestamp] = None,
        country: Optional[str] = None,
    ) -> MatchPrediction:
        raise NotImplementedError


def prediction_from_matrix(
    home_team: str,
    away_team: str,
    matrix: np.ndarray,
    home_xg: float,
    away_xg: float,
    model: str,
) -> MatchPrediction:
    home_win, draw, away_win = outcome_probs(matrix)
    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        home_xg=home_xg,
        away_xg=away_xg,
        score_matrix=matrix,
        model=model,
    )

