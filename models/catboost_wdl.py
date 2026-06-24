from __future__ import annotations

from models.boosted_wdl import BoostedWDLModel


class CatBoostWDLModel(BoostedWDLModel):
    name = "catboost_wdl"

    def __init__(self, max_goals: int = 8):
        super().__init__(max_goals=max_goals, preferred_backend="catboost")

