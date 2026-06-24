from __future__ import annotations

from models.boosted_wdl import BoostedWDLModel


class XGBoostWDLModel(BoostedWDLModel):
    name = "xgboost_wdl"

    def __init__(self, max_goals: int = 8, xgboost_device: str = "cpu"):
        super().__init__(
            max_goals=max_goals,
            preferred_backend="xgboost",
            xgboost_device=xgboost_device,
        )
