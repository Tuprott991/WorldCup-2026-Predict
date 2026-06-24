from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
    lam = float(np.clip(lam, 0.05, 8.0))
    probs = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(max_goals + 1)])
    return probs / probs.sum()


def poisson_score_matrix(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 8,
    rho: float = 0.0,
) -> np.ndarray:
    home = poisson_pmf(home_lambda, max_goals)
    away = poisson_pmf(away_lambda, max_goals)
    matrix = np.outer(home, away)

    if rho:
        # Dixon-Coles low-score correction. Negative rho mildly increases common draw cells.
        lam = float(np.clip(home_lambda, 0.05, 8.0))
        mu = float(np.clip(away_lambda, 0.05, 8.0))
        matrix[0, 0] *= max(0.01, 1 - lam * mu * rho)
        matrix[0, 1] *= max(0.01, 1 + lam * rho)
        matrix[1, 0] *= max(0.01, 1 + mu * rho)
        matrix[1, 1] *= max(0.01, 1 - rho)

    return matrix / matrix.sum()


def outcome_probs(matrix: np.ndarray) -> Tuple[float, float, float]:
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    total = home + draw + away
    return home / total, draw / total, away / total


def reweight_matrix_to_outcomes(
    matrix: np.ndarray,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
) -> np.ndarray:
    matrix = matrix.copy()
    targets = np.array([home_prob, draw_prob, away_prob], dtype=float)
    targets = np.clip(targets, 1e-6, 1.0)
    targets = targets / targets.sum()

    masks = [
        np.tril(np.ones_like(matrix, dtype=bool), -1),
        np.eye(matrix.shape[0], dtype=bool),
        np.triu(np.ones_like(matrix, dtype=bool), 1),
    ]
    for mask, target in zip(masks, targets):
        current = float(matrix[mask].sum())
        if current > 0:
            matrix[mask] *= target / current
    return matrix / matrix.sum()


def matrix_from_wdl_probs(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    max_goals: int = 8,
    total_goals: float = 2.55,
) -> np.ndarray:
    ratio = math.log((home_prob + 1e-5) / (away_prob + 1e-5))
    home_share = 1.0 / (1.0 + math.exp(-ratio))
    draw_adjust = float(np.clip(0.35 - draw_prob, -0.20, 0.25))
    total = float(np.clip(total_goals + draw_adjust, 1.6, 4.2))
    home_lambda = float(np.clip(total * home_share, 0.25, 4.5))
    away_lambda = float(np.clip(total * (1.0 - home_share), 0.25, 4.5))
    matrix = poisson_score_matrix(home_lambda, away_lambda, max_goals=max_goals)
    return reweight_matrix_to_outcomes(matrix, home_prob, draw_prob, away_prob)

