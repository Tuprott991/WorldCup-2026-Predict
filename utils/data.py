from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    results: pd.DataFrame
    shootouts: pd.DataFrame
    goalscorers: pd.DataFrame
    former_names: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_dataset(dataset_dir: str | Path = "dataset") -> DatasetBundle:
    dataset_dir = Path(dataset_dir)
    former = _read_csv(dataset_dir / "former_names.csv")
    results = _read_csv(dataset_dir / "results.csv")
    shootouts = _read_csv(dataset_dir / "shootouts.csv")
    goalscorers = _read_csv(dataset_dir / "goalscorers.csv")

    results = clean_results(results, former)
    shootouts = clean_shootouts(shootouts, former)
    goalscorers = clean_goalscorers(goalscorers, former)
    return DatasetBundle(results=results, shootouts=shootouts, goalscorers=goalscorers, former_names=former)


def clean_results(results: pd.DataFrame, former_names: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = normalize_team_names(df, former_names, ["home_team", "away_team"])
    df["is_played"] = df["home_score"].notna() & df["away_score"].notna()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_shootouts(shootouts: pd.DataFrame, former_names: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if shootouts.empty:
        return shootouts
    df = shootouts.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return normalize_team_names(df, former_names, ["home_team", "away_team", "winner", "first_shooter"])


def clean_goalscorers(goalscorers: pd.DataFrame, former_names: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if goalscorers.empty:
        return goalscorers
    df = goalscorers.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return normalize_team_names(df, former_names, ["home_team", "away_team", "team"])


def normalize_team_names(
    df: pd.DataFrame,
    former_names: Optional[pd.DataFrame],
    columns: list[str],
) -> pd.DataFrame:
    if former_names is None or former_names.empty:
        return df
    out = df.copy()
    aliases = {
        str(row["former"]): str(row["current"])
        for _, row in former_names.dropna(subset=["former", "current"]).iterrows()
    }
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda x: aliases.get(x, x) if pd.notna(x) else x)
    return out


def filter_training_matches(results: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    cutoff_ts = pd.to_datetime(cutoff)
    return results[(results["is_played"]) & (results["date"] <= cutoff_ts)].copy()


def get_worldcup_2026_fixtures(results: pd.DataFrame) -> pd.DataFrame:
    fixtures = results[
        (results["tournament"].eq("FIFA World Cup"))
        & (results["date"].dt.year.eq(2026))
    ].copy()
    fixtures = fixtures.sort_values(["date", "city", "home_team"]).reset_index(drop=True)
    if fixtures.empty:
        raise ValueError("No FIFA World Cup 2026 fixtures found in dataset/results.csv.")
    return fixtures


def default_cutoff(results: pd.DataFrame, mode: str) -> pd.Timestamp:
    if mode == "pre_tournament":
        return pd.Timestamp("2026-06-10")
    played_2026_wc = results[
        results["tournament"].eq("FIFA World Cup")
        & results["date"].dt.year.eq(2026)
        & results["is_played"]
    ]
    if played_2026_wc.empty:
        return pd.Timestamp("2026-06-10")
    return played_2026_wc["date"].max()


def tournament_weight(name: str) -> float:
    text = str(name).lower()
    if "fifa world cup" == text:
        return 1.35
    if "world cup qualification" in text:
        return 1.10
    if "qualification" in text:
        return 0.95
    if "nations league" in text:
        return 0.90
    if "friendly" in text:
        return 0.60
    if any(token in text for token in ["euro", "copa", "asian cup", "african cup", "gold cup"]):
        return 1.05
    return 0.80


def recency_weights(dates: pd.Series, half_life_days: float = 730.0) -> np.ndarray:
    max_date = pd.to_datetime(dates).max()
    age_days = (max_date - pd.to_datetime(dates)).dt.days.clip(lower=0).to_numpy()
    return np.exp(-np.log(2.0) * age_days / half_life_days)

