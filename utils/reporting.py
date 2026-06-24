from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.simulation import SimulationResult


def write_simulation_outputs(result: SimulationResult, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage_probabilities": output_dir / "stage_probabilities.csv",
        "champion_probabilities": output_dir / "champion_probabilities.csv",
        "group_probabilities": output_dir / "group_probabilities.csv",
        "group_projection": output_dir / "group_projection.csv",
        "match_predictions": output_dir / "match_predictions.csv",
        "metadata": output_dir / "metadata.json",
    }
    result.stage_probabilities.to_csv(paths["stage_probabilities"], index=False)
    result.champion_probabilities.to_csv(paths["champion_probabilities"], index=False)
    result.group_probabilities.to_csv(paths["group_probabilities"], index=False)
    result.group_projection.to_csv(paths["group_projection"], index=False)
    result.match_predictions.to_csv(paths["match_predictions"], index=False)
    paths["metadata"].write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    return paths


def print_top_tables(result: SimulationResult, top_n: int = 12) -> None:
    pd.set_option("display.max_columns", 20)
    print("\nTop champion probabilities")
    print(result.champion_probabilities.head(top_n).to_string(index=False, formatters={"champion": "{:.3%}".format}))
    print("\nStage probabilities")
    cols = ["team", "round_of_32", "quarterfinal", "semifinal", "final", "champion"]
    formatters = {col: "{:.3%}".format for col in cols if col != "team"}
    print(result.stage_probabilities[cols].head(top_n).to_string(index=False, formatters=formatters))
    print("\nGroup projections")
    group_cols = ["group", "team", "expected_points", "expected_gd"]
    print(result.group_projection[group_cols].to_string(index=False, formatters={"expected_points": "{:.2f}".format, "expected_gd": "{:.2f}".format}))

