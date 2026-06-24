from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from models.bayesian_poisson import BayesianPoissonModel
from models.boosted_wdl import BoostedWDLModel
from models.catboost_wdl import CatBoostWDLModel
from models.dixon_coles import DixonColesPoissonModel
from models.elo import EloModel
from models.ensemble import CalibratedEnsembleModel
from models.xgboost_wdl import XGBoostWDLModel
from utils.data import default_cutoff, filter_training_matches, get_worldcup_2026_fixtures, load_dataset
from utils.reporting import print_top_tables, write_simulation_outputs
from utils.simulation import WorldCupSimulator


MODEL_REGISTRY = {
    "elo": EloModel,
    "dixon-coles": DixonColesPoissonModel,
    "boosted": BoostedWDLModel,
    "xgboost": XGBoostWDLModel,
    "catboost": CatBoostWDLModel,
    "bayesian": BayesianPoissonModel,
    "ensemble": CalibratedEnsembleModel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the FIFA World Cup 2026 with multiple match models.")
    parser.add_argument("--dataset-dir", default="dataset", help="Folder containing results.csv, shootouts.csv, goalscorers.csv, former_names.csv.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="ensemble", help="Model to train and simulate.")
    parser.add_argument("--mode", choices=["pre_tournament", "live"], default="pre_tournament", help="pre_tournament ignores all World Cup 2026 scores; live uses known scores up to cutoff.")
    parser.add_argument("--cutoff", default=None, help="Training cutoff date, e.g. 2026-06-10. Defaults by mode.")
    parser.add_argument("--simulations", type=int, default=20000, help="Monte Carlo tournament simulations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="outputs/latest", help="Where CSV tables and metadata are written.")
    parser.add_argument("--max-goals", type=int, default=8, help="Maximum goals in scoreline matrix.")
    parser.add_argument("--xgboost-device", choices=["cpu", "cuda"], default="cpu", help="Device for XGBoost training.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar during Monte Carlo simulation.")
    parser.add_argument("--quiet", action="store_true", help="Do not print summary tables.")
    return parser.parse_args()


def build_model(name: str, max_goals: int, xgboost_device: str = "cpu"):
    cls = MODEL_REGISTRY[name]
    if name == "xgboost":
        return cls(max_goals=max_goals, xgboost_device=xgboost_device)
    if name == "boosted":
        return cls(max_goals=max_goals, xgboost_device=xgboost_device)
    if name == "ensemble":
        return cls(max_goals=max_goals, xgboost_device=xgboost_device)
    return cls(max_goals=max_goals)


def main() -> None:
    args = parse_args()
    bundle = load_dataset(args.dataset_dir)
    cutoff = args.cutoff or default_cutoff(bundle.results, args.mode)
    train_matches = filter_training_matches(bundle.results, cutoff)
    fixtures = get_worldcup_2026_fixtures(bundle.results)

    print(f"Loaded {len(bundle.results):,} total matches.")
    print(f"Training on {len(train_matches):,} played matches up to {cutoff}.")
    print(f"Simulating {len(fixtures)} World Cup 2026 group fixtures with model={args.model}.")

    model = build_model(args.model, args.max_goals, args.xgboost_device)
    train_start = time.perf_counter()
    model.fit(train_matches)
    train_seconds = time.perf_counter() - train_start

    simulator = WorldCupSimulator(
        model=model,
        shootouts=bundle.shootouts,
        simulations=args.simulations,
        seed=args.seed,
        use_known_results=args.mode == "live",
        known_results_cutoff=cutoff,
        show_progress=not args.no_progress and not args.quiet,
    )
    simulate_start = time.perf_counter()
    result = simulator.simulate(fixtures)
    simulate_seconds = time.perf_counter() - simulate_start
    result.metadata.update(
        {
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "training_cutoff": str(cutoff),
            "training_matches": int(len(train_matches)),
            "fixture_rows": int(len(fixtures)),
            "mode": args.mode,
            "train_seconds": round(train_seconds, 3),
            "simulate_seconds": round(simulate_seconds, 3),
            "xgboost_device": args.xgboost_device,
        }
    )
    if hasattr(model, "weights"):
        result.metadata["ensemble_weights"] = getattr(model, "weights")
    if hasattr(model, "validation_log_loss"):
        result.metadata["validation_log_loss"] = getattr(model, "validation_log_loss")
    if hasattr(model, "backend"):
        result.metadata["backend"] = getattr(model, "backend")
    if hasattr(model, "models"):
        result.metadata["model_backends"] = {
            name: getattr(sub_model, "backend", getattr(sub_model, "name", name))
            for name, sub_model in getattr(model, "models").items()
        }

    paths = write_simulation_outputs(result, args.output_dir)
    if not args.quiet:
        print_top_tables(result)
    print("\nWrote outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
