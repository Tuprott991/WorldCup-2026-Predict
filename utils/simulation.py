from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from models.base import BaseMatchModel, MatchPrediction


GROUP_LETTERS = list("ABCDEFGHIJKL")

ROUND32_SLOTS = [
    (73, "2A", "2B", None),
    (74, "1E", "3", ["A", "B", "C", "D", "F"]),
    (75, "1F", "2C", None),
    (76, "1C", "2F", None),
    (77, "1I", "3", ["C", "D", "F", "G", "H"]),
    (78, "2E", "2I", None),
    (79, "1A", "3", ["C", "E", "F", "H", "I"]),
    (80, "1L", "3", ["E", "H", "I", "J", "K"]),
    (81, "1D", "3", ["B", "E", "F", "I", "J"]),
    (82, "1G", "3", ["A", "E", "H", "I", "J"]),
    (83, "2K", "2L", None),
    (84, "1H", "2J", None),
    (85, "1B", "3", ["E", "F", "G", "I", "J"]),
    (86, "1J", "2H", None),
    (87, "1K", "3", ["D", "E", "I", "J", "L"]),
    (88, "2D", "2G", None),
]

NEXT_ROUNDS = {
    "round_of_16": [(89, 73, 75), (90, 74, 77), (91, 76, 78), (92, 79, 80), (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)],
    "quarterfinal": [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)],
    "semifinal": [(101, 97, 98), (102, 99, 100)],
    "final": [(104, 101, 102)],
}


@dataclass
class SimulationResult:
    stage_probabilities: pd.DataFrame
    champion_probabilities: pd.DataFrame
    group_probabilities: pd.DataFrame
    group_projection: pd.DataFrame
    match_predictions: pd.DataFrame
    metadata: dict[str, Any]


class PredictionCache:
    def __init__(self, model: BaseMatchModel):
        self.model = model
        self.cache: dict[tuple, MatchPrediction] = {}

    def predict(self, home: str, away: str, neutral: bool, tournament: str = "FIFA World Cup") -> MatchPrediction:
        key = (home, away, bool(neutral), tournament)
        if key not in self.cache:
            self.cache[key] = self.model.predict_match(home, away, neutral=neutral, tournament=tournament)
        return self.cache[key]


class WorldCupSimulator:
    def __init__(
        self,
        model: BaseMatchModel,
        shootouts: pd.DataFrame | None = None,
        simulations: int = 20000,
        seed: int = 42,
        use_known_results: bool = False,
        known_results_cutoff: pd.Timestamp | None = None,
        show_progress: bool = True,
    ):
        self.model = model
        self.cache = PredictionCache(model)
        self.shootout_rates = self._shootout_rates(shootouts if shootouts is not None else pd.DataFrame())
        self.simulations = simulations
        self.rng = np.random.default_rng(seed)
        self.use_known_results = use_known_results
        self.known_results_cutoff = pd.to_datetime(known_results_cutoff) if known_results_cutoff is not None else None
        self.show_progress = show_progress

    def simulate(self, fixtures: pd.DataFrame) -> SimulationResult:
        fixtures = fixtures.copy().sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
        group_map = infer_groups(fixtures)
        teams = sorted(set(fixtures["home_team"]).union(fixtures["away_team"]))
        team_group = {team: group for group, members in group_map.items() for team in members}

        stage_counts = {team: Counter() for team in teams}
        group_counts = {team: Counter() for team in teams}
        group_totals = {team: defaultdict(float) for team in teams}

        for _ in progress_iter(range(self.simulations), total=self.simulations, enabled=self.show_progress):
            standings = {team: self._blank_standing(team, team_group[team]) for team in teams}
            for _, row in fixtures.iterrows():
                hg, ag = self._match_score(row)
                self._apply_group_result(standings[row["home_team"]], standings[row["away_team"]], hg, ag)

            ranked = {group: rank_group([standings[t] for t in members], self.rng) for group, members in group_map.items()}
            qualifiers = self._qualifiers(ranked)

            for group, rows in ranked.items():
                for pos, standing in enumerate(rows, start=1):
                    team = standing["team"]
                    group_counts[team][f"finish_{pos}"] += 1
                    for key in ["points", "gf", "ga", "gd"]:
                        group_totals[team][key] += standing[key]

            for team in teams:
                stage_counts[team]["group"] += 1
            for team in qualifiers.values():
                stage_counts[team]["round_of_32"] += 1

            winners = self._play_knockout(qualifiers, fixtures)
            for stage, stage_winners in winners.items():
                for team in stage_winners:
                    stage_counts[team][stage] += 1

        stage_df = self._stage_df(stage_counts)
        champion_df = stage_df[["team", "champion"]].sort_values("champion", ascending=False).reset_index(drop=True)
        group_prob_df = self._group_probs_df(group_counts, team_group)
        group_projection_df = self._group_projection_df(group_totals, group_counts, team_group)
        match_predictions = predict_fixture_table(
            self.cache,
            fixtures,
            include_known_results=self.use_known_results,
            known_results_cutoff=self.known_results_cutoff,
        )
        metadata = {
            "simulations": self.simulations,
            "model": self.model.name,
            "use_known_results": self.use_known_results,
            "known_results_cutoff": str(self.known_results_cutoff.date()) if self.known_results_cutoff is not None else None,
            "knockout_note": "Round-of-32 slots follow FIFA match slots; advancing third-place teams are assigned by a valid backtracking heuristic.",
        }
        return SimulationResult(stage_df, champion_df, group_prob_df, group_projection_df, match_predictions, metadata)

    def _match_score(self, row: pd.Series) -> tuple[int, int]:
        if (
            self.use_known_results
            and self.known_results_cutoff is not None
            and bool(row.get("is_played", False))
            and pd.to_datetime(row["date"]) <= self.known_results_cutoff
        ):
            return int(row["home_score"]), int(row["away_score"])

        pred = self.cache.predict(row["home_team"], row["away_team"], bool(row["neutral"]), row.get("tournament", "FIFA World Cup"))
        flat_idx = self.rng.choice(pred.score_matrix.size, p=pred.score_matrix.ravel())
        return np.unravel_index(flat_idx, pred.score_matrix.shape)

    def _play_knockout(self, qualifiers: dict[str, str], fixtures: pd.DataFrame) -> dict[str, list[str]]:
        winners_by_match: dict[int, str] = {}
        reached = defaultdict(list)
        third_assignment = assign_third_place_slots(qualifiers)

        for match_id, left_slot, right_slot, _options in ROUND32_SLOTS:
            left = resolve_slot(left_slot, qualifiers, third_assignment, match_id)
            right = resolve_slot(right_slot, qualifiers, third_assignment, match_id)
            winner = self._knockout_winner(left, right)
            winners_by_match[match_id] = winner
            reached["round_of_16"].append(winner)

        stage_reached = {
            "round_of_16": "quarterfinal",
            "quarterfinal": "semifinal",
            "semifinal": "final",
            "final": "champion",
        }
        for source_stage, matches in NEXT_ROUNDS.items():
            target_stage = stage_reached[source_stage]
            for match_id, left_id, right_id in matches:
                left, right = winners_by_match[left_id], winners_by_match[right_id]
                winner = self._knockout_winner(left, right)
                winners_by_match[match_id] = winner
                reached[target_stage].append(winner)
        return reached

    def _knockout_winner(self, home: str, away: str) -> str:
        pred = self.cache.predict(home, away, True, "FIFA World Cup")
        probs = pred.score_matrix
        flat_idx = self.rng.choice(probs.size, p=probs.ravel())
        hg, ag = np.unravel_index(flat_idx, probs.shape)
        if hg > ag:
            return home
        if ag > hg:
            return away
        p_home = self._penalty_prob(home, away)
        return home if self.rng.random() < p_home else away

    def _penalty_prob(self, home: str, away: str) -> float:
        hr = self.shootout_rates.get(home, 0.5)
        ar = self.shootout_rates.get(away, 0.5)
        return float(np.clip(hr / (hr + ar), 0.35, 0.65))

    @staticmethod
    def _blank_standing(team: str, group: str) -> dict:
        return {"team": team, "group": group, "played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0}

    @staticmethod
    def _apply_group_result(home: dict, away: dict, hg: int, ag: int) -> None:
        home["played"] += 1
        away["played"] += 1
        home["gf"] += hg
        home["ga"] += ag
        away["gf"] += ag
        away["ga"] += hg
        home["gd"] = home["gf"] - home["ga"]
        away["gd"] = away["gf"] - away["ga"]
        if hg > ag:
            home["points"] += 3
        elif hg < ag:
            away["points"] += 3
        else:
            home["points"] += 1
            away["points"] += 1

    def _qualifiers(self, ranked: dict[str, list[dict]]) -> dict[str, str]:
        qualifiers = {}
        thirds = []
        for group, rows in ranked.items():
            qualifiers[f"1{group}"] = rows[0]["team"]
            qualifiers[f"2{group}"] = rows[1]["team"]
            thirds.append(rows[2])
        thirds = sorted(thirds, key=lambda r: (r["points"], r["gd"], r["gf"], self.rng.random()), reverse=True)[:8]
        for row in thirds:
            qualifiers[f"3{row['group']}"] = row["team"]
        return qualifiers

    @staticmethod
    def _shootout_rates(shootouts: pd.DataFrame) -> dict[str, float]:
        wins = Counter()
        games = Counter()
        if shootouts.empty:
            return {}
        for _, row in shootouts.iterrows():
            home, away, winner = row.get("home_team"), row.get("away_team"), row.get("winner")
            games[home] += 1
            games[away] += 1
            wins[winner] += 1
        return {team: (wins[team] + 1.0) / (games[team] + 2.0) for team in games}

    def _stage_df(self, counts: dict[str, Counter]) -> pd.DataFrame:
        stages = ["group", "round_of_32", "round_of_16", "quarterfinal", "semifinal", "final", "champion"]
        rows = []
        for team, counter in counts.items():
            row = {"team": team}
            for stage in stages:
                row[stage] = counter[stage] / self.simulations
            rows.append(row)
        return pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)

    def _group_probs_df(self, counts: dict[str, Counter], team_group: dict[str, str]) -> pd.DataFrame:
        rows = []
        for team, counter in counts.items():
            row = {"team": team, "group": team_group[team]}
            for pos in range(1, 5):
                row[f"finish_{pos}"] = counter[f"finish_{pos}"] / self.simulations
            row["top_2"] = row["finish_1"] + row["finish_2"]
            row["top_3"] = row["top_2"] + row["finish_3"]
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["group", "finish_1"], ascending=[True, False]).reset_index(drop=True)

    def _group_projection_df(self, totals: dict[str, defaultdict], counts: dict[str, Counter], team_group: dict[str, str]) -> pd.DataFrame:
        rows = []
        for team in totals:
            row = {"team": team, "group": team_group[team]}
            for key in ["points", "gf", "ga", "gd"]:
                row[f"expected_{key}"] = totals[team][key] / self.simulations
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["group", "expected_points"], ascending=[True, False]).reset_index(drop=True)


def infer_groups(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    graph = defaultdict(set)
    first_date = {}
    for _, row in fixtures.iterrows():
        h, a = row["home_team"], row["away_team"]
        graph[h].add(a)
        graph[a].add(h)
        first_date[h] = min(first_date.get(h, row["date"]), row["date"])
        first_date[a] = min(first_date.get(a, row["date"]), row["date"])

    seen = set()
    components = []
    for team in graph:
        if team in seen:
            continue
        stack = [team]
        comp = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.add(node)
            stack.extend(graph[node] - seen)
        components.append(sorted(comp))

    anchored = {}
    remaining = []
    for comp in components:
        if "Mexico" in comp:
            anchored["A"] = comp
        elif "Canada" in comp:
            anchored["B"] = comp
        elif "United States" in comp:
            anchored["D"] = comp
        else:
            remaining.append(comp)

    available = [g for g in GROUP_LETTERS if g not in anchored]
    remaining.sort(key=lambda comp: min(first_date[t] for t in comp))
    for group, comp in zip(available, remaining):
        anchored[group] = comp
    return {group: anchored[group] for group in GROUP_LETTERS if group in anchored}


def rank_group(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    return sorted(rows, key=lambda r: (r["points"], r["gd"], r["gf"], rng.random()), reverse=True)


def assign_third_place_slots(qualifiers: dict[str, str]) -> dict[int, str]:
    available = {slot[1] for slot in qualifiers if slot.startswith("3")}
    third_slots = [(match_id, options) for match_id, _left, right, options in ROUND32_SLOTS if right == "3"]

    def search(idx: int, used: set[str], assignment: dict[int, str]) -> dict[int, str] | None:
        if idx == len(third_slots):
            return assignment
        match_id, options = third_slots[idx]
        candidates = [g for g in options if g in available and g not in used]
        for group in candidates:
            result = search(idx + 1, used | {group}, {**assignment, match_id: f"3{group}"})
            if result is not None:
                return result
        return None

    result = search(0, set(), {})
    if result is not None:
        return result

    unused = [g for g in available]
    fallback = {}
    for match_id, options in third_slots:
        group = next((g for g in options if g in unused), unused[0])
        fallback[match_id] = f"3{group}"
        if group in unused:
            unused.remove(group)
    return fallback


def resolve_slot(slot: str, qualifiers: dict[str, str], third_assignment: dict[int, str], match_id: int) -> str:
    if slot == "3":
        return qualifiers[third_assignment[match_id]]
    return qualifiers[slot]


def predict_fixture_table(
    cache: PredictionCache,
    fixtures: pd.DataFrame,
    include_known_results: bool = False,
    known_results_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows = []
    for _, row in fixtures.iterrows():
        pred = cache.predict(row["home_team"], row["away_team"], bool(row["neutral"]), row.get("tournament", "FIFA World Cup"))
        show_known = (
            include_known_results
            and known_results_cutoff is not None
            and bool(row.get("is_played", False))
            and pd.to_datetime(row["date"]) <= known_results_cutoff
        )
        item = pred.as_dict()
        item.update(
            {
                "date": row["date"].date(),
                "city": row.get("city"),
                "country": row.get("country"),
                "neutral": bool(row["neutral"]),
                "known_home_score": row["home_score"] if show_known else np.nan,
                "known_away_score": row["away_score"] if show_known else np.nan,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def progress_iter(iterable, total: int, enabled: bool = True):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="Monte Carlo simulations", unit="sim")
    except Exception:
        return iterable
