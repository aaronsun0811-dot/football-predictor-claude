"""World Cup 2026 Monte Carlo simulator.

The 2026 format: 48 teams, 12 groups of 4. Top two from each group plus the
eight best third-placed teams advance to a 32-team knockout round.

Uses national-team Elo from the ``ratings`` table (scope='national')
populated by ``scrape/update.py``. Outcomes use Elo win-expectancy with a
draw share calibrated against 20 years of international football (~26%).
Knockout draws go to extra time then a coin-flip penalty shootout.

CLI:
    python worldcup.py                     # synthetic Elo-top-48 draw
    python worldcup.py --groups draw.json  # actual draw JSON
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from data.database import DEFAULT_DB_PATH, Database, init_database
from scrape.registry import LeagueRegistry

app = typer.Typer(
    add_completion=False,
    help="Simulate the 2026 World Cup.",
    invoke_without_command=True,
)
console = Console()

DRAW_SHARE_GROUP = 0.26
DRAW_SHARE_KNOCKOUT_90 = 0.22
ELO_HOME_BOOST_HOSTS = 65
SCALE = 400.0
AVG_GROUP_GOALS_TOTAL = 2.6


@dataclass(frozen=True)
class Team:
    name: str
    elo: float
    is_host: bool = False


def _latest_national_ratings(db: Database) -> pd.DataFrame:
    """One row per team with the most recent national Elo."""
    with db.session() as session:
        frame = pd.read_sql_table("ratings", db.engine)
    if frame.empty:
        return frame
    frame = frame[frame["scope"] == "national"].copy()
    if frame.empty:
        return frame
    frame["rating_date"] = pd.to_datetime(frame["rating_date"])
    frame = frame.sort_values("rating_date").groupby("team", as_index=False).tail(1)
    return frame.sort_values("elo", ascending=False).reset_index(drop=True)


def load_groups(path: Path | None, db: Database) -> dict[str, list[Team]]:
    """Load groups from JSON or synthesise from the Elo top 48."""
    ratings = _latest_national_ratings(db)
    if ratings.empty:
        raise RuntimeError(
            "No national-team Elo found. Run `python predict.py update` "
            "(or call `IncrementalUpdater().update_national_team_elos()`)."
        )
    elo_lookup = dict(zip(ratings["team"], ratings["elo"]))

    registry = LeagueRegistry()
    hosts = set((registry.worldcup or {}).get("hosts") or [])

    if path is not None:
        blob = json.loads(Path(path).read_text())
        groups: dict[str, list[Team]] = {}
        for group_id, names in blob.items():
            teams: list[Team] = []
            for name in names:
                if name not in elo_lookup:
                    raise KeyError(
                        f"Team '{name}' not found in national ratings. Check spelling "
                        f"(eloratings.net uses e.g. 'United States', 'South Korea')."
                    )
                teams.append(Team(name=name, elo=float(elo_lookup[name]), is_host=name in hosts))
            groups[group_id] = teams
        return groups

    top48 = ratings.head(48)["team"].tolist()
    if len(top48) < 48:
        raise RuntimeError(f"Only {len(top48)} teams in national ratings; need 48.")

    # Snake pots — deterministic. Real FIFA draw uses confederation constraints
    # we do not model. Provide an explicit groups.json for the real bracket.
    pots = [top48[i * 12 : (i + 1) * 12] for i in range(4)]
    groups: dict[str, list[Team]] = {}
    for idx in range(12):
        gid = chr(ord("A") + idx)
        groups[gid] = [
            Team(name=pots[0][idx], elo=float(elo_lookup[pots[0][idx]]), is_host=pots[0][idx] in hosts),
            Team(name=pots[1][idx], elo=float(elo_lookup[pots[1][idx]]), is_host=pots[1][idx] in hosts),
            Team(name=pots[2][idx], elo=float(elo_lookup[pots[2][idx]]), is_host=pots[2][idx] in hosts),
            Team(name=pots[3][idx], elo=float(elo_lookup[pots[3][idx]]), is_host=pots[3][idx] in hosts),
        ]
    return groups


def _elo_match_probs(
    home: Team,
    away: Team,
    *,
    draw_share: float,
    neutral: bool = True,
) -> tuple[float, float, float]:
    boost = 0.0
    if not neutral and home.is_host:
        boost += ELO_HOME_BOOST_HOSTS
    diff = (home.elo + boost) - away.elo
    expected_home = 1.0 / (1 + 10 ** (-diff / SCALE))
    home_p = max(min(expected_home - draw_share / 2, 1 - draw_share), 0)
    away_p = max(1 - draw_share - home_p, 0)
    return home_p, draw_share, away_p


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm — fine for small λ (group games).
    L = pow(2.718281828, -lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _sim_group_match(rng: random.Random, home: Team, away: Team) -> tuple[int, int]:
    diff = home.elo - away.elo
    skew = 0.6 * (diff / SCALE)
    lam_home = max(AVG_GROUP_GOALS_TOTAL / 2 * (1 + skew), 0.2)
    lam_away = max(AVG_GROUP_GOALS_TOTAL / 2 * (1 - skew), 0.2)
    return _poisson(rng, lam_home), _poisson(rng, lam_away)


def _simulate_group(rng: random.Random, group: list[Team]) -> list[dict]:
    n = len(group)
    pts = [0] * n
    gf = [0] * n
    ga = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            hg, ag = _sim_group_match(rng, group[i], group[j])
            gf[i] += hg
            ga[i] += ag
            gf[j] += ag
            ga[j] += hg
            if hg > ag:
                pts[i] += 3
            elif hg < ag:
                pts[j] += 3
            else:
                pts[i] += 1
                pts[j] += 1
    return [
        {"team": group[i], "pts": pts[i], "gf": gf[i], "ga": ga[i], "gd": gf[i] - ga[i]}
        for i in range(n)
    ]


def _simulate_knockout(rng: random.Random, a: Team, b: Team) -> Team:
    home_p, _draw_p, away_p = _elo_match_probs(a, b, draw_share=DRAW_SHARE_KNOCKOUT_90, neutral=True)
    r = rng.random()
    if r < home_p:
        return a
    if r < home_p + away_p:
        return b
    # 90' draw -> ET slight edge to higher Elo -> penalties coin flip.
    diff = (a.elo - b.elo) / SCALE
    et_home_bonus = 0.04 * diff
    return a if rng.random() < 0.5 + et_home_bonus else b


def simulate(
    groups: dict[str, list[Team]],
    *,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> dict:
    rng = random.Random(seed)
    teams = [t for group in groups.values() for t in group]
    by_name = {t.name: t for t in teams}

    group_topfinish: Counter = Counter()
    third_place: Counter = Counter()
    round_reached: dict[str, Counter] = defaultdict(Counter)
    champions: Counter = Counter()

    for _ in range(n_sims):
        survivors_by_group: dict[str, list[Team]] = {}
        thirds: list[tuple[Team, int, int, int]] = []
        for gid, group in groups.items():
            table = _simulate_group(rng, group)
            ranked = sorted(table, key=lambda r: (-r["pts"], -r["gd"], -r["gf"], rng.random()))
            survivors_by_group[gid] = [r["team"] for r in ranked[:2]]
            third = ranked[2]
            thirds.append((third["team"], third["pts"], third["gd"], third["gf"]))
            for r in ranked[:2]:
                group_topfinish[r["team"].name] += 1
            third_place[third["team"].name] += 1

        thirds.sort(key=lambda r: (-r[1], -r[2], -r[3], rng.random()))
        best_thirds = [r[0] for r in thirds[:8]]

        round32: list[Team] = []
        for gid in sorted(groups):
            round32.extend(survivors_by_group[gid])
        round32.extend(best_thirds)
        rng.shuffle(round32)  # Simplified bracket draw.

        bracket = round32
        for stage_size in (32, 16, 8, 4, 2):
            stage_name = f"R{stage_size}"
            for t in bracket:
                round_reached[stage_name][t.name] += 1
            next_round = []
            for i in range(0, len(bracket), 2):
                next_round.append(_simulate_knockout(rng, bracket[i], bracket[i + 1]))
            bracket = next_round
        champions[bracket[0].name] += 1

    rows = []
    for name, team in by_name.items():
        rows.append(
            {
                "team": name,
                "elo": team.elo,
                "p_top2_group": group_topfinish[name] / n_sims,
                "p_third": third_place[name] / n_sims,
                "p_R16": round_reached["R16"][name] / n_sims,
                "p_quarters": round_reached["R8"][name] / n_sims,
                "p_semis": round_reached["R4"][name] / n_sims,
                "p_final": round_reached["R2"][name] / n_sims,
                "p_champion": champions[name] / n_sims,
            }
        )
    frame = pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
    return {"n_sims": n_sims, "table": frame}


@app.command()
def run(
    groups_file: Path = typer.Option(None, "--groups", help="Path to JSON of groups."),
    n_sims: int = typer.Option(10_000, help="Monte Carlo iterations."),
    top: int = typer.Option(20, help="How many teams to print."),
    seed: int = typer.Option(42, help="RNG seed."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, help="SQLite database path."),
) -> None:
    db = init_database(db_path)
    groups = load_groups(groups_file, db)
    result = simulate(groups, n_sims=n_sims, seed=seed)
    frame = result["table"].head(top)

    table = Table(title=f"World Cup 2026 — {result['n_sims']:,} sims")
    for label, just in [
        ("Team", "left"),
        ("Elo", "right"),
        ("R16", "right"),
        ("QF", "right"),
        ("SF", "right"),
        ("Final", "right"),
        ("Champion", "right"),
    ]:
        table.add_column(label, justify=just)
    for _, row in frame.iterrows():
        table.add_row(
            row["team"],
            f"{row['elo']:.0f}",
            f"{row['p_R16']:.1%}",
            f"{row['p_quarters']:.1%}",
            f"{row['p_semis']:.1%}",
            f"{row['p_final']:.1%}",
            f"{row['p_champion']:.1%}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
