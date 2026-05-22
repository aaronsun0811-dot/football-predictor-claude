"""Tests for the cross-source overlap detector + field-priority resolver."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.source_resolver import (
    FIELD_PRIORITY,
    find_overlaps,
    find_score_conflicts,
    overlap_summary,
    pick_winning_value,
)


# ---------------------------------------------------------------------------
# pick_winning_value — the core priority-resolution primitive
# ---------------------------------------------------------------------------

def test_pick_winning_value_respects_priority_order() -> None:
    src, val = pick_winning_value(
        {"football-data.co.uk": 2, "football-data.org": 3, "api-football": 1},
        "home_goals",
    )
    # football-data.org is first in FIELD_PRIORITY["home_goals"]
    assert src == "football-data.org"
    assert val == 3


def test_pick_winning_value_skips_null_higher_priority() -> None:
    src, val = pick_winning_value(
        {"football-data.org": None, "api-football": 2, "football-data.co.uk": 1},
        "home_goals",
    )
    # fd.org has the highest priority but null → skip down to api-football
    assert src == "api-football"
    assert val == 2


def test_pick_winning_value_returns_none_when_all_null() -> None:
    src, val = pick_winning_value(
        {"football-data.org": None, "api-football": None},
        "home_goals",
    )
    assert src is None and val is None


def test_pick_winning_value_falls_back_to_first_when_field_unknown() -> None:
    src, val = pick_winning_value(
        {"src_b": 9, "src_a": 7},
        "some_field_not_in_priority_table",
    )
    # No declared priority → first non-null source in dict-insertion order wins
    assert src == "src_b"
    assert val == 9


# ---------------------------------------------------------------------------
# find_overlaps — single-source vs multi-source distinction
# ---------------------------------------------------------------------------

def test_find_overlaps_returns_empty_when_no_cross_source_coverage() -> None:
    frame = pd.DataFrame([
        {"source": "fd.co.uk", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
        {"source": "fd.co.uk", "league_key": "pl",
         "date": date(2024, 8, 18), "home_team": "Spurs", "away_team": "Liverpool",
         "home_goals": 1, "away_goals": 1},
    ])
    overlaps = find_overlaps(frame)
    assert overlaps.empty


def test_find_overlaps_detects_two_sources_for_same_fixture() -> None:
    frame = pd.DataFrame([
        {"source": "football-data.org", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
        {"source": "football-data.co.uk", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
    ])
    overlaps = find_overlaps(frame)
    assert len(overlaps) == 1
    assert overlaps.iloc[0]["n_sources"] == 2
    assert set(overlaps.iloc[0]["sources"]) == {"football-data.org", "football-data.co.uk"}


def test_find_overlaps_uses_canonical_team_names() -> None:
    """'Real Madrid CF' (fd.org) and 'Real Madrid' (fd.co.uk) should merge."""
    frame = pd.DataFrame([
        {"source": "football-data.org", "league_key": "la_liga",
         "date": date(2024, 9, 1),
         "home_team": "Real Madrid CF", "away_team": "FC Bayern München",
         "home_goals": 3, "away_goals": 2},
        {"source": "football-data.co.uk", "league_key": "la_liga",
         "date": date(2024, 9, 1),
         "home_team": "Real Madrid", "away_team": "Bayern Munich",
         "home_goals": 3, "away_goals": 2},
    ])
    overlaps = find_overlaps(frame)
    assert len(overlaps) == 1


def test_find_overlaps_handles_empty_input() -> None:
    overlaps = find_overlaps(pd.DataFrame())
    assert overlaps.empty


# ---------------------------------------------------------------------------
# find_score_conflicts — overlap + disagreement
# ---------------------------------------------------------------------------

def test_no_conflict_when_scores_agree() -> None:
    frame = pd.DataFrame([
        {"source": "football-data.org", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
        {"source": "football-data.co.uk", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
    ])
    conflicts = find_score_conflicts(frame)
    assert conflicts.empty


def test_score_conflict_picks_priority_winner() -> None:
    """fd.org says 2-1, fd.co.uk says 3-1 → fd.org wins by priority."""
    frame = pd.DataFrame([
        {"source": "football-data.org", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
        {"source": "football-data.co.uk", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 3, "away_goals": 1},
    ])
    conflicts = find_score_conflicts(frame)
    assert len(conflicts) == 1
    row = conflicts.iloc[0]
    assert row["winning_source"] == "football-data.org"
    assert row["winning_score"] == "2-1"
    assert row["home_goals_by_source"]["football-data.org"] == 2
    assert row["home_goals_by_source"]["football-data.co.uk"] == 3


def test_score_conflict_falls_through_to_next_priority_when_top_missing() -> None:
    """api-football says 2-1, fd.co.uk says 3-1, no fd.org → api-football wins."""
    frame = pd.DataFrame([
        {"source": "api-football", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
        {"source": "football-data.co.uk", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 3, "away_goals": 1},
    ])
    conflicts = find_score_conflicts(frame)
    assert len(conflicts) == 1
    assert conflicts.iloc[0]["winning_source"] == "api-football"
    assert conflicts.iloc[0]["winning_score"] == "2-1"


# ---------------------------------------------------------------------------
# overlap_summary — what /data-health gets
# ---------------------------------------------------------------------------

def test_overlap_summary_with_no_overlaps_reports_zeros() -> None:
    frame = pd.DataFrame([
        {"source": "fd.co.uk", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
    ])
    summary = overlap_summary(frame)
    assert summary["overlap_count"] == 0
    assert summary["score_conflict_count"] == 0
    assert summary["top_conflicts"] == []
    assert summary["top_overlaps_no_conflict"] == []


def test_overlap_summary_separates_conflicts_from_benign_overlaps() -> None:
    """A fixture covered by two sources that agree counts as overlap-without-conflict."""
    frame = pd.DataFrame([
        # Benign overlap (agree on score)
        {"source": "football-data.org", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
        {"source": "football-data.co.uk", "league_key": "pl",
         "date": date(2024, 8, 17), "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 2, "away_goals": 1},
        # Conflict (disagree on score)
        {"source": "football-data.org", "league_key": "pl",
         "date": date(2024, 8, 18), "home_team": "Spurs", "away_team": "Liverpool",
         "home_goals": 1, "away_goals": 1},
        {"source": "football-data.co.uk", "league_key": "pl",
         "date": date(2024, 8, 18), "home_team": "Spurs", "away_team": "Liverpool",
         "home_goals": 2, "away_goals": 1},
    ])
    summary = overlap_summary(frame)
    assert summary["overlap_count"] == 2
    assert summary["score_conflict_count"] == 1
    assert len(summary["top_conflicts"]) == 1
    assert len(summary["top_overlaps_no_conflict"]) == 1
    assert summary["top_conflicts"][0]["winning_source"] == "football-data.org"


def test_overlap_summary_handles_empty_frame() -> None:
    summary = overlap_summary(pd.DataFrame())
    assert summary == {
        "overlap_count": 0,
        "score_conflict_count": 0,
        "top_conflicts": [],
        "top_overlaps_no_conflict": [],
    }


def test_field_priority_table_has_known_score_keys() -> None:
    """Sanity check: the priority table covers the columns we declare in docs."""
    for required in ("home_goals", "away_goals", "result"):
        assert required in FIELD_PRIORITY
        assert "football-data.org" in FIELD_PRIORITY[required]
