"""Tests for the team-name canonicalization layer."""
from __future__ import annotations

import pandas as pd
import pytest

from data.team_normalize import (
    alias_groups,
    canonicalize,
    display_name,
    known_canonicals,
    normalize_frame_columns,
)


def test_canonicalize_returns_canonical_for_known_aliases() -> None:
    # Real-world duplicates the diagnostics already surfaced.
    assert canonicalize("Real Madrid CF") == "Real Madrid"
    assert canonicalize("Real Madrid") == "Real Madrid"
    assert canonicalize("FC Barcelona") == "Barcelona"
    assert canonicalize("Barcelona SC") == "Barcelona"
    assert canonicalize("Man City") == "Manchester City"
    assert canonicalize("Manchester City FC") == "Manchester City"
    assert canonicalize("Manchester City") == "Manchester City"


def test_canonicalize_is_case_insensitive() -> None:
    assert canonicalize("REAL MADRID CF") == "Real Madrid"
    assert canonicalize("liverpool fc") == "Liverpool"


def test_canonicalize_handles_chinese_input() -> None:
    assert canonicalize("皇家马德里") == "Real Madrid"
    assert canonicalize("曼联") == "Manchester United"


def test_canonicalize_unknown_team_with_no_boilerplate_passes_through() -> None:
    """An unknown team with no FC/CF/AC suffix is unchanged."""
    assert canonicalize("Atlantis") == "Atlantis"
    assert canonicalize("Bumblepuff") == "Bumblepuff"


def test_canonicalize_strips_boilerplate_suffix_even_for_unknowns() -> None:
    """An unknown 'X FC' canonicalizes to 'X' so cross-source name unification works.

    Earlier behavior returned the original (with FC) — that meant
    football-data.org's 'Brentford FC' and TheSportsDB's 'Brentford' never
    merged and the audit module silently dropped half its rows.
    """
    assert canonicalize("Hypothetical United FC") == "Hypothetical United"
    assert canonicalize("Brentford FC") == "Brentford"
    assert canonicalize("Crystal Palace FC") == "Crystal Palace"
    # Leading boilerplate too
    assert canonicalize("FC Hypothetical United") == "Hypothetical United"


def test_canonicalize_strips_multi_layer_boilerplate() -> None:
    """Iterative stripping handles names with both a year and a club suffix."""
    # Year + FC (real cases from football-data.org Serie A)
    assert canonicalize("Bologna FC 1909") == "Bologna"
    assert canonicalize("Parma Calcio 1913") == "Parma"
    # Year only — Pisa 1909, Como 1907 (founded-year suffixes)
    assert canonicalize("Pisa 1909") == "Pisa"
    assert canonicalize("Como 1907") == "Como"
    # Trailing 2-digit number ("Stade Brestois 29" — département code).
    # Now overridden by explicit alias "Stade Brestois → Brest", so the
    # result is the canonical *English* form. Heuristic stripping is what
    # routes the alias lookup correctly under the hood.
    assert canonicalize("Stade Brestois 29") == "Brest"
    # A made-up "Foo 29" with no explicit alias falls back to plain strip
    assert canonicalize("Foo Bar 29") == "Foo Bar"
    # Club abbreviation tokens added to the strip set
    assert canonicalize("Angers SCO") == "Angers"
    assert canonicalize("Genoa CFC") == "Genoa"


def test_canonicalize_handles_none_and_empty() -> None:
    assert canonicalize(None) is None
    assert canonicalize("") == ""
    assert canonicalize("  ") == ""


def test_display_name_returns_chinese_when_available() -> None:
    assert display_name("Real Madrid", lang="zh") == "皇家马德里"
    assert display_name("Real Madrid CF", lang="zh") == "皇家马德里"
    # Falling back to English when no Chinese exists. Note that "Atlantis FC"
    # canonicalizes to "Atlantis" via the boilerplate-strip heuristic, so the
    # English fallback is the stripped form.
    assert display_name("Atlantis FC", lang="zh") == "Atlantis"


def test_display_name_defaults_to_english() -> None:
    assert display_name("Real Madrid CF") == "Real Madrid"


def test_normalize_frame_columns_handles_a_dataframe() -> None:
    frame = pd.DataFrame({
        "home_team": ["Real Madrid CF", "FC Barcelona", "Atlantis FC"],
        "away_team": ["Manchester City FC", "Bayern München", "Unknown FC"],
        "home_goals": [2, 1, 0],
        "away_goals": [1, 1, 0],
    })
    out = normalize_frame_columns(frame.copy())
    # Known teams resolve via the alias map; unknown "FC"-suffixed teams now
    # get their suffix stripped so cross-source merges work (was the silent
    # bug behind the audit module under-counting resolved predictions).
    assert list(out["home_team"]) == ["Real Madrid", "Barcelona", "Atlantis"]
    assert list(out["away_team"]) == ["Manchester City", "Bayern Munich", "Unknown"]


def test_known_canonicals_returns_distinct_sorted_list() -> None:
    canonicals = known_canonicals()
    assert "Real Madrid" in canonicals
    assert "Barcelona" in canonicals
    assert canonicals == sorted(canonicals)
    # Should be no duplicates
    assert len(canonicals) == len(set(canonicals))


def test_alias_groups_contains_real_madrid_cluster() -> None:
    groups = alias_groups()
    rm_variants = groups.get("Real Madrid", [])
    # Both lowercased variants of "Real Madrid" and "Real Madrid CF" + 皇马
    assert "real madrid" in rm_variants
    assert "real madrid cf" in rm_variants
    assert "皇家马德里" in rm_variants
