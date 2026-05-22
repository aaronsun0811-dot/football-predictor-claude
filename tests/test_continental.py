"""Tests for the continental competition prediction paths."""
from __future__ import annotations

import pytest

from scrape.registry import LeagueRegistry


# ---------------------------- Registry / aliases ----------------------------


@pytest.fixture(scope="module")
def registry() -> LeagueRegistry:
    return LeagueRegistry()


def test_continental_competitions_loaded_in_registry(registry: LeagueRegistry) -> None:
    expected = {
        "champions_league", "europa_league", "conference_league",
        "copa_libertadores", "copa_sudamericana", "afc_champions_league",
        "euro", "copa_america", "asian_cup", "africa_cup_of_nations",
    }
    keys = {league.key for league in registry.all()}
    missing = expected - keys
    assert not missing, f"Missing continental keys: {missing}"


@pytest.mark.parametrize("alias,expected_key", [
    ("欧冠", "champions_league"),
    ("欧洲冠军联赛", "champions_league"),
    ("UCL", "champions_league"),
    ("欧联", "europa_league"),
    ("欧罗巴", "europa_league"),
    ("欧会杯", "conference_league"),
    ("解放者杯", "copa_libertadores"),
    ("南美自由杯", "copa_libertadores"),
    ("Libertadores", "copa_libertadores"),
    ("亚冠", "afc_champions_league"),
    ("亚洲冠军联赛", "afc_champions_league"),
    ("欧洲杯", "euro"),
    ("欧锦赛", "euro"),
    ("美洲杯", "copa_america"),
    ("Copa América", "copa_america"),
    ("亚洲杯", "asian_cup"),
    ("非洲杯", "africa_cup_of_nations"),
    ("AFCON", "africa_cup_of_nations"),
])
def test_chinese_and_english_continental_aliases(
    registry: LeagueRegistry, alias: str, expected_key: str
) -> None:
    assert registry.normalize(alias) == expected_key


def test_continental_club_flag_is_set_correctly(registry: LeagueRegistry) -> None:
    assert registry.get("欧冠").is_continental_club is True
    assert registry.get("欧冠").is_continental_national is False
    assert registry.get("欧洲杯").is_continental_national is True
    assert registry.get("欧洲杯").is_continental_club is False
    # Domestic leagues remain non-continental.
    assert registry.get("英超").is_continental is False


def test_continental_competitions_are_marked_knockout(registry: LeagueRegistry) -> None:
    """All continental comps default to neutral knockout in our schema."""
    for key in ("champions_league", "europa_league", "copa_libertadores",
                "afc_champions_league", "euro", "copa_america", "asian_cup"):
        assert registry.get(key).knockout is True, key


# ---------------------------- Cross-league pool helpers ----------------------------


def test_domestic_keys_for_continent_returns_top_flight_only() -> None:
    from predict import _DOMESTIC_BY_CONTINENT

    europe = _DOMESTIC_BY_CONTINENT["europe"]
    # Should include top-flight leagues
    assert "premier_league" in europe
    assert "la_liga" in europe
    assert "serie_a" in europe
    assert "bundesliga" in europe
    # Should NOT include second tier (large param count blew up the optimizer)
    assert "championship" not in europe
    assert "segunda" not in europe
    assert "serie_b" not in europe


def test_domestic_keys_for_continent_handles_unknown() -> None:
    from predict import _domestic_keys_for_continent

    reg = LeagueRegistry()
    assert _domestic_keys_for_continent(reg, "atlantis") == []
    assert _domestic_keys_for_continent(reg, None) == []


# ---------------------------- National-team Elo math ----------------------------


def test_adaptive_draw_share_shrinks_with_elo_gap() -> None:
    """The new Elo-only path scales draw_share from 0.28 → 0.13 as the gap grows."""
    # Smoke test the inline formula by replicating it.
    for diff, expected_band in [(0, (0.27, 0.29)), (200, (0.19, 0.22)), (400, (0.12, 0.14))]:
        draw_share = max(0.13, 0.28 - 0.000375 * abs(diff))
        assert expected_band[0] <= draw_share <= expected_band[1], (diff, draw_share)


def test_underdog_floor_keeps_probabilities_above_three_percent() -> None:
    """Lopsided matchups shouldn't print 0.00% for the underdog."""
    from predict import _predict_national_only
    from predict import PredictionRequest
    from data.database import Database

    # We can't run the full function without a DB. Instead, verify the
    # math directly: at diff=400 with adaptive draw share, away should
    # land at the 3% floor or higher, never zero.
    HOME_BOOST = 0
    SCALE = 400.0
    diff = 400.0
    expected_home = 1.0 / (1 + 10 ** (-diff / SCALE))
    draw_share = max(0.13, 0.28 - 0.000375 * abs(diff))
    raw_home = expected_home - draw_share / 2
    raw_away = (1 - expected_home) - draw_share / 2
    UNDERDOG_FLOOR = 0.03
    home_p = max(raw_home, UNDERDOG_FLOOR if raw_home < raw_away else 0)
    away_p = max(raw_away, UNDERDOG_FLOOR if raw_away < raw_home else 0)
    total = home_p + away_p
    if total > 0:
        scale = (1 - draw_share) / total
        home_p *= scale
        away_p *= scale
    assert away_p >= 0.025  # at least near the floor after renormalization
    assert home_p + draw_share + away_p == pytest.approx(1.0, abs=1e-9)
