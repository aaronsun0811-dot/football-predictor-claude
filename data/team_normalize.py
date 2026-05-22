"""Team-name canonicalization across data sources.

Different football data providers spell the same team differently:
  football-data.co.uk      "Real Madrid"
  football-data.org        "Real Madrid CF"
  API-Football             "Real Madrid"
  TheSportsDB              "Real Madrid"
  user typing 中文          "皇家马德里"

Without normalization, the Dixon-Coles model treats them as separate
teams — wrong parameters, wrong predictions, broken joins between sources.

This module owns:
  * Loading ``config/team_aliases.yaml`` (a hand-curated canonical map)
  * ``canonicalize(name)`` — returns the canonical English name
  * ``display_name(name, lang)`` — UI helper for bilingual labels
  * A heuristic fallback so unknown teams degrade gracefully
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIASES_PATH = PROJECT_ROOT / "config" / "team_aliases.yaml"

# Common suffix/prefix tokens that don't change which club we're talking about.
# Used as a fallback heuristic when a team isn't in the explicit alias map.
#
# Grouped by what they actually are, for maintainability:
_BOILERPLATE_TOKENS = {
    # Football-club abbreviations
    "fc", "f.c.", "f.c", "cf", "c.f.", "c.f", "cd", "c.d.",
    "sc", "s.c.", "ac", "a.c.", "fk", "kfc", "ksc",
    "afc", "cfc", "ofc", "ofk", "fsv", "sv", "tsv", "vfb", "vfl",
    "sco", "ol", "om", "psg",  # French abbreviations (Olympique de X, etc.)
    "ss", "us", "asd", "acf", "ssd",  # Italian (Società Sportiva, etc.)
    "rcd", "rc", "rfc",  # Spanish/Belgian
    # Generic descriptors
    "club", "calcio", "futbol", "football", "sport", "sporting",
}
_STRIP_TRAILING_TOKEN_RE = re.compile(
    r"\s+(?:" + "|".join(re.escape(t) for t in _BOILERPLATE_TOKENS) + r")$",
    flags=re.I,
)
_STRIP_LEADING_TOKEN_RE = re.compile(
    r"^(?:" + "|".join(re.escape(t) for t in _BOILERPLATE_TOKENS) + r")\s+",
    flags=re.I,
)
# Year-ish suffixes ("Bologna FC 1909", "Pisa 1909", "Stade Brestois 29",
# "Como 1907"). 2-4 digit numbers at the end — too short and we'd start
# stripping legitimate names like "AC Milan 2".
_STRIP_TRAILING_NUMBER_RE = re.compile(r"\s+\d{2,4}$")


_load_lock = threading.Lock()
_cached_map: dict[str, dict[str, Any]] | None = None  # alias_lc → {canonical, zh}


def _load_aliases(path: Path = ALIASES_PATH) -> dict[str, dict[str, Any]]:
    """Build a flat lowercase-keyed lookup: every alias points to its canonical."""
    global _cached_map
    with _load_lock:
        if _cached_map is not None:
            return _cached_map
        if not path.exists():
            _cached_map = {}
            return _cached_map
        raw = yaml.safe_load(path.read_text()) or {}
        flat: dict[str, dict[str, Any]] = {}
        for canonical, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            aliases = entry.get("aliases") or []
            zh = entry.get("zh")
            # The canonical itself is always a valid alias.
            for variant in [canonical, *aliases]:
                flat[variant.casefold().strip()] = {
                    "canonical": canonical,
                    "zh": zh,
                }
            # And the Chinese name resolves back to the canonical too.
            if zh:
                flat[zh.casefold().strip()] = {"canonical": canonical, "zh": zh}
        _cached_map = flat
        return flat


def _reset_cache_for_tests() -> None:
    """Force-reload the alias map. Tests call this when they mock the file."""
    global _cached_map
    with _load_lock:
        _cached_map = None
    canonicalize.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=4096)
def canonicalize(name: str | None) -> str | None:
    """Return the canonical English name. Unknown names pass through unchanged.

    Lookups:
      1. Exact match (lowercase) on the alias map.
      2. Heuristic-stripped match (e.g. "Liverpool FC" → "Liverpool").
      3. Pass-through (return ``name`` unchanged).
    """
    if name is None:
        return None
    name = name.strip()
    if not name:
        return name
    aliases = _load_aliases()
    needle = name.casefold()

    hit = aliases.get(needle)
    if hit:
        return hit["canonical"]

    # Heuristic: strip common boilerplate suffixes/prefixes iteratively until
    # a fixed point. Multiple layers are common — "Bologna FC 1909" needs to
    # lose both "1909" and "FC" to land at "Bologna". A single pass only
    # catches the outermost layer.
    #
    # Returning the stripped form (rather than the original) is intentional:
    # football-data.org writes "Brentford FC" while other sources write
    # "Brentford", and without an explicit alias for every tier-1 team the two
    # never merged. The (rare) risk of a coincidental collision with a
    # *different* team that happens to share the stripped name is dominated
    # by the (common) benefit of cross-source name unification.
    stripped = name
    for _ in range(4):  # bounded loop; "Bologna FC 1909" needs 2 passes max
        previous = stripped
        stripped = _STRIP_TRAILING_NUMBER_RE.sub("", stripped).strip()
        stripped = _STRIP_TRAILING_TOKEN_RE.sub("", stripped).strip()
        stripped = _STRIP_LEADING_TOKEN_RE.sub("", stripped).strip()
        if stripped == previous:
            break

    if stripped and stripped.casefold() != needle:
        hit = aliases.get(stripped.casefold())
        if hit:
            return hit["canonical"]
        return stripped
    return name


def display_name(name: str | None, *, lang: str = "en") -> str | None:
    """Return Chinese or English display name. Falls back to canonical English."""
    if name is None:
        return None
    canonical = canonicalize(name) or name
    aliases = _load_aliases()
    hit = aliases.get(canonical.casefold())
    if lang == "zh" and hit and hit.get("zh"):
        return hit["zh"]
    return canonical


def normalize_frame_columns(
    frame, columns: tuple[str, ...] = ("home_team", "away_team"),
):
    """In-place canonicalize team-name columns on a pandas DataFrame."""
    for col in columns:
        if col in frame.columns:
            frame[col] = frame[col].astype(str).map(canonicalize)
    return frame


def known_canonicals() -> list[str]:
    """Return all canonical team names (useful for autocomplete or sanity)."""
    aliases = _load_aliases()
    return sorted({v["canonical"] for v in aliases.values()})


def alias_groups() -> dict[str, list[str]]:
    """``canonical → list of variants``. Inverse of the flat alias map."""
    aliases = _load_aliases()
    groups: dict[str, list[str]] = {}
    for variant_lc, entry in aliases.items():
        groups.setdefault(entry["canonical"], []).append(variant_lc)
    return groups
