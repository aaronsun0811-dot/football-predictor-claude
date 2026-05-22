"""i18n coverage gate.

The R44 i18n cleanup + QA round shipped with two regressions that the existing
verification didn't catch:

  1. ``app.js`` had two unescaped double-quotes inside double-quoted JS string
     literals, which made the whole file fail to parse. The browser logged
     hundreds of ``Alpine Expression Error: $t is not defined`` — but the
     pytest suite happily passed, because nothing tested that ``app.js``
     parses as JS.

  2. A bash verification script used the regex ``\\$t\\('([a-z_]+)'\\)`` to
     extract i18n keys from HTML. That regex doesn't match keys containing
     digits (``bt_subtitle_line1``, ``bt_fit_failed_pts_2``). The script
     reported "all keys resolve" while seven keys were actually missing from
     the dictionary — only visible in the browser as the rendered key name
     itself ("bt_subtitle_line1") instead of translated text.

These tests close both gaps. They run with the regular pytest suite, so any
future i18n migration that breaks the dictionary contract fails CI loudly.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "static" / "index.html"
APPJS = ROOT / "static" / "app.js"

# The regex that DOES match all key shapes — including digits + underscores.
# This is the form that previously had a bug ([a-z_]+ instead of [a-z0-9_]+).
KEY_NAME = r"[a-z0-9_]+"
T_CALL_RE = re.compile(rf"\$t\('({KEY_NAME})'\)")
DEFINED_KEY_RE = re.compile(rf"^    ({KEY_NAME}):", re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _used_keys() -> set[str]:
    """Every distinct ``$t('...')`` call site in the HTML."""
    return set(T_CALL_RE.findall(HTML.read_text()))


def _split_zh_en_blocks(appjs: str) -> tuple[str, str]:
    """Slice out the ``zh: { ... }`` and ``en: { ... }`` literal blocks.

    The structure we're parsing:

        const I18N = {
          zh: {
            key1: "...",
            ...
          },
          en: {
            ...
          },
        };

    Returns (zh_block_text, en_block_text).
    """
    zh_start = appjs.find("\n  zh: {\n")
    en_start = appjs.find("\n  en: {\n")
    assert zh_start > 0, "could not find `  zh: {` block start in app.js"
    assert en_start > zh_start, "could not find `  en: {` block start after zh"

    # zh block ends at the line `  },` immediately before en_start.
    zh_block = appjs[zh_start:en_start]
    # en block ends at the next `\n  },` or `\n  }\n};`.
    en_block_tail_match = re.search(r"\n  \},?\s*\n", appjs[en_start:])
    assert en_block_tail_match, "could not find end of en block"
    en_block = appjs[en_start : en_start + en_block_tail_match.start()]
    return zh_block, en_block


def _defined_keys() -> tuple[set[str], set[str]]:
    """Return (zh_keys, en_keys) — every key defined in each locale block."""
    appjs = APPJS.read_text()
    zh_block, en_block = _split_zh_en_blocks(appjs)
    return set(DEFINED_KEY_RE.findall(zh_block)), set(DEFINED_KEY_RE.findall(en_block))


# ---------------------------------------------------------------------------
# Tests — locale-key coverage
# ---------------------------------------------------------------------------

def test_every_used_key_is_defined_in_zh():
    """Catches: HTML calls ``$t('foo')`` but the zh block doesn't define
    ``foo``. Renders as the literal string "foo" in the UI."""
    used = _used_keys()
    zh, _ = _defined_keys()
    missing = sorted(used - zh)
    assert not missing, (
        f"{len(missing)} i18n key(s) used in HTML but missing from zh block:\n"
        f"  {missing[:20]}"
        + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
    )


def test_every_used_key_is_defined_in_en():
    """Same as above, for the en block."""
    used = _used_keys()
    _, en = _defined_keys()
    missing = sorted(used - en)
    assert not missing, (
        f"{len(missing)} i18n key(s) used in HTML but missing from en block:\n"
        f"  {missing[:20]}"
        + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
    )


def test_zh_and_en_blocks_have_identical_keys():
    """The two locale blocks must be symmetric — switching languages must
    never expose a key that exists in one block but not the other."""
    zh, en = _defined_keys()
    only_in_zh = sorted(zh - en)
    only_in_en = sorted(en - zh)
    assert not only_in_zh and not only_in_en, (
        f"zh/en blocks out of sync. "
        f"Only in zh ({len(only_in_zh)}): {only_in_zh[:10]}; "
        f"only in en ({len(only_in_en)}): {only_in_en[:10]}"
    )


# ---------------------------------------------------------------------------
# Tests — app.js syntactic validity
# ---------------------------------------------------------------------------

def test_appjs_parses_as_javascript():
    """Catches: unescaped quotes / brackets / etc. break ``app.js`` so the
    whole app fails to load. Uses ``node --check`` if node is available;
    falls back to a regex-based unescaped-quote detector otherwise."""
    # Prefer node --check if installed (definitive answer, catches everything).
    try:
        result = subprocess.run(
            ["node", "--check", str(APPJS)],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            pytest.fail(
                f"app.js failed `node --check`:\n"
                f"--- stderr ---\n{result.stderr[:2000]}"
            )
        return
    except FileNotFoundError:
        # node not available — fall through to regex check
        pass
    except subprocess.TimeoutExpired:
        pytest.fail("node --check on app.js timed out")

    # Fallback: scan every `key: "..."` line for unescaped inner double-quotes.
    # This is the exact failure mode we hit during the R44 QA round.
    bad = []
    for i, line in enumerate(APPJS.read_text().splitlines(), 1):
        m = re.match(r'^    \w+:\s*"(.*)",?\s*$', line)
        if not m:
            continue
        value = m.group(1)
        # Walk char by char; flag any " not preceded by a backslash.
        for j, ch in enumerate(value):
            if ch == '"' and (j == 0 or value[j - 1] != "\\"):
                bad.append((i, line.strip()[:140]))
                break
    if bad:
        pytest.fail(
            f"{len(bad)} line(s) in app.js contain unescaped double-quotes "
            f"inside a double-quoted string literal:\n"
            + "\n".join(f"  L{ln}: {snip}" for ln, snip in bad[:5])
        )


# ---------------------------------------------------------------------------
# Tests — i18n dictionary entries actually translate (not just keys present)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "max_identical_values",
    [50],  # sanity threshold — see docstring for why
)
def test_zh_values_actually_contain_cjk(max_identical_values):
    """Most zh-block values should contain CJK characters. A few legitimate
    exceptions exist (units, code samples, single-letter labels, punctuation),
    so this test only fires when MANY values look untranslated — which would
    indicate someone accidentally pasted en values into the zh block.

    Threshold (50) is well above the count of legitimate ASCII-only entries
    today (~30: things like ``api_endpoints_listing``, ``ui_lang_zh_label``,
    odds labels, etc.) but well below the size that would suggest a real
    sync bug.
    """
    appjs = APPJS.read_text()
    zh_block, _ = _split_zh_en_blocks(appjs)
    # Extract all key: "value" pairs, count how many values have NO CJK
    no_cjk = []
    for m in re.finditer(r'^    (\w+):\s*"(.*?)",?\s*$', zh_block, re.MULTILINE):
        key, value = m.group(1), m.group(2)
        if not re.search(r"[一-鿿]", value):
            no_cjk.append(key)
    assert len(no_cjk) <= max_identical_values, (
        f"{len(no_cjk)} zh-block values have no CJK characters (threshold: "
        f"{max_identical_values}). Sample: {no_cjk[:15]}. Did en values "
        f"accidentally get pasted into the zh block?"
    )
