"""Tests for the explanation content.

The content file is prose, so these tests check its *contract* — that every
finding michi can emit has an explanation, and that explanations offer options
instead of giving advice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from michi.core.io import load_table
from michi.explain import explanation_for, explanations
from michi.inspection import profile_table

_ADVICE = re.compile(
    r"\b(we recommend|you should|michi recommends|the best (option|choice)|"
    r"always use|never use)\b",
    re.IGNORECASE,
)

# --- coverage --------------------------------------------------------------


def test_every_emitted_finding_has_an_explanation(
    messy_csv: Path, tidy_csv: Path
) -> None:
    """No finding michi produces is left without prose behind it."""
    kinds: set[str] = set()
    for path, target in ((messy_csv, "purchased"), (tidy_csv, "label")):
        profile = profile_table(load_table(path), target=target)
        kinds.update(finding.kind for finding in profile.findings)

    missing = sorted(kind for kind in kinds if explanation_for(kind) is None)
    assert missing == []


def test_unknown_kind_returns_none() -> None:
    """A finding without content degrades gracefully rather than raising."""
    assert explanation_for("not-a-real-finding") is None


# --- content contract ------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(explanations()))
def test_explanation_has_title_and_meaning(kind: str) -> None:
    """Every entry states what it is and what it means."""
    explanation = explanations()[kind]
    assert explanation.title.strip()
    assert len(explanation.what) > 40


@pytest.mark.parametrize("kind", sorted(explanations()))
def test_explanation_offers_at_least_two_options(kind: str) -> None:
    """Options are a menu; a single option would be a recommendation."""
    assert len(explanations()[kind].options) >= 2


@pytest.mark.parametrize("kind", sorted(explanations()))
def test_explanation_never_gives_advice(kind: str) -> None:
    """michi automates implementation, never judgement."""
    explanation = explanations()[kind]
    body = " ".join([explanation.what, *explanation.options, explanation.caution or ""])
    match = _ADVICE.search(body)
    assert match is None, f"{kind} gives advice: {match.group(0)!r}"
