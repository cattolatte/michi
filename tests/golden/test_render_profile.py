"""Golden tests for rendered output.

michi's product *is* its output, so rendering is pinned with snapshots. The
profile under test is built by hand rather than derived from a file, so the
snapshot contains no timestamps, paths, or machine-specific values and stays
stable across platforms.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console
from syrupy.assertion import SnapshotAssertion

from michi.core.artifacts import (
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    Finding,
    Severity,
    SourceInfo,
)
from michi.core.io import load_table
from michi.inspection import profile_table
from michi.report import render_profile, render_profile_html


@pytest.fixture
def fixed_profile() -> DatasetProfile:
    """A profile with every field pinned, so renderings are reproducible."""
    return DatasetProfile(
        source=SourceInfo(
            path="data/passengers.csv",
            sha256="c" * 64,
            size_bytes=61_194,
            file_format="csv",
            total_rows=891,
        ),
        n_rows=891,
        n_columns=4,
        duplicate_rows=2,
        columns=(
            ColumnProfile(
                name="age",
                kind=ColumnKind.NUMERIC,
                dtype="float64",
                count=714,
                missing=177,
                unique=88,
                stats={
                    "mean": 29.699,
                    "std": 14.526,
                    "min": 0.42,
                    "p25": 20.125,
                    "median": 28.0,
                    "p75": 38.0,
                    "max": 80.0,
                    "skew": 0.389,
                    "outliers": 11.0,
                },
                histogram=((0.0, 40.0, 500), (40.0, 80.0, 214)),
            ),
            ColumnProfile(
                name="cabin",
                kind=ColumnKind.CATEGORICAL,
                dtype="object",
                count=204,
                missing=687,
                unique=147,
                top_values=(("B96", 4), ("C23", 4), ("G6", 4)),
            ),
            ColumnProfile(
                name="embarked",
                kind=ColumnKind.CATEGORICAL,
                dtype="object",
                count=889,
                missing=2,
                unique=3,
                top_values=(("S", 644), ("C", 168), ("Q", 77)),
            ),
            ColumnProfile(
                name="survived",
                kind=ColumnKind.NUMERIC,
                dtype="int64",
                count=891,
                missing=0,
                unique=2,
                stats={"mean": 0.384, "min": 0.0, "max": 1.0},
                histogram=((0.0, 0.5, 549), (0.5, 1.0, 342)),
            ),
        ),
        findings=(
            Finding(
                kind="high-missing",
                severity=Severity.HIGH,
                columns=("cabin",),
                summary="77.1% missing (687 of 891)",
                metrics={"pct": 77.104},
            ),
            Finding(
                kind="missing",
                severity=Severity.WARN,
                columns=("age",),
                summary="19.9% missing (177 of 891)",
                metrics={"pct": 19.865},
            ),
            Finding(
                kind="outliers",
                severity=Severity.INFO,
                columns=("age",),
                summary="11 values beyond 1.5×IQR (1.5%)",
                metrics={"count": 11},
            ),
        ),
        target="survived",
        michi_version="0.0.0",
        created_at="2026-01-01T00:00:00Z",
    )


def _render(profile: DatasetProfile, **kwargs: object) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=88, force_terminal=False, no_color=True)
    render_profile(profile, console, **kwargs)  # type: ignore[arg-type]
    return buffer.getvalue()


# --- terminal --------------------------------------------------------------


def test_terminal_rendering_is_stable(
    fixed_profile: DatasetProfile, snapshot: SnapshotAssertion
) -> None:
    """The default terminal report matches its approved form."""
    assert _render(fixed_profile) == snapshot


def test_explained_rendering_is_stable(
    fixed_profile: DatasetProfile, snapshot: SnapshotAssertion
) -> None:
    """The explained terminal report matches its approved form."""
    assert _render(fixed_profile, explain=True) == snapshot


# --- html ------------------------------------------------------------------


def test_html_rendering_is_stable(
    fixed_profile: DatasetProfile, snapshot: SnapshotAssertion
) -> None:
    """The HTML report matches its approved form."""
    assert render_profile_html(fixed_profile) == snapshot


def test_html_report_is_fully_offline(fixed_profile: DatasetProfile) -> None:
    """A report must open with no network access of any kind."""
    html = render_profile_html(fixed_profile)
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()


def test_html_report_is_self_contained(fixed_profile: DatasetProfile) -> None:
    """Styles and charts are embedded, not linked."""
    html = render_profile_html(fixed_profile)
    assert "<style>" in html
    assert "<svg" in html
    assert "<link" not in html.lower()


def test_html_escapes_column_names(tmp_path: Path) -> None:
    """Column names from untrusted files cannot inject markup."""
    import pandas as pd

    path = tmp_path / "inject.csv"
    pd.DataFrame({"<script>alert(1)</script>": [1, 2, 3]}).to_csv(path, index=False)
    html = render_profile_html(profile_table(load_table(path)))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
