"""Tests for the artifact value objects."""

from __future__ import annotations

import pytest

from michi.core.artifacts import (
    PROFILE_SCHEMA_VERSION,
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    Finding,
    Severity,
    SourceInfo,
)


def _source(rows: int = 2) -> SourceInfo:
    return SourceInfo(
        path="data.csv",
        sha256="a" * 64,
        size_bytes=128,
        file_format="csv",
        total_rows=rows,
    )


def _column(name: str = "age", **overrides: object) -> ColumnProfile:
    defaults: dict[str, object] = {
        "name": name,
        "kind": ColumnKind.NUMERIC,
        "dtype": "int64",
        "count": 90,
        "missing": 10,
        "unique": 45,
    }
    defaults.update(overrides)
    return ColumnProfile(**defaults)  # type: ignore[arg-type]


# --- invariants ------------------------------------------------------------


def test_column_rejects_more_unique_than_present() -> None:
    """A column cannot hold more distinct values than non-missing ones."""
    with pytest.raises(ValueError, match="cannot exceed"):
        _column(count=5, unique=6)


def test_column_rejects_negative_counts() -> None:
    """Negative counts are impossible and rejected at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        _column(missing=-1)


def test_profile_rejects_target_outside_columns() -> None:
    """The target must name a column that the profile actually describes."""
    with pytest.raises(ValueError, match="not a column"):
        DatasetProfile(
            source=_source(),
            n_rows=2,
            n_columns=1,
            duplicate_rows=0,
            columns=(_column(),),
            target="nonexistent",
        )


def test_profile_rejects_mismatched_column_count() -> None:
    """A profile's declared width must match the columns it carries."""
    with pytest.raises(ValueError, match="column profiles"):
        DatasetProfile(
            source=_source(),
            n_rows=2,
            n_columns=3,
            duplicate_rows=0,
            columns=(_column(),),
        )


def test_finding_rejects_empty_kind() -> None:
    """Every finding needs a kind so an explanation can be looked up."""
    with pytest.raises(ValueError, match="non-empty"):
        Finding(kind="", severity=Severity.INFO, columns=(), summary="x")


# --- derived values --------------------------------------------------------


def test_missing_percentage_is_share_of_all_rows() -> None:
    """Missing percentage counts against total rows, not present rows."""
    assert _column(count=90, missing=10).missing_pct == pytest.approx(10.0)


def test_empty_column_reports_zero_missing_percentage() -> None:
    """A zero-row column reports 0% rather than dividing by zero."""
    assert _column(count=0, missing=0, unique=0).missing_pct == 0.0


def test_findings_sort_most_severe_first() -> None:
    """Findings order by severity so the terminal shows what matters first."""
    profile = DatasetProfile(
        source=_source(),
        n_rows=2,
        n_columns=1,
        duplicate_rows=0,
        columns=(_column(),),
        findings=(
            Finding(kind="b", severity=Severity.INFO, columns=(), summary="i"),
            Finding(kind="a", severity=Severity.HIGH, columns=(), summary="h"),
            Finding(kind="c", severity=Severity.WARN, columns=(), summary="w"),
        ),
    )
    assert [f.severity for f in profile.findings_by_severity()] == [
        Severity.HIGH,
        Severity.WARN,
        Severity.INFO,
    ]


def test_missing_cells_sums_every_column() -> None:
    """Dataset-level missingness aggregates the per-column counts."""
    profile = DatasetProfile(
        source=_source(100),
        n_rows=100,
        n_columns=2,
        duplicate_rows=0,
        columns=(
            _column("a", count=90, missing=10, unique=5),
            _column("b", count=80, missing=20, unique=5),
        ),
    )
    assert profile.missing_cells == 30


def test_column_lookup_raises_for_unknown_name() -> None:
    """Looking up an absent column is a KeyError, not a silent None."""
    profile = DatasetProfile(
        source=_source(),
        n_rows=2,
        n_columns=1,
        duplicate_rows=0,
        columns=(_column(),),
    )
    with pytest.raises(KeyError):
        profile.column("absent")


# --- serialisation ---------------------------------------------------------


def test_profile_round_trips_through_dict() -> None:
    """Serialising and rebuilding a profile preserves it exactly."""
    profile = DatasetProfile(
        source=_source(100),
        n_rows=100,
        n_columns=2,
        duplicate_rows=3,
        columns=(
            _column("age", histogram=((0.0, 1.0, 4), (1.0, 2.0, 6))),
            _column(
                "city",
                kind=ColumnKind.CATEGORICAL,
                dtype="object",
                count=100,
                missing=0,
                unique=3,
                top_values=(("tokyo", 60), ("osaka", 40)),
            ),
        ),
        findings=(
            Finding(
                kind="missing",
                severity=Severity.WARN,
                columns=("age",),
                summary="10.0% missing",
                metrics={"pct": 10.0},
            ),
        ),
        target="city",
    )
    assert DatasetProfile.from_dict(profile.to_dict()) == profile


def test_profile_records_its_schema_version() -> None:
    """Artifacts carry the schema version that produced them."""
    profile = DatasetProfile(
        source=_source(),
        n_rows=2,
        n_columns=1,
        duplicate_rows=0,
        columns=(_column(),),
    )
    assert profile.to_dict()["schema_version"] == PROFILE_SCHEMA_VERSION


def test_serialised_profile_is_json_safe() -> None:
    """Every value in a serialised profile survives a JSON round trip."""
    import json

    profile = DatasetProfile(
        source=_source(),
        n_rows=2,
        n_columns=1,
        duplicate_rows=0,
        columns=(_column(stats={"mean": 1.5}),),
    )
    assert json.loads(json.dumps(profile.to_dict()))["columns"][0]["stats"] == {
        "mean": 1.5
    }
