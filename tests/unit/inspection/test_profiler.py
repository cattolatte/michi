"""Tests for dataset profiling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from michi.core.artifacts import ColumnKind
from michi.core.errors import DataError
from michi.core.io import load_table
from michi.inspection import profile_table


def _profile(path: Path, target: str | None = None):  # type: ignore[no-untyped-def]
    return profile_table(load_table(path), target=target)


# --- shape and provenance --------------------------------------------------


def test_profile_matches_dataset_shape(messy_csv: Path) -> None:
    """The profile reports the same shape as the underlying file."""
    profile = _profile(messy_csv)
    assert profile.n_rows == 120
    assert profile.n_columns == 13
    assert len(profile.columns) == profile.n_columns


def test_unknown_target_names_available_columns(messy_csv: Path) -> None:
    """Naming a target that does not exist fails with a helpful message."""
    with pytest.raises(DataError, match="available columns"):
        _profile(messy_csv, target="not_a_column")


# --- column classification -------------------------------------------------


def test_numeric_columns_are_classified_numeric(messy_csv: Path) -> None:
    """Integer and float columns land in the numeric kind."""
    assert _profile(messy_csv).column("age").kind is ColumnKind.NUMERIC


def test_all_null_column_is_classified_empty(messy_csv: Path) -> None:
    """A column with no values at all is its own kind, not a category."""
    assert _profile(messy_csv).column("notes").kind is ColumnKind.EMPTY


def test_low_cardinality_strings_are_categorical(messy_csv: Path) -> None:
    """Repeated string values are categories, not free text."""
    assert _profile(messy_csv).column("country").kind is ColumnKind.CATEGORICAL


def test_unique_long_strings_are_text(messy_csv: Path) -> None:
    """A string column distinct in every row is treated as text."""
    assert _profile(messy_csv).column("record_id").kind is ColumnKind.TEXT


def test_datetime_dtype_is_classified_datetime(tmp_path: Path) -> None:
    """Real datetime columns are recognised from their dtype."""
    path = tmp_path / "dates.csv"
    pd.DataFrame({"when": pd.date_range("2024-01-01", periods=60, freq="D")}).to_csv(
        path, index=False
    )
    frame = pd.read_csv(path, parse_dates=["when"])
    parquet = tmp_path / "dates.parquet"
    frame.to_parquet(parquet)
    assert _profile(parquet).column("when").kind is ColumnKind.DATETIME


# --- statistics ------------------------------------------------------------


def test_missing_counts_are_exact(messy_csv: Path) -> None:
    """Missing values are counted, not estimated."""
    column = _profile(messy_csv).column("cabin")
    assert column.missing == 105
    assert column.count == 15


def test_numeric_statistics_are_present_and_ordered(messy_csv: Path) -> None:
    """Quantiles come back in a coherent order for a numeric column."""
    stats = _profile(messy_csv).column("age").stats
    assert stats["min"] <= stats["p25"] <= stats["median"] <= stats["p75"]
    assert stats["p75"] <= stats["max"]


def test_skewed_column_reports_positive_skew(messy_csv: Path) -> None:
    """A right-tailed column reports positive skew."""
    assert _profile(messy_csv).column("fare").stats["skew"] > 1.0


def test_histogram_counts_sum_to_present_values(messy_csv: Path) -> None:
    """Histogram bins account for every non-missing numeric value."""
    column = _profile(messy_csv).column("age")
    assert sum(count for _, _, count in column.histogram) == column.count


def test_empty_column_carries_no_statistics(messy_csv: Path) -> None:
    """A column with nothing in it reports no derived statistics."""
    column = _profile(messy_csv).column("notes")
    assert column.stats == {}
    assert column.histogram == ()


def test_categorical_column_reports_top_values(messy_csv: Path) -> None:
    """Category columns carry their most frequent values for rendering."""
    assert _profile(messy_csv).column("country").top_values[0] == ("JP", 120)


# --- pathological input ----------------------------------------------------


def test_single_row_dataset_profiles_without_error(tmp_path: Path) -> None:
    """A one-row dataset is unusual but not a crash."""
    path = tmp_path / "one.csv"
    pd.DataFrame({"a": [1], "b": ["x"]}).to_csv(path, index=False)
    assert _profile(path).n_rows == 1


def test_unicode_column_names_and_values(tmp_path: Path) -> None:
    """Non-ASCII names and values survive profiling intact."""
    path = tmp_path / "unicode.csv"
    pd.DataFrame({"名前": ["さくら", "みち"], "値": [1, 2]}).to_csv(
        path, index=False, encoding="utf-8"
    )
    profile = _profile(path)
    assert "名前" in profile.column_names


def test_mixed_type_column_profiles_without_error(tmp_path: Path) -> None:
    """A column holding several Python types still profiles."""
    path = tmp_path / "mixed.csv"
    pd.DataFrame({"mixed": ["1", "two", "3", "four"] * 5}).to_csv(path, index=False)
    assert _profile(path).n_columns == 1


def test_duplicate_rows_are_counted(tmp_path: Path) -> None:
    """Exactly duplicated rows are reported at the dataset level."""
    path = tmp_path / "dupes.csv"
    pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}).to_csv(path, index=False)
    assert _profile(path).duplicate_rows == 1
