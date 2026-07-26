"""Tests for finding detection.

Each test asserts on the *kinds* of findings produced, never on exact
statistics, so the suite verifies behaviour rather than pinning numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from michi.core.artifacts import Severity
from michi.core.io import load_table
from michi.inspection import profile_table


def _kinds(path: Path, target: str | None = None) -> set[str]:
    profile = profile_table(load_table(path), target=target)
    return {finding.kind for finding in profile.findings}


def _kinds_for(path: Path, column: str, target: str | None = None) -> set[str]:
    profile = profile_table(load_table(path), target=target)
    return {finding.kind for finding in profile.findings if column in finding.columns}


# --- data quality ----------------------------------------------------------


def test_detects_entirely_missing_column(messy_csv: Path) -> None:
    """An all-null column is reported as empty."""
    assert "empty-column" in _kinds_for(messy_csv, "notes")


def test_detects_constant_column(messy_csv: Path) -> None:
    """A column with a single distinct value is reported as constant."""
    assert "constant-column" in _kinds_for(messy_csv, "country")


def test_detects_mostly_missing_column(messy_csv: Path) -> None:
    """A column past the high-missing threshold is flagged at high severity."""
    assert "high-missing" in _kinds_for(messy_csv, "cabin")


def test_detects_moderate_missingness(messy_csv: Path) -> None:
    """Partial missingness is reported without being called severe."""
    assert "missing" in _kinds_for(messy_csv, "salary")


def test_detects_identifier_like_column(messy_csv: Path) -> None:
    """A column distinct in every row is flagged as identifier-like."""
    assert "identifier-like" in _kinds_for(messy_csv, "record_id")


def test_detects_duplicate_columns(messy_csv: Path) -> None:
    """Columns holding identical values are reported together."""
    assert "duplicate-columns" in _kinds_for(messy_csv, "country_copy")


def test_detects_duplicate_rows(tmp_path: Path) -> None:
    """Repeated rows are reported at the dataset level."""
    path = tmp_path / "dupes.csv"
    pd.DataFrame({"a": [1, 1, 2] * 30, "b": ["x", "x", "y"] * 30}).to_csv(
        path, index=False
    )
    assert "duplicate-rows" in _kinds(path)


# --- distribution ----------------------------------------------------------


def test_detects_skew(messy_csv: Path) -> None:
    """A heavily right-tailed column is reported as skewed."""
    assert "high-skew" in _kinds_for(messy_csv, "fare")


def test_detects_outliers(messy_csv: Path) -> None:
    """Values far outside the interquartile range are reported."""
    assert "outliers" in _kinds_for(messy_csv, "fare")


def test_detects_near_perfect_correlation(messy_csv: Path) -> None:
    """Two columns encoding the same quantity are reported as redundant."""
    assert "highly-correlated" in _kinds_for(messy_csv, "age_months")


# --- encoding problems -----------------------------------------------------


def test_detects_numbers_stored_as_text(messy_csv: Path) -> None:
    """Numeric values behind thousands separators are reported."""
    assert "numeric-stored-as-text" in _kinds_for(messy_csv, "amount_text")


def test_detects_dates_stored_as_text(messy_csv: Path) -> None:
    """Date-shaped strings are reported so they can be parsed."""
    assert "datetime-stored-as-text" in _kinds_for(messy_csv, "signup_date")


# --- target-aware ----------------------------------------------------------


def test_detects_class_imbalance(messy_csv: Path) -> None:
    """A skewed label distribution is reported when a target is named."""
    assert "class-imbalance" in _kinds_for(messy_csv, "purchased", target="purchased")


def test_detects_categorical_leakage_against_numeric_target(messy_csv: Path) -> None:
    """A category encoding the label is caught even when the label is numeric."""
    assert "leakage-suspect" in _kinds_for(
        messy_csv, "outcome_code", target="purchased"
    )


def test_detects_numeric_leakage(tmp_path: Path) -> None:
    """A feature almost perfectly correlated with the target is suspected."""
    rows = 200
    path = tmp_path / "leaky.csv"
    label = [float(value % 50) for value in range(rows)]
    pd.DataFrame(
        {
            "feature": [value * 2 + 1 for value in label],
            "noise": [(value * 7) % 13 for value in range(rows)],
            "label": label,
        }
    ).to_csv(path, index=False)
    assert "leakage-suspect" in _kinds_for(path, "feature", target="label")


def test_target_findings_require_a_target(messy_csv: Path) -> None:
    """Without a named target, no target-specific finding is produced."""
    assert "class-imbalance" not in _kinds(messy_csv)
    assert "leakage-suspect" not in _kinds(messy_csv)


# --- false positives -------------------------------------------------------


def test_clean_dataset_produces_no_high_severity_findings(tidy_csv: Path) -> None:
    """A well-formed dataset is not flagged as problematic."""
    profile = profile_table(load_table(tidy_csv), target="label")
    assert [f for f in profile.findings if f.severity is Severity.HIGH] == []


def test_sparse_column_is_not_called_an_identifier(messy_csv: Path) -> None:
    """A mostly-missing column is reported as missing, not identifier-like."""
    assert "identifier-like" not in _kinds_for(messy_csv, "cabin")


def test_one_row_per_value_is_not_called_leakage(messy_csv: Path) -> None:
    """A column with a distinct value per row partitions any target trivially."""
    assert "leakage-suspect" not in _kinds_for(
        messy_csv, "record_id", target="purchased"
    )


def test_tiny_dataset_is_reported(tmp_path: Path) -> None:
    """Very few rows is itself a finding, because every estimate is uncertain."""
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(path, index=False)
    assert "tiny-dataset" in _kinds(path)
