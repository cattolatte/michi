"""Tests for reading tabular data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from michi.core.errors import DataError
from michi.core.io import load_table, supported_formats

# --- format handling -------------------------------------------------------


def test_reads_csv_with_provenance(messy_csv: Path) -> None:
    """A CSV loads with its content hash and row count recorded."""
    table = load_table(messy_csv)
    assert table.frame.shape[0] == 120
    assert len(table.source.sha256) == 64
    assert table.source.file_format == "csv"
    assert table.source.total_rows == 120


def test_reads_parquet(tmp_path: Path, tidy_frame: pd.DataFrame) -> None:
    """Parquet files load through the same interface as CSV."""
    path = tmp_path / "data.parquet"
    tidy_frame.to_parquet(path)
    table = load_table(path)
    assert table.frame.shape == tidy_frame.shape
    assert table.source.file_format == "parquet"


def test_hash_identifies_content_not_filename(
    tmp_path: Path, tidy_frame: pd.DataFrame
) -> None:
    """Two copies of the same bytes hash identically under different names."""
    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    tidy_frame.to_csv(first, index=False)
    tidy_frame.to_csv(second, index=False)
    assert load_table(first).source.sha256 == load_table(second).source.sha256


# --- failure modes ---------------------------------------------------------


def test_missing_file_raises_data_error(tmp_path: Path) -> None:
    """A missing path fails with an actionable michi error."""
    with pytest.raises(DataError, match="no such file"):
        load_table(tmp_path / "absent.csv")


def test_unsupported_extension_lists_what_is_supported(tmp_path: Path) -> None:
    """An unsupported file type names the formats michi does read."""
    path = tmp_path / "data.docx"
    path.write_text("not tabular", encoding="utf-8")
    with pytest.raises(DataError, match="michi reads"):
        load_table(path)


def test_empty_file_raises_data_error(tmp_path: Path) -> None:
    """A zero-byte file is rejected before any parser sees it."""
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(DataError, match="empty"):
        load_table(path)


def test_directory_raises_data_error(tmp_path: Path) -> None:
    """Pointing michi at a directory explains that it wants one file."""
    with pytest.raises(DataError, match="directory"):
        load_table(tmp_path)


# --- sampling --------------------------------------------------------------


def test_small_files_are_never_sampled(messy_csv: Path) -> None:
    """Below the size threshold michi reads everything, so nothing is sampled."""
    table = load_table(messy_csv)
    assert table.sampled is False
    assert table.source.sample_rows is None


def test_sampling_is_reproducible_from_the_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same seed selects the same sample, and the seed is recorded."""
    from michi.core import io

    frame = pd.DataFrame({"value": range(5_000)})
    path = tmp_path / "big.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(io, "LARGE_FILE_BYTES", 100)

    first = io.load_table(path, sample_rows=500, seed=7)
    second = io.load_table(path, sample_rows=500, seed=7)

    assert first.sampled is True
    assert first.source.seed == 7
    assert first.source.total_rows == 5_000
    assert first.frame["value"].tolist() == second.frame["value"].tolist()


def test_sampling_respects_the_requested_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sample never exceeds the number of rows the caller asked for."""
    from michi.core import io

    frame = pd.DataFrame({"value": range(5_000)})
    path = tmp_path / "big.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(io, "LARGE_FILE_BYTES", 100)

    table = io.load_table(path, sample_rows=300, seed=0)
    assert table.frame.shape[0] <= 300


def test_full_flag_overrides_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--full` reads every row however large the file is."""
    from michi.core import io

    frame = pd.DataFrame({"value": range(2_000)})
    path = tmp_path / "big.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(io, "LARGE_FILE_BYTES", 100)

    table = io.load_table(path, full=True)
    assert table.sampled is False
    assert table.frame.shape[0] == 2_000


# --- surface ---------------------------------------------------------------


def test_supported_formats_are_extensions() -> None:
    """Reported formats are file extensions users can recognise."""
    assert all(item.startswith(".") for item in supported_formats())
