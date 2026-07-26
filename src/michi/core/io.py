"""Reading tabular data into michi, with honest sampling for large files.

Design Principles
-----------------
- **First contact must be fast.** Files above a size threshold are sampled
  rather than fully loaded, and the sample is always recorded in the artifact
  so no number is ever silently based on part of the data.
- **Sampling is random and seeded**, not "the first N rows": a head-slice of a
  sorted file is a systematically wrong picture of the data.
- **Streaming, bounded memory.** Row counts and samples come from Arrow's
  incremental readers, so a 50 GB CSV costs a constant amount of RAM.
- pandas and pyarrow are implementation details; callers receive a
  :class:`LoadedTable` and never need to know which engine parsed the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from michi.core.artifacts import SourceInfo
from michi.core.errors import DataError
from michi.core.hashing import hash_file

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "DEFAULT_SAMPLE_ROWS",
    "LARGE_FILE_BYTES",
    "LoadedTable",
    "load_table",
    "supported_formats",
]

DEFAULT_SAMPLE_ROWS: Final = 200_000
"""Rows retained when a file is large enough to trigger sampling."""

LARGE_FILE_BYTES: Final = 256 * 1024 * 1024
"""Files larger than this are sampled unless the caller asks for everything."""

_CSV_SUFFIXES: Final = {".csv": ",", ".tsv": "\t", ".txt": ","}
_PARQUET_SUFFIXES: Final = {".parquet", ".pq"}
_EXCEL_SUFFIXES: Final = {".xlsx", ".xlsm", ".xls"}


@dataclass(frozen=True, slots=True)
class LoadedTable:
    """A dataframe together with the provenance of the file it came from."""

    frame: pd.DataFrame
    source: SourceInfo

    @property
    def sampled(self) -> bool:
        """Whether the frame holds a sample rather than the full dataset."""
        return self.source.sampled


def supported_formats() -> tuple[str, ...]:
    """Return the file extensions michi can read.

    Examples
    --------
    >>> ".csv" in supported_formats()
    True
    """
    return tuple(
        sorted({*_CSV_SUFFIXES, *_PARQUET_SUFFIXES, *_EXCEL_SUFFIXES})
    )


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _CSV_SUFFIXES:
        return "csv"
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    if suffix in _EXCEL_SUFFIXES:
        return "excel"
    msg = (
        f"unsupported file type {suffix or path.name!r}; michi reads "
        f"{', '.join(supported_formats())}"
    )
    raise DataError(msg)


def load_table(
    path: Path,
    *,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    full: bool = False,
    seed: int = 0,
) -> LoadedTable:
    """Read a tabular file, sampling it when it is large.

    Parameters
    ----------
    path
        File to read (``.csv``, ``.tsv``, ``.parquet``, or — with the
        ``michi[excel]`` extra — ``.xlsx``).
    sample_rows
        Target number of rows to keep when sampling is triggered.
    full
        Read every row regardless of file size.
    seed
        Seed for the sampling RNG, recorded in the artifact so the same
        sample can be reproduced.

    Returns
    -------
    LoadedTable
        The frame plus its :class:`~michi.core.artifacts.SourceInfo`.

    Raises
    ------
    DataError
        If the file is missing, empty, of an unsupported type, or unparseable.
    """
    if not path.exists():
        msg = f"no such file: {path}"
        raise DataError(msg)
    if path.is_dir():
        msg = f"{path} is a directory; michi inspects a single data file"
        raise DataError(msg)

    file_format = _detect_format(path)
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        msg = f"{path} is empty"
        raise DataError(msg)

    should_sample = not full and size_bytes > LARGE_FILE_BYTES

    if file_format == "csv":
        frame, total_rows, sampled = _read_csv(
            path, sample_rows=sample_rows, should_sample=should_sample, seed=seed
        )
    elif file_format == "parquet":
        frame, total_rows, sampled = _read_parquet(
            path, sample_rows=sample_rows, should_sample=should_sample, seed=seed
        )
    else:
        frame, total_rows, sampled = _read_excel(path)

    if frame.shape[1] == 0:
        msg = f"{path} has no columns michi can read"
        raise DataError(msg)

    source = SourceInfo(
        path=str(path),
        sha256=hash_file(path),
        size_bytes=size_bytes,
        file_format=file_format,
        total_rows=total_rows,
        sampled=sampled,
        sample_rows=int(frame.shape[0]) if sampled else None,
        seed=seed if sampled else None,
    )
    return LoadedTable(frame=frame, source=source)


def _read_csv(
    path: Path, *, sample_rows: int, should_sample: bool, seed: int
) -> tuple[pd.DataFrame, int, bool]:
    """Read a delimited text file, optionally sampling it."""
    import pandas as pd

    delimiter = _CSV_SUFFIXES.get(path.suffix.lower(), ",")

    if not should_sample:
        try:
            frame = pd.read_csv(path, sep=delimiter, encoding="utf-8")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, sep=delimiter, encoding="latin-1")
        except pd.errors.EmptyDataError as err:
            msg = f"{path} contains no parseable rows"
            raise DataError(msg) from err
        except Exception as err:  # noqa: BLE001 - third-party failure boundary
            msg = f"could not parse {path.name} as delimited text: {err}"
            raise DataError(msg) from err
        return frame, int(frame.shape[0]), False

    total_rows = _count_csv_rows(path, delimiter)
    probability = min(1.0, sample_rows / total_rows) if total_rows else 1.0
    frames = _stream_csv_sample(path, delimiter, probability, seed)
    if not frames:
        msg = f"{path} contains no parseable rows"
        raise DataError(msg)
    frame = pd.concat(frames, ignore_index=True)
    if frame.shape[0] > sample_rows:
        frame = frame.iloc[:sample_rows].reset_index(drop=True)
    return frame, total_rows, True


def _count_csv_rows(path: Path, delimiter: str) -> int:
    """Count data rows in a delimited file using Arrow's streaming reader."""
    from pyarrow import csv as pa_csv

    options = pa_csv.ParseOptions(delimiter=delimiter)
    try:
        with pa_csv.open_csv(str(path), parse_options=options) as reader:
            return sum(batch.num_rows for batch in reader)
    except Exception as err:  # noqa: BLE001 - third-party failure boundary
        msg = f"could not scan {path.name}: {err}"
        raise DataError(msg) from err


def _stream_csv_sample(
    path: Path, delimiter: str, probability: float, seed: int
) -> list[pd.DataFrame]:
    """Stream a delimited file, keeping each row with the given probability."""
    import numpy as np
    from pyarrow import csv as pa_csv

    rng = np.random.default_rng(seed)
    options = pa_csv.ParseOptions(delimiter=delimiter)
    frames: list[pd.DataFrame] = []
    try:
        with pa_csv.open_csv(str(path), parse_options=options) as reader:
            for batch in reader:
                mask = rng.random(batch.num_rows) < probability
                if not mask.any():
                    continue
                chunk = batch.to_pandas()
                frames.append(chunk.loc[mask].reset_index(drop=True))
    except Exception as err:  # noqa: BLE001 - third-party failure boundary
        msg = f"could not sample {path.name}: {err}"
        raise DataError(msg) from err
    return frames


def _read_parquet(
    path: Path, *, sample_rows: int, should_sample: bool, seed: int
) -> tuple[pd.DataFrame, int, bool]:
    """Read a parquet file, optionally sampling it via its row groups."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    try:
        parquet_file = pq.ParquetFile(str(path))
    except Exception as err:  # noqa: BLE001 - third-party failure boundary
        msg = f"could not open {path.name} as parquet: {err}"
        raise DataError(msg) from err

    total_rows = int(parquet_file.metadata.num_rows)

    if not should_sample:
        frame = parquet_file.read().to_pandas()
        return frame, total_rows, False

    probability = min(1.0, sample_rows / total_rows) if total_rows else 1.0
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for batch in parquet_file.iter_batches():
        mask = rng.random(batch.num_rows) < probability
        if not mask.any():
            continue
        chunk = batch.to_pandas()
        frames.append(chunk.loc[mask].reset_index(drop=True))
    if not frames:
        return parquet_file.read().to_pandas(), total_rows, False
    frame = pd.concat(frames, ignore_index=True)
    if frame.shape[0] > sample_rows:
        frame = frame.iloc[:sample_rows].reset_index(drop=True)
    return frame, total_rows, True


def _read_excel(path: Path) -> tuple[pd.DataFrame, int, bool]:
    """Read the first sheet of a spreadsheet (requires the excel extra)."""
    import pandas as pd

    try:
        frame = pd.read_excel(path)
    except ImportError as err:
        msg = (
            "reading spreadsheets requires the excel extra: "
            "pip install 'michi[excel]'"
        )
        raise DataError(msg) from err
    except Exception as err:  # noqa: BLE001 - third-party failure boundary
        msg = f"could not read {path.name} as a spreadsheet: {err}"
        raise DataError(msg) from err
    return frame, int(frame.shape[0]), False
