"""Artifact value objects shared across michi verbs.

Artifacts are michi's only form of state: immutable value objects that
serialise to plain JSON, carry an explicit ``schema_version``, and can be
round-tripped without loss. Every verb is a function that reads and/or writes
them.

Design Principles
-----------------
- Artifacts are immutable (`frozen=True, slots=True`) and framework-free:
  no pandas, sklearn, or pydantic types leak into a serialised artifact.
- Serialisation is explicit (`to_dict` / `from_dict`) rather than automatic,
  because the on-disk schema is a public contract that must be readable and
  reviewable independently of the Python classes.
- Every artifact records the michi version and schema version that produced
  it, so old files remain interpretable after the schema evolves.
- Value objects hold no behaviour beyond validation and serialisation;
  analysis lives in the domain packages.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

from michi import __version__

__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "ColumnKind",
    "ColumnProfile",
    "DatasetProfile",
    "Finding",
    "Severity",
    "SourceInfo",
    "utc_now_iso",
]

PROFILE_SCHEMA_VERSION = "1.0"
"""Schema version of the profile artifact. Frozen under semver at michi 1.0."""


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with second precision.

    Examples
    --------
    >>> stamp = utc_now_iso()
    >>> stamp.endswith("Z")
    True
    """
    return _dt.datetime.now(tz=_dt.UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


class ColumnKind(str, Enum):
    """michi's own column taxonomy, independent of any dataframe library.

    The kind drives which statistics are computed and which findings apply.
    It is deliberately coarser than a pandas dtype: users think in terms of
    "numeric" and "categorical", not ``int64`` versus ``Int64``.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    EMPTY = "empty"


class Severity(str, Enum):
    """How strongly a finding warrants attention.

    michi never decides *what to do* about a finding; severity only orders
    the user's attention.
    """

    HIGH = "high"
    WARN = "warn"
    INFO = "info"

    @property
    def rank(self) -> int:
        """Sort rank, most severe first.

        Examples
        --------
        >>> Severity.HIGH.rank < Severity.INFO.rank
        True
        """
        return {"high": 0, "warn": 1, "info": 2}[self.value]


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Per-column statistics produced by ``michi inspect``.

    Parameters
    ----------
    name
        Column name as it appears in the source data.
    kind
        michi's column taxonomy entry (see :class:`ColumnKind`).
    dtype
        The underlying storage dtype, recorded for traceability only.
    count
        Number of non-missing values.
    missing
        Number of missing values.
    unique
        Number of distinct non-missing values.
    stats
        Kind-specific numeric statistics; empty for kinds that have none.
    top_values
        Most frequent values as ``(value, count)`` pairs, for non-numeric
        kinds.

    Examples
    --------
    >>> col = ColumnProfile(
    ...     name="age", kind=ColumnKind.NUMERIC, dtype="float64",
    ...     count=90, missing=10, unique=45,
    ... )
    >>> round(col.missing_pct, 1)
    10.0
    """

    name: str
    kind: ColumnKind
    dtype: str
    count: int
    missing: int
    unique: int
    stats: Mapping[str, float] = field(default_factory=dict)
    top_values: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.count < 0 or self.missing < 0 or self.unique < 0:
            msg = f"column {self.name!r}: counts must be non-negative"
            raise ValueError(msg)
        if self.unique > self.count:
            msg = (
                f"column {self.name!r}: unique ({self.unique}) cannot exceed "
                f"non-missing count ({self.count})"
            )
            raise ValueError(msg)

    @property
    def total(self) -> int:
        """Total number of rows covered by this column profile."""
        return self.count + self.missing

    @property
    def missing_pct(self) -> float:
        """Percentage of rows whose value is missing (0.0 when empty)."""
        return 100.0 * self.missing / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "dtype": self.dtype,
            "count": self.count,
            "missing": self.missing,
            "unique": self.unique,
            "stats": dict(self.stats),
            "top_values": [list(pair) for pair in self.top_values],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a column profile from :meth:`to_dict` output."""
        return cls(
            name=str(payload["name"]),
            kind=ColumnKind(payload["kind"]),
            dtype=str(payload["dtype"]),
            count=int(payload["count"]),
            missing=int(payload["missing"]),
            unique=int(payload["unique"]),
            stats=dict(payload.get("stats", {})),
            top_values=tuple(
                (str(value), int(count))
                for value, count in payload.get("top_values", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class Finding:
    """One observation about a dataset that a user may want to act on.

    A finding states *what is true about the data*, never what michi thinks
    should be done about it. Options are offered separately, by the
    explanation layer, as menus.

    Examples
    --------
    >>> Finding(
    ...     kind="high-missing", severity=Severity.HIGH,
    ...     columns=("cabin",), summary="77.1% missing",
    ... ).kind
    'high-missing'
    """

    kind: str
    severity: Severity
    columns: tuple[str, ...]
    summary: str
    metrics: Mapping[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            msg = "finding kind must be a non-empty identifier"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "kind": self.kind,
            "severity": self.severity.value,
            "columns": list(self.columns),
            "summary": self.summary,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a finding from :meth:`to_dict` output."""
        return cls(
            kind=str(payload["kind"]),
            severity=Severity(payload["severity"]),
            columns=tuple(str(name) for name in payload.get("columns", [])),
            summary=str(payload["summary"]),
            metrics=dict(payload.get("metrics", {})),
        )


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Provenance of the data a profile was computed from.

    The content hash is what makes a profile reproducible: two profiles of
    the same ``sha256`` describe the same bytes, regardless of file name.
    """

    path: str
    sha256: str
    size_bytes: int
    file_format: str
    total_rows: int
    sampled: bool = False
    sample_rows: int | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "format": self.file_format,
            "total_rows": self.total_rows,
            "sampled": self.sampled,
            "sample_rows": self.sample_rows,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild source info from :meth:`to_dict` output."""
        sample_rows = payload.get("sample_rows")
        seed = payload.get("seed")
        return cls(
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            file_format=str(payload["format"]),
            total_rows=int(payload["total_rows"]),
            sampled=bool(payload.get("sampled", False)),
            sample_rows=None if sample_rows is None else int(sample_rows),
            seed=None if seed is None else int(seed),
        )


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """The complete profile artifact written by ``michi inspect``.

    Examples
    --------
    >>> profile = DatasetProfile(
    ...     source=SourceInfo("d.csv", "abc", 10, "csv", 2),
    ...     n_rows=2, n_columns=1, duplicate_rows=0,
    ...     columns=(ColumnProfile("a", ColumnKind.NUMERIC, "int64", 2, 0, 2),),
    ... )
    >>> DatasetProfile.from_dict(profile.to_dict()) == profile
    True
    """

    source: SourceInfo
    n_rows: int
    n_columns: int
    duplicate_rows: int
    columns: tuple[ColumnProfile, ...]
    findings: tuple[Finding, ...] = ()
    target: str | None = None
    schema_version: str = PROFILE_SCHEMA_VERSION
    michi_version: str = __version__
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if len(self.columns) != self.n_columns:
            msg = (
                f"profile declares {self.n_columns} columns but carries "
                f"{len(self.columns)} column profiles"
            )
            raise ValueError(msg)
        if self.target is not None and self.target not in self.column_names:
            msg = f"target {self.target!r} is not a column of this dataset"
            raise ValueError(msg)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Names of every profiled column, in source order."""
        return tuple(column.name for column in self.columns)

    @property
    def missing_cells(self) -> int:
        """Total number of missing cells across all columns."""
        return sum(column.missing for column in self.columns)

    @property
    def missing_pct(self) -> float:
        """Percentage of all cells that are missing."""
        total = self.n_rows * self.n_columns
        return 100.0 * self.missing_cells / total if total else 0.0

    def column(self, name: str) -> ColumnProfile:
        """Return the profile of one column.

        Raises
        ------
        KeyError
            If no column with that name exists.
        """
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def findings_by_severity(self) -> tuple[Finding, ...]:
        """Findings ordered most severe first, then by kind and column."""
        return tuple(
            sorted(
                self.findings,
                key=lambda f: (f.severity.rank, f.kind, f.columns),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole artifact to a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "michi_version": self.michi_version,
            "created_at": self.created_at,
            "source": self.source.to_dict(),
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_rows": self.duplicate_rows,
            "target": self.target,
            "columns": [column.to_dict() for column in self.columns],
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a profile from :meth:`to_dict` output."""
        columns: Sequence[Mapping[str, Any]] = payload.get("columns", [])
        findings: Sequence[Mapping[str, Any]] = payload.get("findings", [])
        target = payload.get("target")
        return cls(
            source=SourceInfo.from_dict(payload["source"]),
            n_rows=int(payload["n_rows"]),
            n_columns=int(payload["n_columns"]),
            duplicate_rows=int(payload["duplicate_rows"]),
            columns=tuple(ColumnProfile.from_dict(item) for item in columns),
            findings=tuple(Finding.from_dict(item) for item in findings),
            target=None if target is None else str(target),
            schema_version=str(payload.get("schema_version", PROFILE_SCHEMA_VERSION)),
            michi_version=str(payload.get("michi_version", __version__)),
            created_at=str(payload.get("created_at", utc_now_iso())),
        )
