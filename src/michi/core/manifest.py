"""The run manifest artifact.

A manifest is the durable record of one evaluation, benchmark cell, or sweep
cell: what was measured, on which bytes, with which model, in which
environment. It is what makes a michi result reproducible a year later, and
what ``michi report`` renders.

Design Principles
-----------------
- **A result without provenance is an anecdote.** Every manifest carries the
  dataset hash, model identity, seed, and environment, so a number can always
  be traced back to the conditions that produced it.
- **Metrics carry uncertainty.** A metric is a value *and* an interval where
  one could be computed, because a point estimate invites comparisons the data
  cannot support.
- **Baselines are part of the record, not an option.** A score is meaningless
  without knowing what a trivial model scores on the same data.
- Manifests are plain JSON with an explicit schema version, readable without
  michi installed.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from michi import __version__
from michi.core.artifacts import Finding, SourceInfo, utc_now_iso

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "Environment",
    "Metric",
    "ModelSpec",
    "RunManifest",
    "capture_environment",
]

MANIFEST_SCHEMA_VERSION = "1.0"
"""Schema version of the run manifest. Frozen under semver at michi 1.0."""


@dataclass(frozen=True, slots=True)
class Metric:
    """A measured quantity, with an interval when one could be estimated.

    Parameters
    ----------
    name
        Metric identifier, e.g. ``"accuracy"`` or ``"rmse"``.
    value
        The point estimate.
    ci_low, ci_high
        Bounds of the confidence interval, or ``None`` when uncertainty was
        not estimated.
    greater_is_better
        Direction of improvement, recorded so renderers and comparisons never
        have to guess.

    Examples
    --------
    >>> Metric("rmse", 3.2, greater_is_better=False).has_interval
    False
    >>> Metric("accuracy", 0.9, 0.86, 0.94).has_interval
    True
    """

    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    greater_is_better: bool = True

    def __post_init__(self) -> None:
        if self.ci_low is not None and self.ci_high is not None:
            if self.ci_low > self.ci_high:
                msg = f"metric {self.name!r}: interval bounds are inverted"
                raise ValueError(msg)

    @property
    def has_interval(self) -> bool:
        """Whether an uncertainty interval was estimated for this metric."""
        return self.ci_low is not None and self.ci_high is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "greater_is_better": self.greater_is_better,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a metric from :meth:`to_dict` output."""
        low = payload.get("ci_low")
        high = payload.get("ci_high")
        return cls(
            name=str(payload["name"]),
            value=float(payload["value"]),
            ci_low=None if low is None else float(low),
            ci_high=None if high is None else float(high),
            greater_is_better=bool(payload.get("greater_is_better", True)),
        )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Identity of the model a run evaluated.

    ``reference`` is exactly what the user typed, so a manifest can always be
    re-executed by copying one string.
    """

    reference: str
    loader: str
    class_name: str
    params: Mapping[str, str] = field(default_factory=dict)
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "reference": self.reference,
            "loader": self.loader,
            "class_name": self.class_name,
            "params": dict(self.params),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a model spec from :meth:`to_dict` output."""
        digest = payload.get("sha256")
        return cls(
            reference=str(payload["reference"]),
            loader=str(payload["loader"]),
            class_name=str(payload["class_name"]),
            params=dict(payload.get("params", {})),
            sha256=None if digest is None else str(digest),
        )


@dataclass(frozen=True, slots=True)
class Environment:
    """The machine and library versions a run executed against."""

    python: str
    platform: str
    packages: Mapping[str, str] = field(default_factory=dict)
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "python": self.python,
            "platform": self.platform,
            "packages": dict(self.packages),
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild an environment record from :meth:`to_dict` output."""
        commit = payload.get("git_commit")
        return cls(
            python=str(payload["python"]),
            platform=str(payload["platform"]),
            packages=dict(payload.get("packages", {})),
            git_commit=None if commit is None else str(commit),
        )


def capture_environment() -> Environment:
    """Record the current interpreter, platform, and relevant library versions.

    Only libraries that can change a number are recorded — michi itself and
    the numeric stack. Reading the git commit is best-effort and never fails
    a run.
    """
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str] = {"michi": __version__}
    for name in ("numpy", "pandas", "scikit-learn", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue

    return Environment(
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages=packages,
        git_commit=_git_commit(),
    )


def _git_commit() -> str | None:
    """Return the current git commit, or ``None`` outside a repository.

    Implemented by reading ``.git`` directly rather than shelling out: michi
    makes no subprocess calls, and a data tool must not depend on a git
    binary being installed.
    """
    from pathlib import Path

    directory = Path.cwd().resolve()
    for candidate in (directory, *directory.parents):
        git_dir = candidate / ".git"
        if not git_dir.exists():
            continue
        try:
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not head.startswith("ref:"):
            return head[:40] or None
        ref_path = git_dir / head.removeprefix("ref:").strip()
        try:
            return ref_path.read_text(encoding="utf-8").strip()[:40] or None
        except OSError:
            packed = git_dir / "packed-refs"
            if not packed.exists():
                return None
            target = head.removeprefix("ref:").strip()
            try:
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(f" {target}"):
                        return line.split(" ", maxsplit=1)[0][:40]
            except OSError:
                return None
            return None
    return None


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The durable record of one measured run.

    Examples
    --------
    >>> manifest = RunManifest(
    ...     run_id="20260101T000000Z-abcd1234",
    ...     kind="eval",
    ...     dataset=SourceInfo("d.csv", "a" * 64, 10, "csv", 100),
    ...     target="label",
    ...     task="classification",
    ...     model=ModelSpec("m.pkl", "joblib", "LogisticRegression"),
    ...     metrics=(Metric("accuracy", 0.9),),
    ... )
    >>> RunManifest.from_dict(manifest.to_dict()) == manifest
    True
    """

    run_id: str
    kind: str
    dataset: SourceInfo
    target: str
    task: str
    model: ModelSpec
    metrics: tuple[Metric, ...]
    baselines: Mapping[str, tuple[Metric, ...]] = field(default_factory=dict)
    checks: tuple[Finding, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    n_rows: int = 0
    duration_s: float = 0.0
    environment: Environment = field(default_factory=capture_environment)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    michi_version: str = __version__
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.run_id:
            msg = "a run manifest needs a run_id"
            raise ValueError(msg)
        if self.task not in {"classification", "regression"}:
            msg = f"unknown task {self.task!r}"
            raise ValueError(msg)

    def metric(self, name: str) -> Metric:
        """Return one metric by name.

        Raises
        ------
        KeyError
            If the run did not record that metric.
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    @property
    def primary(self) -> Metric:
        """The headline metric — the first recorded, by convention."""
        if not self.metrics:
            msg = "manifest records no metrics"
            raise ValueError(msg)
        return self.metrics[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole artifact to a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "michi_version": self.michi_version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "kind": self.kind,
            "dataset": self.dataset.to_dict(),
            "target": self.target,
            "task": self.task,
            "model": self.model.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "baselines": {
                name: [metric.to_dict() for metric in metrics]
                for name, metrics in self.baselines.items()
            },
            "checks": [check.to_dict() for check in self.checks],
            "details": dict(self.details),
            "seed": self.seed,
            "n_rows": self.n_rows,
            "duration_s": round(self.duration_s, 4),
            "environment": self.environment.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a manifest from :meth:`to_dict` output."""
        metrics: Sequence[Mapping[str, Any]] = payload.get("metrics", [])
        baselines: Mapping[str, Sequence[Mapping[str, Any]]] = payload.get(
            "baselines", {}
        )
        checks: Sequence[Mapping[str, Any]] = payload.get("checks", [])
        return cls(
            run_id=str(payload["run_id"]),
            kind=str(payload["kind"]),
            dataset=SourceInfo.from_dict(payload["dataset"]),
            target=str(payload["target"]),
            task=str(payload["task"]),
            model=ModelSpec.from_dict(payload["model"]),
            metrics=tuple(Metric.from_dict(item) for item in metrics),
            baselines={
                name: tuple(Metric.from_dict(item) for item in items)
                for name, items in baselines.items()
            },
            checks=tuple(Finding.from_dict(item) for item in checks),
            details=dict(payload.get("details", {})),
            seed=int(payload.get("seed", 0)),
            n_rows=int(payload.get("n_rows", 0)),
            duration_s=float(payload.get("duration_s", 0.0)),
            environment=Environment.from_dict(payload["environment"]),
            schema_version=str(
                payload.get("schema_version", MANIFEST_SCHEMA_VERSION)
            ),
            michi_version=str(payload.get("michi_version", __version__)),
            created_at=str(payload.get("created_at", utc_now_iso())),
        )
