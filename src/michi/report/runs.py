"""Reading recorded runs back off disk.

Design Principles
-----------------
- **The runs directory is the database.** There is no index, no daemon, and
  no state beyond the JSON files themselves, so a runs directory can be
  copied, committed, or emailed and still make sense.
- **Grouping follows the data, not the clock.** Runs are grouped by the
  dataset they scored and the target they predicted, because those are what
  make two numbers comparable at all.
- **Unreadable files are reported, never fatal.** One malformed manifest must
  not stop a report over ninety good ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from michi.core.errors import ReportError
from michi.core.manifest import RunManifest

__all__ = ["RunGroup", "group_runs", "load_manifests"]


@dataclass(frozen=True, slots=True)
class RunGroup:
    """Runs that scored the same target on the same data, so are comparable."""

    dataset: str
    dataset_sha: str
    target: str
    task: str
    manifests: tuple[RunManifest, ...]

    @property
    def primary_metric(self) -> str:
        """The metric these runs are ranked by."""
        return self.manifests[0].metrics[0].name if self.manifests[0].metrics else ""

    def ranked(self) -> tuple[RunManifest, ...]:
        """Manifests ordered best first by their headline metric."""
        scored = [item for item in self.manifests if item.metrics]
        if not scored:
            return self.manifests
        greater_is_better = scored[0].primary.greater_is_better
        return tuple(
            sorted(
                scored,
                key=lambda item: item.primary.value,
                reverse=greater_is_better,
            )
        )


def load_manifests(source: Path) -> tuple[RunManifest, ...]:
    """Load every run manifest under a path.

    Parameters
    ----------
    source
        A directory of ``*.json`` manifests, or a single manifest file.

    Returns
    -------
    tuple of RunManifest
        Every manifest that parsed, oldest first.

    Raises
    ------
    ReportError
        If the path does not exist, or holds no readable manifest.
    """
    if not source.exists():
        msg = f"no such path: {source}"
        raise ReportError(msg)

    paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
    if not paths:
        msg = (
            f"{source} contains no run manifests. Run 'michi eval' or "
            "'michi bench' first — they write manifests to runs/ by default."
        )
        raise ReportError(msg)

    manifests: list[RunManifest] = []
    skipped: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifests.append(RunManifest.from_dict(payload))
        except (OSError, ValueError, KeyError, TypeError):
            skipped.append(path.name)

    if not manifests:
        listed = ", ".join(skipped[:5])
        msg = f"no readable run manifest in {source} (tried: {listed})"
        raise ReportError(msg)

    manifests.sort(key=lambda item: item.created_at)
    return tuple(manifests)


def group_runs(manifests: tuple[RunManifest, ...]) -> tuple[RunGroup, ...]:
    """Group runs into sets that can honestly be compared with one another.

    Examples
    --------
    Runs over different datasets or different targets never share a group,
    because their metrics are not on the same scale.
    """
    buckets: dict[tuple[str, str], list[RunManifest]] = {}
    for manifest in manifests:
        key = (manifest.dataset.sha256, manifest.target)
        buckets.setdefault(key, []).append(manifest)

    groups: list[RunGroup] = []
    for (sha, target), items in buckets.items():
        groups.append(
            RunGroup(
                dataset=Path(items[0].dataset.path).name,
                dataset_sha=sha,
                target=target,
                task=items[0].task,
                manifests=tuple(items),
            )
        )
    groups.sort(key=lambda group: (group.dataset, group.target))
    return tuple(groups)
