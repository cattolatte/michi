"""Executing a sweep, with caching and resumption.

Design Principles
-----------------
- **A long sweep will be interrupted.** Machines sleep, sessions drop, and
  someone always hits Ctrl-C. Resumption is therefore not a nicety; a sweep
  that cannot resume is a sweep that cannot be run.
- **Caching is by content, never by position.** A cell is skipped only when a
  recorded result carries the same hash of data, recipe, model, seed, and
  folds. Changing anything invalidates exactly the cells it affects.
- **One cell's failure is not the sweep's failure.** A model that cannot train
  on one recipe is recorded and the grid continues; losing thirty completed
  cells to the thirty-first would be indefensible.
- **Progress is honest.** The full grid is known before the first fit, so
  "12 of 30" means what it says.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from michi.core.errors import RunError
from michi.core.hashing import hash_file
from michi.core.io import load_table
from michi.core.manifest import RunManifest
from michi.sweep.plan import SweepCell, SweepPlan

__all__ = ["CellOutcome", "SweepResult", "run_sweep"]


@dataclass(frozen=True, slots=True)
class CellOutcome:
    """What happened to one cell of the grid."""

    cell: SweepCell
    status: str
    manifest: RunManifest | None = None
    error: str | None = None
    seconds: float = 0.0

    @property
    def score(self) -> float | None:
        """The cell's headline metric, if it produced one."""
        if self.manifest is None or not self.manifest.metrics:
            return None
        return self.manifest.primary.value


@dataclass(frozen=True, slots=True)
class SweepResult:
    """The outcome of a whole sweep."""

    plan: SweepPlan
    outcomes: tuple[CellOutcome, ...] = ()
    duration_s: float = 0.0
    sweep_dir: Path = field(default_factory=lambda: Path("runs"))

    def counts(self) -> dict[str, int]:
        """How many cells ran, were reused, or failed."""
        tally: dict[str, int] = {"ran": 0, "cached": 0, "failed": 0}
        for outcome in self.outcomes:
            tally[outcome.status] = tally.get(outcome.status, 0) + 1
        return tally

    def ranked(self) -> tuple[CellOutcome, ...]:
        """Cells with results, best first."""
        scored = [item for item in self.outcomes if item.score is not None]
        if not scored:
            return ()
        greater_is_better = True
        first = scored[0].manifest
        if first is not None and first.metrics:
            greater_is_better = first.primary.greater_is_better
        return tuple(
            sorted(
                scored,
                key=lambda item: item.score or 0.0,
                reverse=greater_is_better,
            )
        )


def run_sweep(
    plan: SweepPlan,
    *,
    sweep_dir: Path | None = None,
    force: bool = False,
    progress: Any = None,
) -> SweepResult:
    """Execute every cell of a sweep, reusing recorded results where valid.

    Parameters
    ----------
    plan
        The grid to execute.
    sweep_dir
        Where cell manifests are written; defaults to ``<runs_dir>/sweep``.
    force
        Re-run every cell, ignoring any recorded result.
    progress
        Optional callable invoked as ``progress(index, total, cell, status)``
        after each cell.

    Returns
    -------
    SweepResult
        One outcome per cell, in grid order.

    Raises
    ------
    RunError
        If the plan's data cannot be read.
    """
    from michi.bench import run_benchmark
    from michi.recipes import load_recipe

    started = time.perf_counter()
    data_path = Path(plan.data)
    if not data_path.exists():
        msg = f"sweep data not found: {data_path}"
        raise RunError(msg)

    destination = sweep_dir or (Path(plan.runs_dir) / "sweep")
    destination.mkdir(parents=True, exist_ok=True)

    table = load_table(data_path)
    data_sha = table.source.sha256
    recipe_shas = {
        path: hash_file(Path(path)) for path in plan.recipes if Path(path).exists()
    }
    recipe_cache: dict[str, Any] = {}
    recorded = _recorded_keys(destination) if not force else {}

    cells = plan.cells()
    outcomes: list[CellOutcome] = []
    for index, cell in enumerate(cells, start=1):
        key = cell.key(
            data_sha=data_sha,
            folds=plan.folds,
            recipe_sha=recipe_shas.get(cell.recipe or "", ""),
        )
        if key in recorded:
            outcome = CellOutcome(cell=cell, status="cached", manifest=recorded[key])
            outcomes.append(outcome)
            _report(progress, index, len(cells), cell, "cached")
            continue

        cell_started = time.perf_counter()
        try:
            recipe = None
            if cell.recipe is not None:
                if cell.recipe not in recipe_cache:
                    recipe_cache[cell.recipe] = load_recipe(Path(cell.recipe))
                recipe = recipe_cache[cell.recipe]

            result = run_benchmark(
                table,
                target=plan.target,
                models=(cell.model,),
                task=plan.task,
                folds=plan.folds,
                recipe=recipe,
                seed=cell.seed,
                group_id=f"sweep-{key}",
            )
            manifest = _cell_manifest(result, cell, key, plan)
            _write(manifest, destination / f"{key}.json")
            outcomes.append(
                CellOutcome(
                    cell=cell,
                    status="ran",
                    manifest=manifest,
                    seconds=time.perf_counter() - cell_started,
                )
            )
            _report(progress, index, len(cells), cell, "ran")
        except Exception as err:  # third-party failure boundary
            # One cell failing must never cost the cells already completed.
            outcomes.append(
                CellOutcome(
                    cell=cell,
                    status="failed",
                    error=str(err).splitlines()[0][:200],
                    seconds=time.perf_counter() - cell_started,
                )
            )
            _report(progress, index, len(cells), cell, "failed")

    return SweepResult(
        plan=plan,
        outcomes=tuple(outcomes),
        duration_s=time.perf_counter() - started,
        sweep_dir=destination,
    )


def _cell_manifest(
    result: Any, cell: SweepCell, key: str, plan: SweepPlan
) -> RunManifest:
    """Take the model's manifest out of a one-model benchmark and tag it."""
    from dataclasses import replace

    manifests = [
        item for item in result.manifests if item.model.reference == cell.model
    ]
    if not manifests:
        msg = f"cell {cell.label} produced no result"
        raise RunError(msg)

    manifest: RunManifest = manifests[0]
    details = dict(manifest.details)
    details.update(
        {
            "sweep_key": key,
            "sweep_cell": cell.to_dict(),
            "sweep_plan": plan.to_dict(),
        }
    )
    tagged: RunManifest = replace(
        manifest, kind="sweep", details=details, run_id=f"sweep-{key}"
    )
    return tagged


def _recorded_keys(directory: Path) -> dict[str, RunManifest]:
    """Load previously recorded cells, keyed by their content hash."""
    recorded: dict[str, RunManifest] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifest = RunManifest.from_dict(payload)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        key = manifest.details.get("sweep_key")
        if isinstance(key, str):
            recorded[key] = manifest
    return recorded


def _write(manifest: RunManifest, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _report(
    progress: Any, index: int, total: int, cell: SweepCell, status: str
) -> None:
    if callable(progress):
        progress(index, total, cell, status)
