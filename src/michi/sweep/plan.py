"""The sweep plan: a grid of experiments, declared in a file.

Design Principles
-----------------
- **The grid is a file, not a script.** A sweep you can read, diff, and put in
  a paper's appendix is worth more than a loop nobody else can run.
- **Every cell is identified by its inputs.** A cell's key is a hash of the
  data, recipe, model, seed, and folds that produce it, which is what makes
  caching and resumption correct rather than approximate.
- **Cells are enumerated up front.** Knowing the full grid before starting
  means progress is honest and a resumed sweep knows exactly what remains.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from michi.core.errors import RunError
from michi.core.hashing import hash_payload

__all__ = ["SWEEP_SCHEMA_VERSION", "SweepCell", "SweepPlan", "load_plan"]

SWEEP_SCHEMA_VERSION = "1.0"
"""Schema version of the sweep plan file."""


@dataclass(frozen=True, slots=True)
class SweepCell:
    """One point in the grid: a model, a recipe, and a seed."""

    model: str
    seed: int
    recipe: str | None = None

    @property
    def label(self) -> str:
        """A short human label for progress output."""
        recipe = Path(self.recipe).stem if self.recipe else "none"
        return f"{self.model} · {recipe} · seed {self.seed}"

    def key(self, *, data_sha: str, folds: int, recipe_sha: str = "") -> str:
        """A content hash of everything that determines this cell's result.

        Two cells with the same key must produce the same numbers, which is
        what lets a resumed sweep skip work without ever reusing a stale
        result.
        """
        return hash_payload(
            {
                "data": data_sha,
                "recipe": recipe_sha,
                "model": self.model,
                "seed": self.seed,
                "folds": folds,
            }
        )[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialise for inclusion in a run manifest."""
        return {"model": self.model, "seed": self.seed, "recipe": self.recipe}


@dataclass(frozen=True, slots=True)
class SweepPlan:
    """A declarative grid of experiments.

    Examples
    --------
    >>> plan = SweepPlan(
    ...     data="train.csv", target="y", models=("rf", "linear"), seeds=(0, 1)
    ... )
    >>> len(plan.cells())
    4
    """

    data: str
    target: str
    models: tuple[str, ...] = ()
    recipes: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (0,)
    folds: int = 5
    task: str | None = None
    runs_dir: str = "runs"
    source: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.data:
            msg = "a sweep needs a 'data' path"
            raise RunError(msg)
        if not self.target:
            msg = "a sweep needs a 'target' column"
            raise RunError(msg)
        if not self.models:
            msg = "a sweep needs at least one model in 'grid.models'"
            raise RunError(msg)
        if not self.seeds:
            msg = "a sweep needs at least one seed"
            raise RunError(msg)

    def cells(self) -> tuple[SweepCell, ...]:
        """Enumerate every cell, in a stable order."""
        recipes: Sequence[str | None] = self.recipes or (None,)
        return tuple(
            SweepCell(model=model, seed=seed, recipe=recipe)
            for recipe in recipes
            for model in self.models
            for seed in seed_list(self.seeds)
        )

    @property
    def size(self) -> int:
        """How many cells the grid contains."""
        return len(self.cells())

    def to_dict(self) -> dict[str, Any]:
        """Serialise the plan, for recording alongside results."""
        return {
            "schema_version": SWEEP_SCHEMA_VERSION,
            "data": self.data,
            "target": self.target,
            "task": self.task,
            "folds": self.folds,
            "runs_dir": self.runs_dir,
            "grid": {
                "models": list(self.models),
                "recipes": list(self.recipes),
                "seeds": list(self.seeds),
            },
        }


def seed_list(seeds: Sequence[int]) -> tuple[int, ...]:
    """Normalise seeds to a tuple of integers."""
    return tuple(int(seed) for seed in seeds)


def load_plan(source: Path) -> SweepPlan:
    """Read a sweep plan from a YAML file.

    Raises
    ------
    RunError
        If the file is missing, unparseable, or incomplete — with a message
        naming what is wrong, because this is a file a human typed.
    """
    import yaml

    if not source.exists():
        msg = f"no such sweep file: {source}"
        raise RunError(msg)

    try:
        payload: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        msg = f"could not parse {source.name} as YAML: {err}"
        raise RunError(msg) from err
    except OSError as err:
        msg = f"could not read {source}: {err}"
        raise RunError(msg) from err

    if not isinstance(payload, dict):
        msg = f"{source.name} must contain a mapping at the top level"
        raise RunError(msg)

    grid = payload.get("grid") or {}
    if not isinstance(grid, dict):
        msg = f"{source.name}: 'grid' must be a mapping of models, recipes, seeds"
        raise RunError(msg)

    base = source.parent

    def _resolve(value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or candidate.exists():
            return str(candidate)
        # Paths in a sweep file are relative to the file, so a sweep can be
        # run from anywhere.
        return str(base / candidate)

    return SweepPlan(
        data=_resolve(str(payload.get("data", ""))) if payload.get("data") else "",
        target=str(payload.get("target", "")),
        models=tuple(str(item) for item in grid.get("models", ())),
        recipes=tuple(_resolve(str(item)) for item in grid.get("recipes", ())),
        seeds=tuple(int(item) for item in grid.get("seeds", (0,))) or (0,),
        folds=int(payload.get("folds") or payload.get("cv") or 5),
        task=None if payload.get("task") is None else str(payload["task"]),
        runs_dir=str(payload.get("runs_dir", "runs")),
        source=source,
    )
