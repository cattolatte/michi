"""The recipe artifact: cleaning decisions as a file you own.

Design Principles
-----------------
- **The artifact is the product.** An interactive session that changes data
  and forgets what it did is worthless the next day. ``michi clean`` writes a
  recipe; the recipe is what gets reviewed, versioned, and re-run.
- **Declarative, not procedural.** A recipe lists *what* should happen, in
  order. How it happens belongs to ``apply`` (pandas) and ``export``
  (generated sklearn code), which must agree.
- **It carries the schema it was written against.** That snapshot turns the
  recipe into a data contract: applying it to a file with different columns
  fails loudly instead of silently doing the wrong thing.
- **Every step records why.** The reason a user dropped a column is the part
  that is impossible to reconstruct six months later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from michi import __version__
from michi.core.artifacts import utc_now_iso
from michi.core.errors import RecipeError

__all__ = ["RECIPE_SCHEMA_VERSION", "Recipe", "RecipeStep", "SourceSchema"]

RECIPE_SCHEMA_VERSION = "1.0"
"""Schema version of the recipe artifact. Frozen under semver at michi 1.0."""

_KNOWN_OPS: dict[str, frozenset[str]] = {
    # Cleaning.
    "drop": frozenset({"columns"}),
    "dedupe": frozenset({"subset"}),
    "cast": frozenset({"columns", "to"}),
    "impute": frozenset({"columns", "strategy", "value"}),
    "clip": frozenset({"columns", "lower_quantile", "upper_quantile"}),
    "encode": frozenset({"columns", "method"}),
    "scale": frozenset({"columns", "method"}),
    # Feature engineering. Added in michi 1.1; the schema shape is unchanged,
    # so a 1.0 recipe still loads and a 1.0 michi rejects these by name rather
    # than misreading them.
    "datepart": frozenset({"columns", "parts"}),
    "log": frozenset({"columns", "method"}),
    "interact": frozenset({"columns", "method"}),
    "binarize": frozenset({"columns", "threshold"}),
    "bin": frozenset({"columns", "bins", "strategy"}),
}


@dataclass(frozen=True, slots=True)
class RecipeStep:
    """One declarative cleaning or preparation operation.

    Parameters
    ----------
    op
        Operation name. Cleaning: ``drop``, ``dedupe``, ``cast``, ``impute``,
        ``clip``, ``encode``, ``scale``. Feature engineering: ``datepart``,
        ``log``, ``interact``, ``binarize``, ``bin``.
    params
        Operation-specific parameters.
    why
        The user's reason, carried into the emitted YAML as a comment. This is
        the part that cannot be reconstructed later.

    Examples
    --------
    >>> step = RecipeStep("impute", {"columns": ["age"], "strategy": "median"})
    >>> step.columns
    ('age',)
    """

    op: str
    params: Mapping[str, Any] = field(default_factory=dict)
    why: str = ""

    def __post_init__(self) -> None:
        if self.op not in _KNOWN_OPS:
            known = ", ".join(sorted(_KNOWN_OPS))
            msg = f"unknown recipe operation {self.op!r}; known operations: {known}"
            raise RecipeError(msg)
        unexpected = set(self.params) - _KNOWN_OPS[self.op]
        if unexpected:
            allowed = ", ".join(sorted(_KNOWN_OPS[self.op])) or "none"
            msg = (
                f"operation {self.op!r} does not take "
                f"{', '.join(sorted(unexpected))}; it takes: {allowed}"
            )
            raise RecipeError(msg)

    @property
    def columns(self) -> tuple[str, ...]:
        """Columns this step names, empty when it applies to whole rows."""
        columns = self.params.get("columns") or self.params.get("subset") or ()
        return tuple(str(name) for name in columns)

    @property
    def is_fitted(self) -> bool:
        """Whether this step learns something from the data it sees.

        Fitted steps — imputation, encoding, scaling, quantile binning —
        must be fitted on training data only. Deterministic steps depend only
        on the row in front of them and can be applied anywhere without risk
        of leakage.

        The distinction is not cosmetic. Quantile bin edges learned from the
        whole dataset have already seen the test fold's distribution.
        Classifying a step here is what puts it inside the cross-validation
        fold in :func:`~michi.recipes.pipeline.build_transformer` and inside
        ``build_pipeline()`` in exported code.

        Examples
        --------
        >>> RecipeStep("log", {"columns": ["fare"]}).is_fitted
        False
        >>> RecipeStep("bin", {"columns": ["age"]}).is_fitted
        True
        """
        return self.op in {"impute", "encode", "scale", "bin"}

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON/YAML-compatible dictionary."""
        payload: dict[str, Any] = {"op": self.op, **dict(self.params)}
        if self.why:
            payload["why"] = self.why
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a step from :meth:`to_dict` output."""
        if "op" not in payload:
            msg = f"recipe step is missing 'op': {dict(payload)!r}"
            raise RecipeError(msg)
        params = {
            key: value for key, value in payload.items() if key not in {"op", "why"}
        }
        return cls(
            op=str(payload["op"]),
            params=params,
            why=str(payload.get("why", "")),
        )


@dataclass(frozen=True, slots=True)
class SourceSchema:
    """The shape of the data a recipe was authored against.

    Applying a recipe to data with a different shape is the moment a silent
    mistake becomes possible, so the shape travels with the recipe.
    """

    path: str = ""
    sha256: str = ""
    n_rows: int = 0
    columns: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON/YAML-compatible dictionary."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_rows": self.n_rows,
            "columns": dict(self.columns),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a schema snapshot from :meth:`to_dict` output."""
        return cls(
            path=str(payload.get("path", "")),
            sha256=str(payload.get("sha256", "")),
            n_rows=int(payload.get("n_rows", 0)),
            columns={
                str(key): str(value)
                for key, value in (payload.get("columns") or {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class Recipe:
    """An ordered list of cleaning decisions, plus the schema behind them.

    Examples
    --------
    >>> recipe = Recipe(steps=(RecipeStep("drop", {"columns": ["id"]}),))
    >>> Recipe.from_dict(recipe.to_dict()).steps[0].op
    'drop'
    """

    steps: tuple[RecipeStep, ...] = ()
    target: str | None = None
    source: SourceSchema = field(default_factory=SourceSchema)
    schema_version: str = RECIPE_SCHEMA_VERSION
    michi_version: str = __version__
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def fitted_steps(self) -> tuple[RecipeStep, ...]:
        """Steps that learn from data, and so must be fitted on training rows."""
        return tuple(step for step in self.steps if step.is_fitted)

    @property
    def deterministic_steps(self) -> tuple[RecipeStep, ...]:
        """Steps that can be applied anywhere without risk of leakage."""
        return tuple(step for step in self.steps if not step.is_fitted)

    def dropped_columns(self) -> tuple[str, ...]:
        """Every column this recipe removes."""
        dropped: list[str] = []
        for step in self.steps:
            if step.op == "drop":
                dropped.extend(step.columns)
        return tuple(dropped)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole artifact to a JSON/YAML-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "michi_version": self.michi_version,
            "created_at": self.created_at,
            "target": self.target,
            "source": self.source.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a recipe from :meth:`to_dict` output.

        Raises
        ------
        RecipeError
            If the payload is not a mapping with a ``steps`` list.
        """
        if not isinstance(payload, Mapping):
            msg = "a recipe file must contain a mapping at the top level"
            raise RecipeError(msg)

        steps_payload = payload.get("steps")
        if steps_payload is None:
            msg = "recipe is missing a 'steps' list"
            raise RecipeError(msg)
        if not isinstance(steps_payload, Sequence) or isinstance(steps_payload, str):
            msg = "recipe 'steps' must be a list of operations"
            raise RecipeError(msg)

        target = payload.get("target")
        return cls(
            steps=tuple(RecipeStep.from_dict(item) for item in steps_payload),
            target=None if target is None else str(target),
            source=SourceSchema.from_dict(payload.get("source") or {}),
            schema_version=str(payload.get("schema_version", RECIPE_SCHEMA_VERSION)),
            michi_version=str(payload.get("michi_version", __version__)),
            created_at=str(payload.get("created_at", utc_now_iso())),
        )
