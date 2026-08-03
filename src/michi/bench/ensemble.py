"""Combining models the user chose, scored against the best of them.

Design Principles
-----------------
- **The comparison is still the product.** Every tool will report an
  ensemble's score. What matters is whether it beat the best single model by
  more than the noise, so an ensemble is cross-validated beside its own
  members and tested with the same corrected resampled *t*-test as anything
  else. An ensemble that ties its best member is reported as a tie.
- **Stacking leaks unless the meta-learner is fed out-of-fold predictions.**
  scikit-learn's stacking estimators cross-validate internally to build the
  meta-features, and michi wraps the whole thing in its own outer folds on top
  — so the reported number comes from data neither layer was fitted on.
- **The user picks the members.** michi does not search for a good
  combination, prune weak members, or weight by validation score. Those are
  modelling judgements, and a tool that makes them silently is an AutoML
  system wearing a different hat.
- **Cost is stated, not hidden.** An ensemble of five models trains five
  models per fold plus a meta-learner. The fit time is reported per member so
  the trade is visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from michi.core.errors import RunError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from michi.bench.preprocess import PreparationPolicy
    from michi.bench.runner import BenchResult
    from michi.core.io import LoadedTable
    from michi.recipes.model import Recipe

__all__ = ["ENSEMBLE_NAME", "METHODS", "run_ensemble"]

METHODS: tuple[str, ...] = ("stack", "vote")
"""How members can be combined."""

ENSEMBLE_NAME = "ensemble"
"""Name the combination appears under in the leaderboard."""

_MIN_MEMBERS = 2


def run_ensemble(
    table: LoadedTable,
    *,
    target: str,
    members: tuple[str, ...],
    method: str = "stack",
    final: str = "linear",
    task: str | None = None,
    folds: int = 5,
    policy: PreparationPolicy | None = None,
    recipe: Recipe | None = None,
    seed: int = 0,
    group: str | None = None,
    group_id: str | None = None,
) -> BenchResult:
    """Cross-validate an ensemble alongside the models it combines.

    Parameters
    ----------
    table
        The dataset, including its provenance.
    target
        Name of the label column.
    members
        Catalogue models to combine. At least two.
    method
        ``"stack"`` trains a meta-learner on out-of-fold member predictions;
        ``"vote"`` averages them.
    final
        Catalogue model used as the stacking meta-learner.
    folds, policy, recipe, seed, group_id
        As for :func:`~michi.bench.runner.run_benchmark`.

    Returns
    -------
    BenchResult
        The ensemble and every member, ranked and compared — so the result
        renders, reports, and records exactly like a benchmark, because it is
        one.

    Raises
    ------
    RunError
        If fewer than two members are named, or the method is unknown.
    """
    from michi.bench.runner import run_benchmark

    if method not in METHODS:
        known = ", ".join(METHODS)
        msg = f"unknown ensemble method {method!r}; michi offers: {known}"
        raise RunError(msg)

    unique = tuple(dict.fromkeys(members))
    if len(unique) < _MIN_MEMBERS:
        msg = (
            f"an ensemble needs at least {_MIN_MEMBERS} distinct models; "
            f"got {', '.join(unique) or 'none'}"
        )
        raise RunError(msg)

    # The ensemble is registered as an ordinary catalogue entry for the
    # duration of the run, so `run_benchmark` compares it to its own members
    # with the same folds, the same baseline, and the same significance test.
    # Nothing downstream — rendering, manifests, reports — needs to know an
    # ensemble is different, because for its purposes it is not.
    from michi.bench.registry import register_transient

    with register_transient(
        ENSEMBLE_NAME,
        summary=_summary(method, unique, final),
        factory=_factory(method, unique, final),
        needs_scaling=True,
    ):
        return run_benchmark(
            table,
            target=target,
            models=(ENSEMBLE_NAME, *unique),
            task=task,
            folds=folds,
            policy=policy,
            recipe=recipe,
            seed=seed,
            group=group,
            group_id=group_id,
        )


def _summary(method: str, members: tuple[str, ...], final: str) -> str:
    """One factual line, as every catalogue entry carries."""
    joined = " + ".join(members)
    if method == "vote":
        return f"soft-vote average of {joined}"
    return f"{joined} stacked under {final}"


def _factory(method: str, members: tuple[str, ...], final: str) -> Any:
    """Build the callable the registry will use to instantiate the ensemble."""

    def build(task: str, seed: int) -> Any:
        from michi.bench.registry import build_model, model_entry

        # Each member carries its own preparation, because scaling is right
        # for a linear model and pointless for a tree. Sharing one pipeline
        # across members would quietly impose one model's needs on all of them.
        estimators = [
            (
                name,
                _prepared(
                    build_model(name, task, seed),
                    needs_scaling=model_entry(name).needs_scaling,
                ),
            )
            for name in members
        ]

        if method == "vote":
            return _voting(estimators, task=task)
        return _stacking(estimators, task=task, final=build_model(final, task, seed))

    return build


def _prepared(estimator: Any, *, needs_scaling: bool) -> Any:
    """Give one member the standardisation it needs, and no more."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not needs_scaling:
        return estimator
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _voting(estimators: list[tuple[str, Any]], *, task: str) -> Any:
    """Average the members' outputs."""
    from sklearn.ensemble import VotingClassifier, VotingRegressor

    if task == "regression":
        return VotingRegressor(estimators)
    # Soft voting averages probabilities rather than counting votes, which
    # keeps the confidence a model expresses instead of discarding it. It
    # needs predict_proba; hard voting is the fallback when a member lacks it.
    if all(_has_proba(estimator) for _, estimator in estimators):
        return VotingClassifier(estimators, voting="soft")
    return VotingClassifier(estimators, voting="hard")


def _stacking(estimators: list[tuple[str, Any]], *, task: str, final: Any) -> Any:
    """Train a meta-learner on out-of-fold member predictions."""
    from sklearn.ensemble import StackingClassifier, StackingRegressor

    # cv=5 here is the *inner* split that produces the meta-features. Without
    # it the meta-learner would train on predictions the members made about
    # rows they had already seen, which is the leak stacking is famous for.
    if task == "regression":
        return StackingRegressor(estimators, final_estimator=final, cv=5)
    return StackingClassifier(estimators, final_estimator=final, cv=5)


def _has_proba(estimator: Any) -> bool:
    """Whether a member can express confidence, not only a label."""
    target = estimator
    steps = getattr(estimator, "steps", None)
    if steps:
        target = steps[-1][1]
    return callable(getattr(target, "predict_proba", None))
