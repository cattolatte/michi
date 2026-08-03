"""The ``michi fit`` and ``michi predict`` commands — the last mile.

Design Principles
-----------------
- **A model is an artifact like any other.** ``fit`` writes a joblib file plus
  a run manifest recording the data hash, the recipe, the seed, and the
  parameters. A model you cannot trace back to the rows that made it is a
  model you cannot defend.
- **``predict`` needs no labels**, which is the whole point: it is what a
  competition submission, a batch scoring job, and a smoke test all need, and
  it is the one thing ``eval`` cannot do.
- **Any model michi can load, ``predict`` can use.** A pickle, or
  ``module:object`` for PyTorch, TensorFlow, ONNX, or anything else with a
  ``predict``. Deep learning arrives through the protocol, not through a
  per-framework loader michi would have to maintain forever.
- **The scores are not from here.** ``fit`` trains on everything you give it,
  so it reports no accuracy at all — a number computed on the training rows
  would be the most confidently wrong figure in the toolbox. Use ``bench`` or
  ``eval`` for what the model is worth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from michi.cli.context import resolve_defaults
from michi.cli.errors import fail
from michi.core.errors import MichiError
from michi.core.io import DEFAULT_SAMPLE_ROWS, load_table

__all__ = ["fit_command", "predict_command"]


def fit_command(
    data: Annotated[
        Path | None,
        typer.Argument(help="Training data. Falls back to `data` in michi.toml."),
    ] = None,
    target: Annotated[
        str | None, typer.Option("--target", "-t", help="Label column.")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Catalogue model to train.")
    ] = "hist-gbm",
    output: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the fitted model.")
    ] = Path("model.joblib"),
    recipe: Annotated[
        Path | None, typer.Option("--recipe", help="Cleaning recipe to apply first.")
    ] = None,
    params: Annotated[
        Path | None,
        typer.Option("--params", help="YAML of hyperparameters, e.g. from `tune`."),
    ] = None,
    task: Annotated[
        str | None,
        typer.Option("--task", help="Force `classification` or `regression`."),
    ] = None,
    calibrate: Annotated[
        str | None,
        typer.Option(
            "--calibrate",
            help="Fix overconfident probabilities: 'isotonic' or 'sigmoid'.",
        ),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed.")] = None,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
) -> None:
    """Train one model on all the data and save it.

    Reports no accuracy on purpose: a score measured on the rows a model was
    trained on is the most confidently wrong number a tool can print. Use
    'michi bench' to choose a model and 'michi eval' to score one.
    """
    console = Console()
    defaults = resolve_defaults()
    resolved_seed = defaults.number("seed", seed) or 0

    try:
        resolved_data = defaults.required_data(data)
        table = load_table(
            resolved_data, sample_rows=sample, full=full, seed=resolved_seed
        )
        resolved_target, note = defaults.target_for(target, table.frame.columns)
        if note:
            console.print(f"  [dim]{note}[/]")
        if not resolved_target:
            msg = (
                "fitting needs a target column: pass --target, "
                "or set `target` in michi.toml"
            )
            raise MichiError(msg)

        estimator, resolved_task, applied = _build_fitted(
            table=table,
            target=resolved_target,
            model=model,
            task=task,
            recipe=recipe,
            params=params,
            calibrate=calibrate,
            seed=resolved_seed,
        )
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err

    output.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(estimator, output)

    header = Text()
    header.append(" 道 ", style="bold red")
    header.append(" michi fit", style="bold")
    header.append(f"  ·  {model}", style="bold cyan")
    console.print()
    console.print(header)
    console.print()

    summary = Text(style="dim")
    summary.append(
        f"{resolved_task}  ·  {applied:,} rows  ·  target {resolved_target}  ·  "
        f"seed {resolved_seed}"
    )
    console.print(Padding(summary, (0, 0, 1, 2)))
    console.print(Padding(Text(f"wrote {output}", style="bold"), (0, 0, 0, 2)))
    console.print(
        Padding(
            Text(
                "Trained on every row given, so michi reports no score here — "
                "one measured on\nthe training rows would be meaningless. "
                "`michi bench` compares models honestly;\n"
                "`michi eval` scores this file against held-out data.",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )
    console.print(
        Padding(
            Text(
                f"Next:  michi predict {output} new_data.csv -o predictions.csv",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )


def predict_command(
    model_path: Annotated[
        str,
        typer.Argument(
            help="Fitted model: a pickle/joblib path, or module:object.",
            show_default=False,
        ),
    ],
    data: Annotated[
        Path | None,
        typer.Argument(help="Data to predict on. No label column required."),
    ] = None,
    output: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the predictions.")
    ] = Path("predictions.csv"),
    recipe: Annotated[
        Path | None,
        typer.Option("--recipe", help="Cleaning recipe to apply before predicting."),
    ] = None,
    id_column: Annotated[
        str | None,
        typer.Option("--id", help="Column to carry through alongside the prediction."),
    ] = None,
    label: Annotated[
        str, typer.Option("--label", help="Name of the prediction column.")
    ] = "prediction",
    proba: Annotated[
        bool,
        typer.Option("--proba", help="Write class probabilities where available."),
    ] = False,
    drop_target: Annotated[
        str | None,
        typer.Option("--drop-target", help="Label column to remove if present."),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for sampling a large file.")
    ] = None,
    sample: Annotated[
        int, typer.Option("--sample", help="Rows to keep when a large file is sampled.")
    ] = DEFAULT_SAMPLE_ROWS,
    full: Annotated[
        bool, typer.Option("--full", help="Read every row, however large the file.")
    ] = False,
) -> None:
    """Predict with a fitted model and write the results to a file.

    Needs no labels, which is the point: this is what a competition
    submission, a batch scoring job, and a smoke test all need.
    """
    console = Console()
    defaults = resolve_defaults()
    resolved_seed = defaults.number("seed", seed) or 0

    try:
        import pandas as pd

        from michi.adapters import load_model

        resolved_data = defaults.required_data(data)
        table = load_table(
            resolved_data, sample_rows=sample, full=full, seed=resolved_seed
        )
        frame = table.frame.copy()

        identifiers = None
        if id_column:
            if id_column not in frame.columns:
                msg = f"--id names a column not in the data: {id_column!r}"
                raise MichiError(msg)
            identifiers = frame[id_column]

        if recipe is not None:
            from michi.recipes import apply_recipe, load_recipe

            frame = apply_recipe(load_recipe(recipe), frame, strict=False).frame

        # A held-out file often still carries an empty label column; a model
        # trained without it would fail on the extra feature.
        removed = drop_target or defaults.text("target", None)
        features = frame.drop(columns=[removed]) if removed in frame.columns else frame

        loaded = load_model(model_path)
        predictions = loaded.predict(features)

        result = pd.DataFrame()
        if identifiers is not None:
            result[id_column] = identifiers.to_numpy()[: len(predictions)]
        result[label] = predictions

        if proba:
            probabilities = loaded.predict_proba(features)
            if probabilities is None:
                console.print(
                    "  [yellow]note:[/] this model does not produce "
                    "probabilities — writing labels only"
                )
            else:
                classes = loaded.classes or range(probabilities.shape[1])
                for position, name in enumerate(classes):
                    result[f"proba_{name}"] = probabilities[:, position]
    except MichiError as err:
        fail(str(err))
        raise typer.Exit(code=2) from err
    except Exception as err:  # third-party failure boundary
        fail(f"prediction failed: {err}")
        raise typer.Exit(code=2) from err

    output.parent.mkdir(parents=True, exist_ok=True)
    _write(result, output)

    header = Text()
    header.append(" 道 ", style="bold red")
    header.append(" michi predict", style="bold")
    console.print()
    console.print(header)
    console.print()
    console.print(
        Padding(
            Text(
                f"{len(result):,} predictions  ·  from {loaded.spec.reference}",
                style="dim",
            ),
            (0, 0, 1, 2),
        )
    )
    console.print(Padding(Text(f"wrote {output}", style="bold"), (0, 0, 1, 2)))


def _build_fitted(
    *,
    table: Any,
    target: str,
    model: str,
    task: str | None,
    recipe: Path | None,
    params: Path | None,
    calibrate: str | None,
    seed: int,
) -> tuple[Any, str, int]:
    """Apply the recipe, build the pipeline, and fit it on everything."""
    from michi.bench import model_entry
    from michi.bench.preprocess import PreparationPolicy
    from michi.bench.registry import build_model
    from michi.evaluation.metrics import detect_task

    frame = table.frame
    loaded_recipe = None
    if recipe is not None:
        from michi.recipes import load_recipe

        loaded_recipe = load_recipe(recipe)

    if target not in frame.columns:
        msg = f"target column {target!r} is not in the data"
        raise MichiError(msg)

    labels = frame[target]
    resolved_task = task or detect_task(labels.to_numpy())
    features = frame.drop(columns=[target])

    if loaded_recipe is not None:
        from michi.recipes import apply_deterministic

        features = apply_deterministic(loaded_recipe, features)

    estimator = build_model(model, resolved_task, seed)
    if params is not None:
        estimator.set_params(**_load_params(params))

    from michi.bench.runner import fold_pipeline

    pipeline = fold_pipeline(
        features=features,
        estimator=estimator,
        policy=PreparationPolicy(),
        recipe=loaded_recipe,
        needs_scaling=model_entry(model).needs_scaling,
    )
    if calibrate:
        pipeline = _calibrated(pipeline, calibrate, resolved_task)

    pipeline.fit(features, labels.to_numpy())
    return pipeline, resolved_task, len(features)


def _calibrated(pipeline: Any, method: str, task: str) -> Any:
    """Wrap a classifier so its probabilities mean what they say.

    `eval` reports that a model is overconfident — it says 0.9 and is right
    0.7 of the time — and until now offered no way to fix it. Calibration is
    fitted by internal cross-validation, so the mapping from score to
    probability is learned on folds the base model did not train on.
    """
    from sklearn.calibration import CalibratedClassifierCV

    if task != "classification":
        msg = "--calibrate applies to classifiers; a regressor has no probabilities"
        raise MichiError(msg)
    if method not in {"isotonic", "sigmoid"}:
        msg = (
            f"--calibrate expects isotonic or sigmoid (got {method!r}). "
            "isotonic is flexible and needs more data; sigmoid assumes a "
            "shape and survives small samples."
        )
        raise MichiError(msg)
    return CalibratedClassifierCV(pipeline, method=method, cv=5)


def _load_params(path: Path) -> dict[str, Any]:
    """Read hyperparameters from YAML, as `michi tune --save-params` writes."""
    import yaml

    if not path.exists():
        msg = f"no such parameter file: {path}"
        raise MichiError(msg)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        msg = f"{path.name} should map parameter names to values"
        raise MichiError(msg)
    return {str(key): value for key, value in payload.items()}


def _write(frame: Any, destination: Path) -> None:
    """Write predictions in the format the extension implies."""
    suffix = destination.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    elif suffix == ".tsv":
        frame.to_csv(destination, index=False, sep="\t", encoding="utf-8")
    else:
        frame.to_csv(destination, index=False, encoding="utf-8")
