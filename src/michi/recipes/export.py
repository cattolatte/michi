"""Compiling a recipe into readable Python you own.

Design Principles
-----------------
- **The exit door is part of the product.** A user who outgrows michi should
  leave with working code, not a dependency. Exported files import pandas and
  sklearn — never michi.
- **Generated code is a teaching text that happens to run.** It is commented,
  formatted, and passes ruff and mypy, because the most common thing a user
  will do with it is read it.
- **The split is the point.** Deterministic steps become a plain function;
  fitted steps become an sklearn ``Pipeline`` that fits inside cross-validation,
  where imputers and encoders cannot see the fold they are scored on.
- ``export`` and ``apply`` must agree on what a recipe means; a test executes
  the generated code and compares.
"""

from __future__ import annotations

from michi.recipes.model import Recipe, RecipeStep

__all__ = ["export_recipe"]


def export_recipe(recipe: Recipe, *, module_name: str = "pipeline") -> str:
    """Compile a recipe into a standalone Python module.

    Parameters
    ----------
    recipe
        The recipe to compile.
    module_name
        Name used in the module docstring's usage example.

    Returns
    -------
    str
        Python source: a ``prepare`` function for deterministic steps and a
        ``build_pipeline`` factory for the fitted ones.
    """
    deterministic = recipe.deterministic_steps
    fitted = recipe.fitted_steps

    lines: list[str] = []
    lines.extend(_module_header(recipe, module_name, deterministic, fitted))
    lines.extend(_prepare_function(deterministic))
    lines.extend(_pipeline_function(fitted))
    lines.extend(_usage_example(recipe, module_name))
    return "\n".join(lines).rstrip() + "\n"


def _module_header(
    recipe: Recipe,
    module_name: str,
    deterministic: tuple[RecipeStep, ...],
    fitted: tuple[RecipeStep, ...],
) -> list[str]:
    source = recipe.source.path or "your data"
    return [
        '"""Data preparation, compiled from a michi recipe.',
        "",
        f"Authored against {source} on {recipe.created_at[:10]}.",
        "",
        "This file is yours. It imports pandas and scikit-learn, and nothing",
        "else — michi is not a runtime dependency of the code it writes.",
        "",
        "Two pieces, because they carry different risks:",
        "",
        f"* ``prepare`` applies the {len(deterministic)} deterministic step(s):",
        "  dropping, deduplicating, casting, clipping. These depend only on the",
        "  row in front of them, so they are safe to run on any data at any time.",
        "",
        f"* ``build_pipeline`` returns an sklearn Pipeline for the {len(fitted)}",
        "  fitted step(s): imputation, encoding, scaling. These *learn* from the",
        "  data they see, so they belong inside cross-validation, where they",
        "  cannot observe the fold they will be scored on.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import pandas as pd",
        *_sklearn_imports(fitted),
        "",
        "",
    ]


def _sklearn_imports(fitted: tuple[RecipeStep, ...]) -> list[str]:
    if not fitted:
        return []
    needed: set[str] = set()
    for step in fitted:
        if step.op == "impute":
            needed.add("from sklearn.impute import SimpleImputer")
        elif step.op == "encode":
            method = str(step.params.get("method", "onehot"))
            needed.add(
                "from sklearn.preprocessing import OneHotEncoder"
                if method == "onehot"
                else "from sklearn.preprocessing import OrdinalEncoder"
            )
        elif step.op == "scale":
            method = str(step.params.get("method", "standard"))
            scaler = {
                "standard": "StandardScaler",
                "minmax": "MinMaxScaler",
                "robust": "RobustScaler",
            }.get(method, "StandardScaler")
            needed.add(f"from sklearn.preprocessing import {scaler}")
    needed.add("from sklearn.compose import ColumnTransformer")
    needed.add("from sklearn.pipeline import Pipeline")
    return sorted(needed)


def _prepare_function(steps: tuple[RecipeStep, ...]) -> list[str]:
    lines = [
        "def prepare(frame: pd.DataFrame) -> pd.DataFrame:",
        '    """Apply the deterministic cleaning steps.',
        "",
        "    Returns a new frame; the input is never modified.",
        '    """',
        "    frame = frame.copy()",
    ]
    if not steps:
        lines.append("    # This recipe has no deterministic steps.")
    for step in steps:
        lines.append("")
        if step.why:
            lines.append(f"    # {step.why}")
        lines.extend(_render_deterministic(step))
    lines.extend(["", "    return frame", "", ""])
    return lines


def _literal(value: object) -> str:
    """Render a Python literal with double quotes, as ruff format emits them."""
    import json

    return json.dumps(value, ensure_ascii=False)


def _render_deterministic(step: RecipeStep) -> list[str]:
    columns = _literal(list(step.columns))
    if step.op == "drop":
        return [
            f"    # Drop {len(step.columns)} column(s).",
            f'    frame = frame.drop(columns={columns}, errors="ignore")',
        ]
    if step.op == "dedupe":
        subset = f"subset={columns}" if step.columns else "subset=None"
        return [
            "    # Remove duplicate rows.",
            f"    frame = frame.drop_duplicates({subset}).reset_index(drop=True)",
        ]
    if step.op == "cast":
        target_type = str(step.params.get("to", "numeric"))
        if target_type == "numeric":
            return [
                f"    # Parse {columns} as numbers, stripping separators and",
                "    # currency marks. Unparseable values become missing.",
                f"    for column in {columns}:",
                "        cleaned = frame[column].astype(str).str.replace("
                'r"[,\\s$€£¥%_]", "", regex=True)',
                '        frame[column] = pd.to_numeric(cleaned, errors="coerce")',
            ]
        if target_type == "datetime":
            return [
                f"    # Parse {columns} as timestamps.",
                f"    for column in {columns}:",
                "        frame[column] = pd.to_datetime(",
                '            frame[column], errors="coerce", format="mixed"',
                "        )",
            ]
        return [
            f"    # Cast {columns} to {target_type}.",
            f"    for column in {columns}:",
            f'        frame[column] = frame[column].astype("{_dtype(target_type)}")',
        ]
    if step.op == "clip":
        lower = float(step.params.get("lower_quantile", 0.01))
        upper = float(step.params.get("upper_quantile", 0.99))
        return [
            f"    # Clip {columns} to the {lower:.0%}–{upper:.0%} quantile range.",
            "    # Note: the bounds come from whatever data is passed in.",
            f"    for column in {columns}:",
            '        values = pd.to_numeric(frame[column], errors="coerce")',
            "        frame[column] = values.clip(",
            f"            lower=values.quantile({lower}),",
            f"            upper=values.quantile({upper}),",
            "        )",
        ]
    return [f"    # Unsupported step: {step.op}"]


def _pipeline_function(steps: tuple[RecipeStep, ...]) -> list[str]:
    if not steps:
        return [
            "def build_pipeline() -> None:",
            '    """This recipe has no fitted steps, so there is no pipeline."""',
            "    return None",
            "",
            "",
        ]

    lines = [
        "def build_pipeline() -> ColumnTransformer:",
        '    """Build the transformer for steps that learn from data.',
        "",
        "    Fit this inside your cross-validation, never on the whole dataset:",
        "    an imputer fitted on all rows has already seen your test fold.",
        '    """',
        "    return ColumnTransformer(",
        "        transformers=[",
    ]
    for index, step in enumerate(steps):
        columns = _literal(list(step.columns))
        if step.why:
            lines.append(f"            # {step.why}")
        lines.append("            (")
        lines.append(f"                {_literal(f'{step.op}_{index}')},")
        lines.append(f"                {_transformer(step)},")
        lines.append(f"                {columns},")
        lines.append("            ),")
    lines.extend(
        [
            "        ],",
            '        remainder="passthrough",',
            "        verbose_feature_names_out=False,",
            "    )",
            "",
            "",
        ]
    )
    return lines


def _transformer(step: RecipeStep) -> str:
    if step.op == "impute":
        strategy = str(step.params.get("strategy", "median"))
        if strategy == "constant":
            value = step.params.get("value", 0)
            return f"SimpleImputer(strategy='constant', fill_value={value!r})"
        if strategy == "drop_rows":
            # Row removal cannot be expressed as a column transformer; it is
            # handled in prepare() instead.
            return '"passthrough"'
        return f'SimpleImputer(strategy="{strategy}")'
    if step.op == "encode":
        method = str(step.params.get("method", "onehot"))
        if method == "ordinal":
            return (
                "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)"
            )
        return 'OneHotEncoder(handle_unknown="ignore", sparse_output=False)'
    if step.op == "scale":
        method = str(step.params.get("method", "standard"))
        return {
            "standard": "StandardScaler()",
            "minmax": "MinMaxScaler()",
            "robust": "RobustScaler()",
        }.get(method, "StandardScaler()")
    return '"passthrough"'


def _usage_example(recipe: Recipe, module_name: str) -> list[str]:
    target = recipe.target or "target"
    lines = [
        'if __name__ == "__main__":',
        "    # Minimal end-to-end use. Replace the model with your own.",
        "    from sklearn.ensemble import RandomForestClassifier",
        "    from sklearn.model_selection import cross_val_score",
        "",
        '    raw = pd.read_csv("data.csv")',
        "    frame = prepare(raw)",
        f'    labels = frame.pop("{target}")',
        "",
    ]
    if recipe.fitted_steps:
        # Only a recipe with fitted steps has a transformer to compose, and
        # only then has the module imported Pipeline.
        lines.extend(
            [
                "    model = Pipeline(",
                "        [",
                '            ("prepare", build_pipeline()),',
                '            ("model", RandomForestClassifier(random_state=0)),',
                "        ]",
                "    )",
            ]
        )
    else:
        lines.append("    model = RandomForestClassifier(random_state=0)")
    lines.extend(
        [
            "    scores = cross_val_score(model, frame, labels, cv=5)",
            '    print(f"{scores.mean():.4f} ± {scores.std():.4f}")',
        ]
    )
    return lines


def _dtype(name: str) -> str:
    return {"category": "category", "string": "string"}.get(name, name)
