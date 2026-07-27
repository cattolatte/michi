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
    lines.extend(_encoder_class(fitted))
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
        "  dropping, casting, clipping, and the feature engineering that depends",
        "  only on the row in front of it. Safe to run on any data at any time.",
        "",
        f"* ``build_pipeline`` returns an sklearn Pipeline for the {len(fitted)}",
        "  fitted step(s): imputation, encoding, scaling, binning. These",
        "  *learn* from the data they see, so they belong inside",
        "  cross-validation, where they cannot observe the fold they will be",
        "  scored on. Quantile bin edges learned from everything have already",
        "  seen the test fold's distribution.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        *_imports(deterministic, fitted),
        "",
        "",
    ]


def _imports(
    deterministic: tuple[RecipeStep, ...], fitted: tuple[RecipeStep, ...]
) -> list[str]:
    """Emit the import block, already in the order the formatter wants.

    Generated code has to pass the same import sort michi's own source does,
    so the grouping and ordering are reproduced here rather than left to luck:
    standard library, then plain ``import`` lines, then ``from`` lines, each
    alphabetical, with names from one module collapsed onto one line.
    """
    stdlib = (
        ["from itertools import combinations"]
        if _uses("interact", deterministic)
        else []
    )

    plain = ["import pandas as pd"]
    if (
        _uses("log", deterministic)
        or _uses("interact", deterministic)
        or _uses("target-encode", fitted)
    ):
        plain.append("import numpy as np")

    modules: dict[str, set[str]] = {}
    for step in fitted:
        module, name = _transformer_import(step)
        if module:
            modules.setdefault(module, set()).add(name)
    if fitted:
        modules.setdefault("sklearn.compose", set()).add("ColumnTransformer")
        modules.setdefault("sklearn.pipeline", set()).add("Pipeline")
    for module, names in _encoder_imports(fitted).items():
        modules.setdefault(module, set()).update(names)

    grouped = [
        f"from {module} import {', '.join(sorted(names))}"
        for module, names in sorted(modules.items())
    ]

    block = (
        [*stdlib, "", *sorted(plain), *grouped]
        if stdlib
        else [*sorted(plain), *grouped]
    )
    return block


def _uses(op: str, steps: tuple[RecipeStep, ...]) -> bool:
    return any(step.op == op for step in steps)


def _transformer_import(step: RecipeStep) -> tuple[str, str]:
    """The module and name a fitted step's transformer comes from."""
    if step.op == "impute":
        return "sklearn.impute", "SimpleImputer"
    if step.op == "encode":
        method = str(step.params.get("method", "onehot"))
        name = "OneHotEncoder" if method == "onehot" else "OrdinalEncoder"
        return "sklearn.preprocessing", name
    if step.op == "scale":
        method = str(step.params.get("method", "standard"))
        name = {
            "standard": "StandardScaler",
            "minmax": "MinMaxScaler",
            "robust": "RobustScaler",
        }.get(method, "StandardScaler")
        return "sklearn.preprocessing", name
    if step.op == "bin":
        return "sklearn.preprocessing", "KBinsDiscretizer"
    return "", ""


def _encoder_imports(fitted: tuple[RecipeStep, ...]) -> dict[str, set[str]]:
    """Extra imports the emitted target encoder needs, if it is emitted."""
    if not _uses("target-encode", fitted):
        return {}
    return {
        "sklearn.base": {"BaseEstimator", "TransformerMixin"},
        "sklearn.model_selection": {"KFold"},
    }


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


_LINE_LIMIT = 88


def _literal(value: object) -> str:
    """Render a Python literal with double quotes, as ruff format emits them."""
    import json

    return json.dumps(value, ensure_ascii=False)


def _wrapped_list(
    columns: list[str], *, indent: str, prefix: str, suffix: str
) -> list[str]:
    """Emit a list literal, exploding it when one line would be too long.

    Generated code has to satisfy the same formatter michi's own source does,
    and a recipe that drops eight columns produces a line well past the limit.
    This reproduces what the formatter would have written.
    """
    single = f"{indent}{prefix}{_literal(columns)}{suffix}"
    if len(single) <= _LINE_LIMIT:
        return [single]

    lines = [f"{indent}{prefix}["]
    lines.extend(f"{indent}    {_literal(column)}," for column in columns)
    lines.append(f"{indent}]{suffix}")
    return lines


def _render_deterministic(step: RecipeStep) -> list[str]:
    columns = _literal(list(step.columns))
    if step.op == "drop":
        names = list(step.columns)
        single = f'    frame = frame.drop(columns={columns}, errors="ignore")'
        if len(single) <= _LINE_LIMIT:
            body = [single]
        else:
            body = [
                "    frame = frame.drop(",
                *_wrapped_list(names, indent="        ", prefix="columns=", suffix=","),
                '        errors="ignore",',
                "    )",
            ]
        return [f"    # Drop {len(names)} column(s).", *body]
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
                f"    # Parse {len(step.columns)} column(s) as numbers, stripping",
                "    # separators and currency marks. Unparseable values "
                "become missing.",
                *_wrapped_list(
                    list(step.columns),
                    indent="    ",
                    prefix="for column in ",
                    suffix=":",
                ),
                "        cleaned = frame[column].astype(str).str.replace("
                'r"[,\\s$€£¥%_]", "", regex=True)',
                '        frame[column] = pd.to_numeric(cleaned, errors="coerce")',
            ]
        if target_type == "datetime":
            return [
                f"    # Parse {len(step.columns)} column(s) as timestamps.",
                *_wrapped_list(
                    list(step.columns),
                    indent="    ",
                    prefix="for column in ",
                    suffix=":",
                ),
                "        frame[column] = pd.to_datetime("
                'frame[column], errors="coerce", format="mixed")',
            ]
        return [
            f"    # Cast {len(step.columns)} column(s) to {target_type}.",
            *_wrapped_list(
                list(step.columns), indent="    ", prefix="for column in ", suffix=":"
            ),
            f'        frame[column] = frame[column].astype("{_dtype(target_type)}")',
        ]
    if step.op == "clip":
        lower = float(step.params.get("lower_quantile", 0.01))
        upper = float(step.params.get("upper_quantile", 0.99))
        return [
            f"    # Clip {len(step.columns)} column(s) to the "
            f"{lower:.0%}–{upper:.0%} range.",
            "    # Note: the bounds come from whatever data is passed in.",
            *_wrapped_list(
                list(step.columns), indent="    ", prefix="for column in ", suffix=":"
            ),
            '        values = pd.to_numeric(frame[column], errors="coerce")',
            "        frame[column] = values.clip(",
            f"            lower=values.quantile({lower}),",
            f"            upper=values.quantile({upper}),",
            "        )",
        ]
    if step.op == "datepart":
        from michi.recipes.apply import DEFAULT_DATE_PARTS

        parts = [str(item) for item in step.params.get("parts") or DEFAULT_DATE_PARTS]
        return [
            f"    # Expand {len(step.columns)} timestamp(s) into "
            f"{len(parts)} component(s).",
            "    # A raw datetime is one enormous integer; the signal is in its parts.",
            *_wrapped_list(
                list(step.columns), indent="    ", prefix="for column in ", suffix=":"
            ),
            "        stamps = pd.to_datetime("
            'frame[column], errors="coerce", format="mixed")',
            *_wrapped_list(parts, indent="        ", prefix="for part in ", suffix=":"),
            # `week` is the one part pandas does not expose on `.dt`. The
            # branch is emitted only when it was actually asked for: dead code
            # in a file meant to be read is worse than a longer emitter.
            *(
                [
                    '            if part == "week":',
                    "                values = stamps.dt.isocalendar().week",
                    "            else:",
                    "                values = getattr(stamps.dt, part)",
                ]
                if "week" in parts
                else ["            values = getattr(stamps.dt, part)"]
            ),
            '            frame[f"{column}_{part}"] = pd.to_numeric('
            'values, errors="coerce")',
        ]
    if step.op == "log":
        method = str(step.params.get("method", "log1p"))
        body = (
            [
                "        frame[column] = np.sign(values) * np.log1p(values.abs())",
            ]
            if method == "signed"
            else [
                "        frame[column] = np.log1p(values.where(values >= 0))",
            ]
        )
        return [
            f"    # Compress a long right tail on {len(step.columns)} column(s).",
            *_wrapped_list(
                list(step.columns), indent="    ", prefix="for column in ", suffix=":"
            ),
            '        values = pd.to_numeric(frame[column], errors="coerce")',
            *body,
        ]
    if step.op == "interact":
        method = str(step.params.get("method", "product"))
        expression = (
            "first / second.replace(0, np.nan)"
            if method == "ratio"
            else "first * second"
        )
        suffix = "_over_" if method == "ratio" else "_x_"
        return [
            f"    # Pairwise {method}s of {len(step.columns)} column(s).",
            "    # A linear model cannot represent a combination it was not handed.",
            *_wrapped_list(
                list(step.columns),
                indent="    ",
                prefix="for left, right in combinations(",
                suffix=", 2):",
            ),
            '        first = pd.to_numeric(frame[left], errors="coerce")',
            '        second = pd.to_numeric(frame[right], errors="coerce")',
            f'        frame[f"{{left}}{suffix}{{right}}"] = {expression}',
        ]
    if step.op == "binarize":
        threshold = float(step.params.get("threshold", 0.0))
        return [
            f"    # Reduce {len(step.columns)} column(s) to above/below {threshold}.",
            "    # A missing value stays missing rather than becoming False.",
            *_wrapped_list(
                list(step.columns), indent="    ", prefix="for column in ", suffix=":"
            ),
            '        values = pd.to_numeric(frame[column], errors="coerce")',
            f"        frame[column] = (values > {threshold})"
            '.astype("float").where(values.notna())',
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
        if step.why:
            lines.append(f"            # {step.why}")
        lines.append("            (")
        lines.append(f"                {_literal(f'{step.op}_{index}')},")
        # A transformer with enough arguments does not fit on one line at this
        # indent, and the generated file has to satisfy the same formatter as
        # michi's own source, so the emitter may hand back several lines.
        rendered = _transformer(step).splitlines()
        lines.extend(f"                {part}" for part in rendered[:-1])
        lines.append(f"                {rendered[-1]},")
        lines.extend(
            _wrapped_list(
                list(step.columns), indent="                ", prefix="", suffix=","
            )
        )
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
    if step.op == "bin":
        bins = int(step.params.get("bins", 5))
        strategy = str(step.params.get("strategy", "quantile"))
        # `subsample=None` because sklearn otherwise subsamples large inputs
        # before computing quantile edges, which makes the bins depend on a
        # random draw — and a recipe that produces different bins on the same
        # data is not a recipe.
        return (
            "KBinsDiscretizer(\n"
            f"    n_bins={bins},\n"
            '    encode="ordinal",\n'
            f'    strategy="{strategy}",\n'
            "    subsample=None,\n"
            ")"
        )
    if step.op == "target-encode":
        from michi.recipes.encoders import DEFAULT_SMOOTHING

        smoothing = float(step.params.get("smoothing", DEFAULT_SMOOTHING))
        return f"OutOfFoldTargetEncoder(smoothing={smoothing})"
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


_ENCODER_SOURCE = '''class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with the target's mean for that category.

    Written out in full rather than imported, so this file keeps its promise
    to depend on nothing but pandas and scikit-learn — and so you can read
    exactly what the encoding did.

    Training rows are encoded out of fold: each row's value comes from folds
    that did not contain it. A row encoded with a mean that included its own
    label has memorised the answer, which is the most common silent leak in
    tabular machine learning.

    Small categories are pulled toward the global mean by `smoothing`, read as
    "how many observations of evidence before a category speaks for itself".
    """

    def __init__(self, smoothing=20.0, n_splits=5, random_state=0):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def _means(self, column, y, prior):
        grouped = y.groupby(column.astype("object"), dropna=False)
        return (grouped.sum() + self.smoothing * prior) / (
            grouped.count() + self.smoothing
        )

    def fit(self, X, y):
        frame = pd.DataFrame(X).reset_index(drop=True)
        target = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        self.prior_ = float(target.mean())
        self.mappings_ = {
            name: self._means(frame[name], target, self.prior_)
            for name in frame.columns
        }
        return self

    def transform(self, X):
        # New rows were not part of fitting, so the full-data mapping is safe.
        frame = pd.DataFrame(X).reset_index(drop=True)
        out = np.empty((len(frame), frame.shape[1]), dtype=float)
        for position, name in enumerate(frame.columns):
            encoded = frame[name].astype("object").map(self.mappings_[name])
            out[:, position] = encoded.fillna(self.prior_).to_numpy(dtype=float)
        return out

    def fit_transform(self, X, y=None, **kwargs):
        # Training rows must not see themselves; each fold is encoded by the
        # others. This is the half that differs from transform, on purpose.
        frame = pd.DataFrame(X).reset_index(drop=True)
        target = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        self.fit(frame, target)

        out = np.full((len(frame), frame.shape[1]), self.prior_, dtype=float)
        splits = min(self.n_splits, len(frame))
        if splits < 2:
            return out

        folds = KFold(n_splits=splits, shuffle=True, random_state=self.random_state)
        for train_index, test_index in folds.split(frame):
            inner_y = target.iloc[train_index]
            prior = float(inner_y.mean())
            for position, name in enumerate(frame.columns):
                mapping = self._means(frame[name].iloc[train_index], inner_y, prior)
                encoded = frame[name].iloc[test_index].astype("object").map(mapping)
                out[test_index, position] = encoded.fillna(prior).to_numpy(dtype=float)
        return out
'''


def _encoder_class(fitted: tuple[RecipeStep, ...]) -> list[str]:
    """Emit the target encoder's source, when the recipe uses one.

    Written into the file rather than imported from michi: the generated code
    promises to depend on pandas and scikit-learn only, and a reader of a
    target encoding deserves to see the arithmetic that produced it.
    """
    if not _uses("target-encode", fitted):
        return []
    return [*_ENCODER_SOURCE.splitlines(), "", ""]
