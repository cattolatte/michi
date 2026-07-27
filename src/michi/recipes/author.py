"""Authoring a recipe — from findings, from flags, or interactively.

Design Principles
-----------------
- **Triage findings, not columns.** Walking two hundred columns one at a time
  is torture, and real datasets have two hundred columns. The wizard walks the
  *issues* ``inspect`` found, grouped, with a bulk answer available for each
  group.
- **Contextual menus only.** Imputation options appear for columns that
  actually have missing values. An encyclopedic list of every possible
  operation is noise for an expert and paralysis for a learner.
- **Flag parity is absolute.** Everything the wizard can produce, flags can
  produce, and the wizard prints the equivalent command when it finishes.
  A capability that exists only interactively is invisible to scripts, to CI,
  and to anyone automating michi.
- **michi never picks.** Every prompt lists options; the default is always
  "leave it alone".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from michi.core.artifacts import ColumnKind, DatasetProfile, Severity
from michi.core.errors import RecipeError
from michi.recipes.model import Recipe, RecipeStep, SourceSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from michi.core.artifacts import Finding

__all__ = [
    "Choice",
    "Question",
    "command_for",
    "questions_for",
    "recipe_from_answers",
    "recipe_from_flags",
]


@dataclass(frozen=True, slots=True)
class Choice:
    """One option in a menu michi offers."""

    key: str
    label: str
    step: RecipeStep | None


@dataclass(frozen=True, slots=True)
class Question:
    """One decision the user is asked to make about a group of findings."""

    kind: str
    columns: tuple[str, ...]
    prompt: str
    detail: str
    choices: tuple[Choice, ...]

    @property
    def is_bulk(self) -> bool:
        """Whether this question covers several columns at once."""
        return len(self.columns) > 1


def questions_for(profile: DatasetProfile) -> tuple[Question, ...]:
    """Turn a profile's findings into the decisions worth asking about.

    Findings of the same kind are grouped into one question, so a dataset with
    forty sparsely-missing columns produces one prompt rather than forty.
    Informational findings are not asked about at all.
    """
    grouped: dict[str, list[Finding]] = {}
    for finding in profile.findings_by_severity():
        if finding.severity is Severity.INFO and finding.kind != "identifier-like":
            continue
        if not finding.columns:
            continue
        grouped.setdefault(finding.kind, []).append(finding)

    questions: list[Question] = []
    for kind, findings in grouped.items():
        columns = tuple(
            dict.fromkeys(name for finding in findings for name in finding.columns)
        )
        builder = _BUILDERS.get(kind)
        if builder is None:
            continue
        question = builder(profile, columns, findings)
        if question is not None:
            questions.append(question)
    return tuple(questions)


def recipe_from_answers(
    profile: DatasetProfile,
    answers: Sequence[tuple[Question, str]],
    *,
    target: str | None = None,
) -> Recipe:
    """Assemble a recipe from the choices a user made."""
    steps: list[RecipeStep] = []
    for question, key in answers:
        choice = next((item for item in question.choices if item.key == key), None)
        if choice is None:
            known = ", ".join(item.key for item in question.choices)
            msg = f"unknown choice {key!r} for {question.kind}; expected: {known}"
            raise RecipeError(msg)
        if choice.step is not None:
            steps.append(choice.step)
    return Recipe(
        steps=_ordered(steps),
        target=target or profile.target,
        source=_schema_of(profile),
    )


def recipe_from_flags(
    profile: DatasetProfile | None,
    *,
    drop: Sequence[str] = (),
    dedupe: bool = False,
    cast: Sequence[tuple[str, str]] = (),
    impute: Sequence[tuple[str, str]] = (),
    clip: Sequence[str] = (),
    encode: Sequence[tuple[str, str]] = (),
    scale: Sequence[tuple[str, str]] = (),
    datepart: Sequence[tuple[str, str]] = (),
    log: Sequence[tuple[str, str]] = (),
    interact: Sequence[str] = (),
    binarize: Sequence[tuple[str, str]] = (),
    bin_: Sequence[tuple[str, str]] = (),
    target: str | None = None,
) -> Recipe:
    """Assemble a recipe entirely from command-line flags.

    This is the non-interactive path, and it can express everything the wizard
    can. Column/strategy pairs arrive as ``("age", "median")``.
    """
    steps: list[RecipeStep] = []
    if drop:
        steps.append(
            RecipeStep("drop", {"columns": list(drop)}, why="requested with --drop")
        )
    if dedupe:
        steps.append(RecipeStep("dedupe", {}, why="requested with --dedupe"))
    for column, target_type in cast:
        steps.append(RecipeStep("cast", {"columns": [column], "to": target_type}))
    for column, strategy in impute:
        steps.append(RecipeStep("impute", {"columns": [column], "strategy": strategy}))
    if clip:
        steps.append(
            RecipeStep(
                "clip",
                {
                    "columns": list(clip),
                    "lower_quantile": 0.01,
                    "upper_quantile": 0.99,
                },
            )
        )
    for column, method in encode:
        steps.append(RecipeStep("encode", {"columns": [column], "method": method}))
    for column, method in scale:
        steps.append(RecipeStep("scale", {"columns": [column], "method": method}))

    for column, parts in datepart:
        steps.append(
            RecipeStep(
                "datepart",
                {
                    "columns": [column],
                    "parts": [item for item in parts.split("+") if item],
                },
                why="requested with --datepart",
            )
        )
    for column, method in log:
        steps.append(
            RecipeStep("log", {"columns": [column], "method": method}, why="skew")
        )
    for column, threshold in binarize:
        steps.append(
            RecipeStep("binarize", {"columns": [column], "threshold": float(threshold)})
        )
    for column, spec in bin_:
        count, _, strategy = spec.partition(":")
        steps.append(
            RecipeStep(
                "bin",
                {
                    "columns": [column],
                    "bins": int(count),
                    "strategy": strategy or "quantile",
                },
            )
        )
    if interact:
        # One step over all named columns, not one per column: `interact`
        # combines columns with each other, so splitting it would produce no
        # pairs at all.
        steps.append(
            RecipeStep(
                "interact",
                {"columns": list(interact), "method": "product"},
                why="requested with --interact",
            )
        )

    return Recipe(
        steps=_ordered(steps),
        target=target or (profile.target if profile else None),
        source=_schema_of(profile) if profile else SourceSchema(),
    )


def command_for(recipe: Recipe, data_path: str) -> str:
    """Render the non-interactive command equivalent to a recipe.

    Printed when a wizard session ends, so an exploratory session always
    converts into something scriptable.

    Examples
    --------
    >>> recipe = Recipe(steps=(RecipeStep("drop", {"columns": ["id"]}),))
    >>> command_for(recipe, "data.csv")
    'michi clean data.csv --drop id'
    """
    parts = [f"michi clean {data_path}"]
    drops: list[str] = []
    for step in recipe.steps:
        columns = step.columns
        if step.op == "drop":
            drops.extend(columns)
        elif step.op == "dedupe":
            parts.append("--dedupe")
        elif step.op == "cast":
            target_type = step.params.get("to", "numeric")
            parts.extend(f"--cast {name}={target_type}" for name in columns)
        elif step.op == "impute":
            strategy = step.params.get("strategy", "median")
            parts.extend(f"--impute {name}={strategy}" for name in columns)
        elif step.op == "clip":
            parts.extend(f"--clip {name}" for name in columns)
        elif step.op == "encode":
            method = step.params.get("method", "onehot")
            parts.extend(f"--encode {name}={method}" for name in columns)
        elif step.op == "scale":
            method = step.params.get("method", "standard")
            parts.extend(f"--scale {name}={method}" for name in columns)
        elif step.op == "datepart":
            pieces = "+".join(str(item) for item in step.params.get("parts") or ())
            parts.extend(
                f"--datepart {name}={pieces}" if pieces else f"--datepart {name}"
                for name in columns
            )
        elif step.op == "log":
            method = step.params.get("method", "log1p")
            parts.extend(f"--log {name}={method}" for name in columns)
        elif step.op == "binarize":
            threshold = step.params.get("threshold", 0.0)
            parts.extend(f"--binarize {name}={threshold}" for name in columns)
        elif step.op == "bin":
            count = step.params.get("bins", 5)
            strategy = step.params.get("strategy", "quantile")
            parts.extend(f"--bin {name}={count}:{strategy}" for name in columns)
        elif step.op == "interact":
            parts.append(f"--interact {','.join(columns)}")
    if drops:
        parts.insert(1, f"--drop {','.join(drops)}")
    if recipe.target:
        parts.append(f"--target {recipe.target}")
    return " ".join(parts)


# --- question builders -----------------------------------------------------


def _keep(label: str = "leave them as they are") -> Choice:
    """The default choice: michi's own preference is always to do nothing."""
    return Choice("keep", label, None)


def _drop_question(
    kind: str, columns: tuple[str, ...], prompt: str, detail: str, why: str
) -> Question:
    return Question(
        kind=kind,
        columns=columns,
        prompt=prompt,
        detail=detail,
        choices=(
            Choice(
                "drop",
                f"drop {'them' if len(columns) > 1 else 'it'}",
                RecipeStep("drop", {"columns": list(columns)}, why=why),
            ),
            _keep(),
        ),
    )


def _build_empty(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    return _drop_question(
        "empty-column",
        columns,
        f"{_count(columns)} entirely empty — every value is missing",
        "They carry no information about any row.",
        "column was entirely missing",
    )


def _build_constant(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    return _drop_question(
        "constant-column",
        columns,
        f"{_count(columns)} a single distinct value throughout",
        "They cannot help a model distinguish one row from another.",
        "column held one distinct value",
    )


def _build_high_missing(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    detail = "; ".join(f"{f.columns[0]} {f.summary}" for f in findings[:4])
    return Question(
        kind="high-missing",
        columns=columns,
        prompt=f"{_count(columns)} more than half missing",
        detail=detail,
        choices=(
            Choice(
                "drop",
                "drop them",
                RecipeStep(
                    "drop", {"columns": list(columns)}, why="more than half missing"
                ),
            ),
            Choice(
                "median",
                "impute the median (numeric) — invents most of the column",
                RecipeStep(
                    "impute",
                    {"columns": list(columns), "strategy": "median"},
                    why="mostly missing; imputed rather than dropped",
                ),
            ),
            Choice(
                "constant",
                "fill with a constant marker",
                RecipeStep(
                    "impute",
                    {"columns": list(columns), "strategy": "constant", "value": 0},
                    why="mostly missing; filled with a sentinel",
                ),
            ),
            _keep(),
        ),
    )


def _build_missing(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    numeric = [
        name for name in columns if _kind_of(profile, name) is ColumnKind.NUMERIC
    ]
    other = [name for name in columns if name not in numeric]
    detail = "; ".join(f"{f.columns[0]} {f.summary}" for f in findings[:4])

    choices: list[Choice] = []
    if numeric:
        choices.append(
            Choice(
                "median",
                f"impute the median for {len(numeric)} numeric column(s)",
                RecipeStep(
                    "impute",
                    {"columns": numeric, "strategy": "median"},
                    why="missing values imputed with the median",
                ),
            )
        )
        choices.append(
            Choice(
                "mean",
                f"impute the mean for {len(numeric)} numeric column(s)",
                RecipeStep(
                    "impute",
                    {"columns": numeric, "strategy": "mean"},
                    why="missing values imputed with the mean",
                ),
            )
        )
    if other:
        choices.append(
            Choice(
                "most_frequent",
                f"impute the most frequent value for {len(other)} column(s)",
                RecipeStep(
                    "impute",
                    {"columns": other, "strategy": "most_frequent"},
                    why="missing values imputed with the mode",
                ),
            )
        )
    choices.append(
        Choice(
            "drop_rows",
            "drop the affected rows",
            RecipeStep(
                "impute",
                {"columns": list(columns), "strategy": "drop_rows"},
                why="rows with missing values removed",
            ),
        )
    )
    choices.append(
        Choice(
            "drop",
            "drop the columns",
            RecipeStep(
                "drop", {"columns": list(columns)}, why="column had missing values"
            ),
        )
    )
    choices.append(_keep())
    return Question(
        kind="missing",
        columns=columns,
        prompt=f"{_count(columns)} some missing values",
        detail=detail,
        choices=tuple(choices),
    )


def _build_duplicate_rows(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question | None:
    return None  # dataset-level; handled separately from column findings


def _build_duplicate_columns(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    # Keep the first of each identical group, offer to drop the rest.
    redundant = [name for finding in findings for name in finding.columns[1:]]
    return Question(
        kind="duplicate-columns",
        columns=tuple(dict.fromkeys(redundant)),
        prompt=f"{len(findings)} group(s) of identical columns",
        detail="; ".join(f.summary for f in findings[:3]),
        choices=(
            Choice(
                "drop",
                "keep the first of each group, drop the copies",
                RecipeStep(
                    "drop",
                    {"columns": list(dict.fromkeys(redundant))},
                    why="identical to another column",
                ),
            ),
            _keep("keep every copy"),
        ),
    )


def _build_correlated(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    redundant = [finding.columns[1] for finding in findings if len(finding.columns) > 1]
    return Question(
        kind="highly-correlated",
        columns=tuple(dict.fromkeys(redundant)),
        prompt=f"{len(findings)} pair(s) of nearly redundant columns",
        detail="; ".join(f"{', '.join(f.columns)}: {f.summary}" for f in findings[:3]),
        choices=(
            Choice(
                "drop",
                "keep one of each pair, drop the other",
                RecipeStep(
                    "drop",
                    {"columns": list(dict.fromkeys(redundant))},
                    why="near-perfectly correlated with another column",
                ),
            ),
            _keep("keep both of each pair"),
        ),
    )


def _build_numeric_text(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    return Question(
        kind="numeric-stored-as-text",
        columns=columns,
        prompt=f"{_count(columns)} numbers stored as text",
        detail="Left as text they are treated as unordered categories.",
        choices=(
            Choice(
                "cast",
                "parse them as numbers (unparseable values become missing)",
                RecipeStep(
                    "cast",
                    {"columns": list(columns), "to": "numeric"},
                    why="values parsed as numbers rather than categories",
                ),
            ),
            _keep("leave them as text"),
        ),
    )


def _build_datetime_text(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    return Question(
        kind="datetime-stored-as-text",
        columns=columns,
        prompt=f"{_count(columns)} dates stored as text",
        detail="Sorting is lexicographic and no time arithmetic is possible.",
        choices=(
            Choice(
                "cast",
                "parse them as timestamps",
                RecipeStep(
                    "cast",
                    {"columns": list(columns), "to": "datetime"},
                    why="values parsed as timestamps",
                ),
            ),
            _keep("leave them as text"),
        ),
    )


def _build_outliers(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    return Question(
        kind="outliers",
        columns=columns,
        prompt=f"{_count(columns)} values far outside the interquartile range",
        detail="They may be recording errors, or the real tail of the data.",
        choices=(
            Choice(
                "clip",
                "clip to the 1st–99th percentile",
                RecipeStep(
                    "clip",
                    {
                        "columns": list(columns),
                        "lower_quantile": 0.01,
                        "upper_quantile": 0.99,
                    },
                    why="extreme values clipped to percentile bounds",
                ),
            ),
            _keep("leave the extremes in place"),
        ),
    )


def _build_identifier(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    return _drop_question(
        "identifier-like",
        columns,
        f"{_count(columns)} a distinct value in every row",
        "That is the signature of an identifier rather than a feature.",
        "column looked like an identifier",
    )


def _build_high_cardinality(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    return Question(
        kind="high-cardinality",
        columns=columns,
        prompt=f"{_count(columns)} very many distinct categories",
        detail="One-hot encoding these would create a very wide, sparse matrix.",
        choices=(
            Choice(
                "ordinal",
                "encode as integer codes instead of one-hot",
                RecipeStep(
                    "encode",
                    {"columns": list(columns), "method": "ordinal"},
                    why="high cardinality; ordinal encoding chosen over one-hot",
                ),
            ),
            Choice(
                "drop",
                "drop them",
                RecipeStep(
                    "drop", {"columns": list(columns)}, why="high-cardinality column"
                ),
            ),
            _keep(),
        ),
    )


def _build_leakage(
    profile: DatasetProfile, columns: tuple[str, ...], findings: list[Finding]
) -> Question:
    return Question(
        kind="leakage-suspect",
        columns=columns,
        prompt=f"{_count(columns)} suspiciously predictive of the target",
        detail=(
            "Usually a value recorded at or after the outcome, which will not "
            "exist when the model is used."
        ),
        choices=(
            Choice(
                "drop",
                "drop them",
                RecipeStep(
                    "drop",
                    {"columns": list(columns)},
                    why="suspected target leakage",
                ),
            ),
            _keep("keep them — they genuinely precede the outcome"),
        ),
    )


_BUILDERS = {
    "empty-column": _build_empty,
    "constant-column": _build_constant,
    "high-missing": _build_high_missing,
    "missing": _build_missing,
    "duplicate-rows": _build_duplicate_rows,
    "duplicate-columns": _build_duplicate_columns,
    "highly-correlated": _build_correlated,
    "numeric-stored-as-text": _build_numeric_text,
    "datetime-stored-as-text": _build_datetime_text,
    "outliers": _build_outliers,
    "identifier-like": _build_identifier,
    "high-cardinality": _build_high_cardinality,
    "leakage-suspect": _build_leakage,
}


# --- helpers ---------------------------------------------------------------


_STEP_ORDER = {
    "drop": 0,
    "dedupe": 1,
    "cast": 2,
    # Derivation sits between casting and imputation: `datepart` needs a real
    # timestamp to read parts from, and the columns it creates should be
    # available for imputation like any other.
    "datepart": 3,
    "impute": 4,
    "clip": 5,
    # Transformation of existing values, before anything derived from them.
    "log": 6,
    "binarize": 7,
    "bin": 8,
    # Interactions last among the derivations, so they multiply the final
    # values rather than the raw ones.
    "interact": 9,
    "encode": 10,
    "scale": 11,
}


def _ordered(steps: list[RecipeStep]) -> tuple[RecipeStep, ...]:
    """Put steps in the only order that makes sense.

    Dropping first avoids work on columns that are about to disappear; casting
    must precede imputation so that numbers-as-text can take a numeric fill;
    encoding and scaling come last because they assume clean values. Feature
    engineering slots in between: derive from clean values, then combine.
    """
    return tuple(sorted(steps, key=lambda step: _STEP_ORDER[step.op]))


def _schema_of(profile: DatasetProfile) -> SourceSchema:
    return SourceSchema(
        path=profile.source.path,
        sha256=profile.source.sha256,
        n_rows=profile.n_rows,
        columns={column.name: column.kind.value for column in profile.columns},
    )


def _kind_of(profile: DatasetProfile, name: str) -> ColumnKind | None:
    try:
        return profile.column(name).kind
    except KeyError:
        return None


def _count(columns: tuple[str, ...]) -> str:
    if len(columns) == 1:
        return f"{columns[0]} is"
    return f"{len(columns)} columns are"
