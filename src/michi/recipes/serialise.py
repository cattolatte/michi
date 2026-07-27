"""Reading and writing recipe files.

Design Principles
-----------------
- **Emitted YAML carries comments.** A recipe is meant to be read, edited, and
  reviewed in a pull request, so michi writes it through a template rather
  than dumping a data structure. ``yaml.dump`` cannot produce a comment, and
  the comments are where the teaching happens.
- **Hand edits are first class.** Recipes are read back with a plain YAML
  parser, so anything a user writes by hand works exactly as if michi wrote
  it. Round-tripping through michi is never required.
- Parse failures name the file and the line, because a recipe is a file a
  human typed into.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from michi.core.errors import RecipeError
from michi.recipes.model import Recipe

__all__ = ["dumps_recipe", "load_recipe", "write_recipe"]

# Strings matching this are safe to emit unquoted: they cannot be read back as
# a number, a boolean, or null, and contain nothing YAML treats specially.
_PLAIN_SCALAR = re.compile(r"(?!(?:true|false|null|yes|no|on|off|~)$)[A-Za-z_][\w.\-]*")

_HEADER = """\
# michi recipe — cleaning decisions you made, as a file you own.
#
# Apply it:   michi apply {name} data.csv -o clean.parquet
# Compile it: michi export {name} -o pipeline.py
#
# Steps run in order. Edit them freely: this file is read with a plain YAML
# parser, so a hand-written recipe works exactly like a generated one.
#
# Cleaning:    drop · dedupe · cast · impute · clip · encode · scale
# Engineering: datepart · log · interact · binarize · bin · target-encode
"""

_LEAKAGE_NOTE = """\
# Note: impute, encode, scale, bin, and target-encode learn from the data they
# see. `michi apply` fits them on whatever file you give it, which is fine for
# exploration. For modelling, use `michi export` — the generated pipeline fits
# them inside the train/test split, where they cannot leak.
"""


def dumps_recipe(recipe: Recipe, *, name: str = "michi.recipe.yaml") -> str:
    """Render a recipe as commented YAML.

    Parameters
    ----------
    recipe
        The recipe to render.
    name
        Filename used in the usage examples in the header.

    Returns
    -------
    str
        YAML text, ready to write.

    Examples
    --------
    >>> text = dumps_recipe(Recipe(steps=(RecipeStep("drop", {"columns": ["id"]}),)))
    >>> "op: drop" in text
    True
    """
    lines: list[str] = [_HEADER.format(name=name)]
    lines.append(f'schema_version: "{recipe.schema_version}"')
    lines.append(f'michi_version: "{recipe.michi_version}"')
    lines.append(f'created_at: "{recipe.created_at}"')
    if recipe.target:
        lines.append(f"target: {_scalar(recipe.target)}")
    lines.append("")

    source = recipe.source
    if source.path or source.columns:
        lines.append("# The data this recipe was written against. Applying it to")
        lines.append("# something with different columns fails loudly, on purpose.")
        lines.append("source:")
        if source.path:
            lines.append(f"  path: {_scalar(source.path)}")
        if source.sha256:
            lines.append(f'  sha256: "{source.sha256}"')
        if source.n_rows:
            lines.append(f"  n_rows: {source.n_rows}")
        if source.columns:
            lines.append("  columns:")
            for column, kind in source.columns.items():
                lines.append(f"    {_scalar(column)}: {kind}")
        lines.append("")

    if recipe.fitted_steps:
        lines.append(_LEAKAGE_NOTE.rstrip())

    lines.append("steps:")
    if not recipe.steps:
        lines[-1] = "steps: []"
    for step in recipe.steps:
        lines.append(f"  - op: {step.op}")
        for key, value in step.params.items():
            lines.append(f"    {key}: {_render(value)}")
        if step.why:
            # `why` is a field, not a comment. A comment would read just as
            # well and be lost the moment the recipe was loaded again — and
            # the reason a column was dropped is the part nobody can
            # reconstruct six months later.
            lines.append(f"    why: {_render(step.why)}")
    lines.append("")
    return "\n".join(lines)


def write_recipe(recipe: Recipe, destination: Path) -> None:
    """Write a recipe to disk as commented YAML."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        dumps_recipe(recipe, name=destination.name), encoding="utf-8"
    )


def load_recipe(source: Path) -> Recipe:
    """Read a recipe from a YAML file.

    Raises
    ------
    RecipeError
        If the file is missing, unparseable, or not a valid recipe.
    """
    import yaml

    if not source.exists():
        msg = f"no such recipe: {source}"
        raise RecipeError(msg)

    try:
        payload: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        msg = f"could not parse {source.name} as YAML: {err}"
        raise RecipeError(msg) from err
    except OSError as err:
        msg = f"could not read {source}: {err}"
        raise RecipeError(msg) from err

    if payload is None:
        msg = f"{source.name} is empty"
        raise RecipeError(msg)
    return Recipe.from_dict(payload)


def _render(value: Any) -> str:
    """Render a value as an inline YAML scalar or flow collection.

    JSON is a subset of YAML 1.2, so JSON encoding is always valid YAML — and
    unlike ``yaml.safe_dump`` it never appends a document-end marker to a bare
    scalar. Plain identifiers are emitted unquoted, because a recipe is meant
    to be read.
    """
    import json

    if isinstance(value, str) and _PLAIN_SCALAR.fullmatch(value):
        return value
    return json.dumps(value, ensure_ascii=False)


def _scalar(value: str) -> str:
    """Render a string as an inline YAML scalar."""
    return _render(value)
