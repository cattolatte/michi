"""Rebuild the README demo from captured michi output.

The text under ``captured/`` is real output, recorded by running michi against
``examples/``-style data at 80 columns. Regenerate that first if michi's output
changes, then run this:

    uv run python docs/demo/build.py
"""

from __future__ import annotations

from pathlib import Path

from render import Scene, build

HERE = Path(__file__).parent
CAPTURED = HERE / "captured"


def read(name: str, skip: int = 0) -> list[str]:
    """Read one captured transcript, trimming leading blank lines."""
    lines = [
        line.rstrip()
        for line in (CAPTURED / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    ]
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines[skip:]


def generated(marker: str, count: int) -> list[str]:
    """Read `count` lines of the exported pipeline, starting at `marker`."""
    lines = (CAPTURED / "pipeline.py").read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(marker))
    return [f"  {line}".rstrip() for line in lines[start : start + count]]


_PATH_PROMPT = "michi (customers.csv → purchased) › path"


def main() -> None:
    """Assemble the scenes and write the SVG."""
    inspect = read("inspect")
    console = read("console")
    findings_at = next(
        index
        for index, line in enumerate(inspect)
        if line.strip().startswith("Findings")
    )

    scenes = [
        Scene(
            command="michi inspect data/customers.csv --target purchased",
            output=[*inspect[:6], "", *inspect[findings_at : findings_at + 12]],
            hold=5.4,
            note="outcome_code predicts the label perfectly — a leak, not a feature",
        ),
        Scene(
            command=(
                "michi clean data/customers.csv --target purchased"
                "   # writes a recipe you own"
            ),
            # Through the `Next:` block: cutting earlier would truncate the
            # reproducing command mid-flight, which is the one line here a
            # reader might actually copy.
            output=read("clean")[:16],
            hold=4.4,
        ),
        Scene(
            command=(
                "michi bench data/customers.csv --recipe michi.recipe.yaml "
                "--models linear,rf,hist-gbm"
            ),
            output=read("bench", skip=1)[:21],
            hold=6.0,
            note="most tools would have declared a winner here",
        ),
        Scene(
            command="michi export michi.recipe.yaml -o pipeline.py",
            # The generated file is the point of `export`, so show it rather
            # than describing it: the two docstrings say why the deterministic
            # steps and the fitted ones are separated at all.
            output=[
                *read("export"),
                "",
                "  ── pipeline.py " + "─" * 62,
                *generated("def prepare", 5),
                "",
                *generated("def build_pipeline", 6),
            ],
            hold=5.4,
        ),
        Scene(
            command="michi        # no arguments opens the console",
            # Up to the `path` prompt; the map itself gets its own scene, so
            # neither has to be squeezed into a panel sized for the other.
            output=console[: console.index(_PATH_PROMPT) + 1],
            hold=5.4,
            note=(
                "Tab completes your own column names — "
                "the one thing a one-shot CLI cannot do"
            ),
        ),
        Scene(
            command="path        # the stages, and what this context can run",
            output=console[console.index(_PATH_PROMPT) + 1 :],
            hold=5.6,
            note=(
                "a map, not a driver — `walk` asks its way through, one stage at a time"
            ),
        ),
    ]
    build(scenes, HERE.parent / "demo.svg", cols=92)


if __name__ == "__main__":
    main()
