"""Tests that error messages stay actionable.

michi promises that an error names the exact command to run next. Two things
quietly break that promise: renaming the distribution without updating the
messages, and letting a terminal renderer interpret the square brackets in
``michi-ml[bench]`` as markup and swallow them. Both have happened; neither
should again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from michi.cli.app import app
from michi.core.errors import DISTRIBUTION, install_hint

runner = CliRunner()


# --- the install hint ------------------------------------------------------


def test_the_hint_names_the_distribution_not_the_import_name() -> None:
    """The import name is `michi`; the thing you install is not."""
    assert install_hint() == f"pip install {DISTRIBUTION}"
    assert install_hint("bench") == f"pip install '{DISTRIBUTION}[bench]'"


def test_the_hint_is_the_only_place_the_package_name_is_written() -> None:
    """A rename must not be able to leave a stale command in a message.

    Any hardcoded `pip install michi...` in the source is a message that a
    future rename will silently falsify.
    """
    source = Path(__file__).parent.parent.parent.parent / "src" / "michi"
    offenders: list[str] = []
    for path in source.rglob("*.py"):
        if path.name == "errors.py":
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if re.search(r"pip install\s+'?michi", line):
                offenders.append(f"{path.name}:{number}")
    assert offenders == []


# --- markup safety ---------------------------------------------------------


def test_an_extra_hint_survives_rendering(tmp_path: Path) -> None:
    """`[bench]` must reach the user, not be eaten as a style tag."""
    data = tmp_path / "data.csv"
    data.write_text("a,b,label\n1,2,0\n3,4,1\n" * 30, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "bench",
            str(data),
            "--target",
            "label",
            "--models",
            "xgb",
            "--cv",
            "3",
            "--no-save",
        ],
    )
    combined = (result.output + str(result.exception or "")).replace("\n", " ")
    assert "[bench]" in re.sub(r"\s+", " ", combined)


def test_error_text_is_rendered_verbatim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path containing brackets is shown as typed, not interpreted."""
    from michi.cli.errors import fail

    fail(f"no such file: {tmp_path / 'data [v2].csv'}")
    assert "[v2]" in capsys.readouterr().err.replace("\n", "")


# --- messages point somewhere ----------------------------------------------


def test_a_missing_dataset_names_both_ways_to_supply_one(tmp_path: Path) -> None:
    """An error says what to do, not only what went wrong."""
    import os

    original = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["inspect"])
    finally:
        os.chdir(original)
    combined = re.sub(r"\s+", " ", result.output)
    assert "michi.toml" in combined
    assert result.exit_code == 2


def test_an_unsupported_file_type_lists_what_works(tmp_path: Path) -> None:
    """A rejected file names the formats michi does read."""
    path = tmp_path / "data.docx"
    path.write_text("not tabular", encoding="utf-8")
    result = runner.invoke(app, ["inspect", str(path)])
    assert ".csv" in result.output
    assert result.exit_code == 2
