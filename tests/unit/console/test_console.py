"""Tests for the interactive console.

The console's contract is that it adds nothing: every verb dispatches to the
same CLI, and every session converts back into one-shot commands. These tests
hold it to that.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from michi.console import COMMANDS, Session, banner, dispatch, expand, split_line
from michi.core.config import ProjectDefaults, load_defaults


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=100, no_color=True, force_terminal=False), buffer


def _run(line: str, session: Session) -> str:
    console, buffer = _console()
    dispatch(line, session, console)
    return buffer.getvalue()


# --- context ---------------------------------------------------------------


def test_prompt_shows_the_context(messy_csv: Path) -> None:
    """State is visible at every keystroke, not one command away."""
    session = Session()
    assert session.prompt == "michi › "
    dispatch(f"use {messy_csv}", session, _console()[0])
    assert "messy.csv" in session.prompt
    dispatch("set target purchased", session, _console()[0])
    assert "purchased" in session.prompt


def test_use_loads_columns_for_completion(messy_csv: Path) -> None:
    """Loading data is what makes completing your own columns possible."""
    session = Session()
    _run(f"use {messy_csv}", session)
    assert "purchased" in session.columns


def test_use_reports_a_missing_file(tmp_path: Path) -> None:
    """A bad path is reported without ending the session."""
    session = Session()
    output = _run(f"use {tmp_path / 'absent.csv'}", session)
    assert "no such file" in output


def test_set_rejects_unknown_settings() -> None:
    """Only the michi.toml keys are settable, and typos say so."""
    session = Session()
    assert "unknown setting" in _run("set nonsense value", session)


def test_set_warns_when_target_is_not_a_column(messy_csv: Path) -> None:
    """michi notes a mismatch but still obeys — it is the user's call."""
    session = Session()
    _run(f"use {messy_csv}", session)
    output = _run("set target not_a_column", session)
    assert "not a column" in output
    assert session.target == "not_a_column"


def test_set_rejects_a_non_numeric_seed() -> None:
    """Numeric settings are validated rather than silently mangled."""
    session = Session()
    assert "whole number" in _run("set seed abc", session)


def test_unset_restores_the_default() -> None:
    """Clearing a value returns it to the built-in default."""
    session = Session()
    _run("set runs_dir elsewhere", session)
    _run("unset runs_dir", session)
    assert session.runs_dir == "runs"


def test_show_context_lists_every_setting() -> None:
    """`show context` is the state dump; nothing is hidden."""
    output = _run("show context", Session())
    for key in ("data", "target", "recipe", "runs_dir", "models", "seed", "cv"):
        assert key in output


def test_show_columns_needs_a_dataset() -> None:
    """Without data, michi says what to do rather than showing nothing."""
    assert "use data.csv" in _run("show columns", Session())


def test_show_models_lists_the_catalogue() -> None:
    """The model menu is reachable from inside the console."""
    output = _run("show models", Session())
    assert "rf" in output
    assert "dummy" in output


# --- line splitting --------------------------------------------------------


def test_windows_paths_survive_splitting() -> None:
    """A backslash is a path separator here, never an escape character."""
    assert split_line(r"use C:\data\train.csv") == ["use", r"C:\data\train.csv"]


def test_quoted_paths_with_spaces_work() -> None:
    """Quoting still groups an argument containing spaces."""
    assert split_line('use "my file.csv"') == ["use", "my file.csv"]


# --- dispatch --------------------------------------------------------------


def test_unknown_command_points_at_help() -> None:
    """A wrong command is a nudge, not a failure."""
    assert "help" in _run("frobnicate", Session())


def test_help_lists_every_command() -> None:
    """Every command is discoverable from `help`."""
    output = _run("help", Session())
    for entry in COMMANDS:
        assert entry.name in output


def test_help_on_one_command_shows_its_usage() -> None:
    """`help bench` explains that shell flags work unchanged."""
    output = _run("help bench", Session())
    assert "bench" in output
    assert "flags" in output


def test_exit_ends_the_session() -> None:
    """`exit` and `quit` both end the loop."""
    console, _ = _console()
    assert dispatch("exit", Session(), console) is False
    assert dispatch("quit", Session(), console) is False


def test_a_bad_line_does_not_end_the_session() -> None:
    """An unbalanced quote is reported, and the session continues."""
    console, buffer = _console()
    assert dispatch('use "unclosed', Session(), console) is True
    assert "parse" in buffer.getvalue()


def test_a_failing_verb_does_not_end_the_session(tmp_path: Path) -> None:
    """A verb that errors leaves the console running so you can retry."""
    session = Session(data=str(tmp_path / "absent.csv"))
    console, _ = _console()
    assert dispatch("inspect", session, console) is True


# --- expansion: the proof it is a skin -------------------------------------


def test_context_fills_the_flags_you_did_not_type(messy_csv: Path) -> None:
    """A bare verb expands into the full one-shot command."""
    session = Session(data=str(messy_csv), target="purchased")
    argv = expand("bench", [], session)
    assert argv[0] == "bench"
    assert str(messy_csv) in argv
    assert "--target" in argv and "purchased" in argv
    assert "--models" in argv
    assert "--cv" in argv


def test_explicit_flags_win_over_context(messy_csv: Path) -> None:
    """What the user typed is never overridden by the session."""
    session = Session(data=str(messy_csv), target="purchased")
    argv = expand("bench", ["--target", "other"], session)
    assert argv.count("--target") == 1
    assert "other" in argv


def test_eval_places_the_model_before_the_data(messy_csv: Path) -> None:
    """Positional order matches the one-shot command exactly."""
    session = Session(data=str(messy_csv), target="purchased")
    argv = expand("eval", ["model.pkl"], session)
    assert argv[1] == "model.pkl"
    assert argv[2] == str(messy_csv)


def test_report_defaults_to_the_runs_directory() -> None:
    """Context supplies the runs directory when none is typed."""
    assert "runs" in expand("report", [], Session())


def test_every_console_verb_exists_in_the_cli() -> None:
    """No capability may exist only in the console."""
    from michi.cli.app import app

    cli_names = {command.name for command in app.registered_commands if command.name}
    console_verbs = {entry.name for entry in COMMANDS if entry.group == "verbs"}
    assert console_verbs <= cli_names


# --- history and saving ----------------------------------------------------


def test_history_records_what_was_run(messy_csv: Path) -> None:
    """The session remembers the commands, in order."""
    session = Session()
    _run(f"use {messy_csv}", session)
    _run("set target purchased", session)
    assert len(session.history) == 2


def test_history_exports_a_replayable_script(messy_csv: Path, tmp_path: Path) -> None:
    """Exploration converts into something scriptable."""
    session = Session(data=str(messy_csv), target="purchased")
    session.record("inspect --explain")
    destination = tmp_path / "session.sh"
    _run(f"history --export {destination}", session)

    script = destination.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash")
    assert "michi inspect" in script
    assert "--target purchased" in script


def test_exported_script_contains_only_one_shot_commands(
    messy_csv: Path, tmp_path: Path
) -> None:
    """Console-only commands never leak into the script."""
    session = Session(data=str(messy_csv))
    session.record("show context")
    session.record("set target purchased")
    session.record("inspect")
    destination = tmp_path / "session.sh"
    _run(f"history --export {destination}", session)

    script = destination.read_text(encoding="utf-8")
    assert "show context" not in script
    assert "set target" not in script
    assert "michi inspect" in script


def test_save_writes_the_context_to_toml(tmp_path: Path) -> None:
    """`save` turns session state into a file you can read and commit."""
    session = Session(data="data.csv", target="churned", seed=7)
    destination = tmp_path / "michi.toml"
    _run(f"save {destination}", session)

    text = destination.read_text(encoding="utf-8")
    assert 'target = "churned"' in text
    assert "seed = 7" in text
    assert session.dirty is False


def test_saved_context_reloads_into_a_session(tmp_path: Path) -> None:
    """What the console saves, the console restores."""
    Session(data="data.csv", target="churned", cv=10).save(tmp_path / "michi.toml")
    defaults = load_defaults(tmp_path)
    restored = Session.from_defaults(defaults)
    assert restored.target == "churned"
    assert restored.cv == 10


# --- banner ----------------------------------------------------------------


def test_banner_is_short() -> None:
    """One screen, no more."""
    assert len(banner().splitlines()) < 15
    assert "michi" in banner()


# --- project defaults ------------------------------------------------------


def test_missing_config_is_not_an_error(tmp_path: Path) -> None:
    """No michi.toml simply means built-in defaults."""
    assert load_defaults(tmp_path).is_empty


def test_config_is_found_in_a_parent_directory(tmp_path: Path) -> None:
    """Running from a subdirectory still finds the project's defaults."""
    (tmp_path / "michi.toml").write_text(
        '[defaults]\ntarget = "churned"\n', encoding="utf-8"
    )
    nested = tmp_path / "notebooks" / "deep"
    nested.mkdir(parents=True)
    assert load_defaults(nested).target == "churned"


def test_broken_config_is_reported(tmp_path: Path) -> None:
    """A malformed config fails loudly rather than being ignored."""
    from michi.core.errors import MichiError

    (tmp_path / "michi.toml").write_text("[defaults\n", encoding="utf-8")
    with pytest.raises(MichiError, match="could not parse"):
        load_defaults(tmp_path)


def test_explicit_values_beat_configured_defaults() -> None:
    """Precedence is flags > michi.toml > built-in, always."""
    defaults = ProjectDefaults(target="from_file")
    assert defaults.resolve("target", "from_flag") == "from_flag"
    assert defaults.resolve("target", None) == "from_file"
    assert defaults.origin("target", "from_flag") == "flag"
    assert defaults.origin("target", None) == "michi.toml"
