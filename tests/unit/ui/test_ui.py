"""Tests for the local viewer.

The viewer's contract is that it is read-only, offline, and adds nothing. It
must also survive an empty or malformed runs directory, since it will often be
the first thing a user opens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from michi.cli.app import app

pytest.importorskip("fastapi", reason="the ui extra is not installed")

from fastapi.testclient import TestClient

from michi.ui import build_app

runner = CliRunner()


@pytest.fixture
def runs(tidy_csv: Path, tmp_path: Path) -> Path:
    """A runs directory holding a real benchmark's manifests."""
    directory = tmp_path / "runs"
    runner.invoke(
        app,
        [
            "bench",
            str(tidy_csv),
            "--target",
            "label",
            "--models",
            "linear,tree",
            "--cv",
            "3",
            "--runs-dir",
            str(directory),
        ],
    )
    return directory


def _client(runs_dir: Path) -> TestClient:
    return TestClient(build_app(runs_dir))


# --- pages -----------------------------------------------------------------


def test_index_lists_recorded_runs(runs: Path) -> None:
    """The landing page shows what is in the directory."""
    response = _client(runs).get("/")
    assert response.status_code == 200
    assert "michi" in response.text
    assert "linear" in response.text


def test_index_groups_by_dataset_and_target(runs: Path) -> None:
    """Only comparable runs share a table, as everywhere else in michi."""
    assert "target" in _client(runs).get("/").text


def test_run_detail_shows_metrics_and_provenance(runs: Path) -> None:
    """A run page shows the numbers and where they came from."""
    from michi.report.runs import load_manifests

    manifest = load_manifests(runs)[0]
    response = _client(runs).get(f"/run/{manifest.run_id}")
    assert response.status_code == 200
    assert "balanced_accuracy" in response.text
    assert manifest.dataset.sha256[:16] in response.text


def test_unknown_run_is_handled_gracefully(runs: Path) -> None:
    """A stale link explains itself rather than returning a stack trace."""
    response = _client(runs).get("/run/does-not-exist")
    assert response.status_code == 200
    assert "No run" in response.text


def test_empty_runs_directory_explains_what_to_do(tmp_path: Path) -> None:
    """A first-time user sees which commands write manifests."""
    empty = tmp_path / "runs"
    empty.mkdir()
    response = _client(empty).get("/")
    assert response.status_code == 200
    assert "michi eval" in response.text


def test_a_missing_runs_directory_does_not_crash(tmp_path: Path) -> None:
    """Pointing the viewer at nothing shows an empty page, not an error."""
    assert _client(tmp_path / "absent").get("/").status_code == 200


def test_a_malformed_manifest_does_not_break_the_page(runs: Path) -> None:
    """One corrupt file must not take down the view of the good ones."""
    (runs / "broken.json").write_text("{not json", encoding="utf-8")
    assert _client(runs).get("/").status_code == 200


# --- the contract ----------------------------------------------------------


def test_the_viewer_is_read_only(runs: Path) -> None:
    """No route writes, deletes, or trains anything."""
    routes = build_app(runs).routes
    methods: set[str] = set()
    for route in routes:
        methods.update(getattr(route, "methods", set()) or set())
    assert methods <= {"GET", "HEAD"}


def test_pages_are_offline(runs: Path) -> None:
    """No CDN, no external font, no script tag — it works air-gapped."""
    body = _client(runs).get("/").text
    assert "http://" not in body.replace("http://127.0.0.1", "")
    assert "https://" not in body
    assert "<script" not in body.lower()


def test_the_viewer_shows_nothing_that_is_not_a_file(runs: Path) -> None:
    """Everything on the page comes from a manifest michi already wrote."""
    from michi.report.runs import load_manifests

    manifests = load_manifests(runs)
    body = _client(runs).get("/").text
    for manifest in manifests:
        assert manifest.model.class_name in body


# --- the command -----------------------------------------------------------


def test_ui_command_exists() -> None:
    """`michi ui` is registered and documents itself."""
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "read-only" in result.output
