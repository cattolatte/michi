"""Tests for experiment sweeps.

The properties that matter are that the grid is what the file says, that
caching is keyed by content, and that one bad cell never costs the good ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from michi.core.errors import RunError
from michi.sweep import SweepPlan, load_plan, run_sweep


def _write_plan(tmp_path: Path, data: Path, **overrides: object) -> Path:
    body = {
        "data": str(data),
        "target": "label",
        "folds": 3,
        "grid": {"models": ["linear", "tree"], "seeds": [0, 1]},
    }
    body.update(overrides)

    import yaml

    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


# --- the plan --------------------------------------------------------------


def test_grid_is_the_product_of_its_axes() -> None:
    """Cells are every combination of model, recipe, and seed."""
    plan = SweepPlan(
        data="d.csv",
        target="y",
        models=("a", "b", "c"),
        recipes=("r1.yaml", "r2.yaml"),
        seeds=(0, 1),
    )
    assert plan.size == 12


def test_grid_without_recipes_still_has_cells() -> None:
    """Recipes are optional; the grid degrades to models × seeds."""
    plan = SweepPlan(data="d.csv", target="y", models=("a",), seeds=(0, 1, 2))
    assert plan.size == 3
    assert all(cell.recipe is None for cell in plan.cells())


def test_cell_order_is_stable() -> None:
    """Two enumerations of the same grid agree, so progress is meaningful."""
    plan = SweepPlan(data="d.csv", target="y", models=("a", "b"), seeds=(0, 1))
    assert plan.cells() == plan.cells()


def test_a_plan_needs_models() -> None:
    """An empty grid is a mistake worth reporting."""
    with pytest.raises(RunError, match="at least one model"):
        SweepPlan(data="d.csv", target="y", models=())


def test_a_plan_needs_a_target() -> None:
    """Without a target there is nothing to predict."""
    with pytest.raises(RunError, match="target"):
        SweepPlan(data="d.csv", target="", models=("a",))


def test_plan_paths_resolve_relative_to_the_file(
    tmp_path: Path, tidy_csv: Path
) -> None:
    """A sweep can be run from anywhere, not only its own directory."""
    nested = tmp_path / "experiments"
    nested.mkdir()
    (nested / "data.csv").write_text(
        tidy_csv.read_text(encoding="utf-8"), encoding="utf-8"
    )
    path = nested / "sweep.yaml"
    path.write_text(
        "data: data.csv\ntarget: label\ngrid:\n  models: [linear]\n",
        encoding="utf-8",
    )
    assert Path(load_plan(path).data).exists()


def test_missing_plan_file_is_reported(tmp_path: Path) -> None:
    """A missing sweep file fails clearly."""
    with pytest.raises(RunError, match="no such sweep file"):
        load_plan(tmp_path / "absent.yaml")


def test_unparseable_plan_is_reported(tmp_path: Path) -> None:
    """A YAML error names the file the user typed into."""
    path = tmp_path / "sweep.yaml"
    path.write_text("grid: [unclosed", encoding="utf-8")
    with pytest.raises(RunError, match=r"sweep\.yaml"):
        load_plan(path)


# --- cell identity ---------------------------------------------------------


def test_cell_keys_depend_on_every_input() -> None:
    """Changing anything that affects the result changes the key."""
    from michi.sweep import SweepCell

    cell = SweepCell(model="rf", seed=0)
    base = cell.key(data_sha="a", folds=5)
    assert cell.key(data_sha="b", folds=5) != base
    assert cell.key(data_sha="a", folds=10) != base
    assert cell.key(data_sha="a", folds=5, recipe_sha="x") != base
    assert SweepCell(model="rf", seed=1).key(data_sha="a", folds=5) != base
    assert SweepCell(model="tree", seed=0).key(data_sha="a", folds=5) != base


def test_identical_cells_share_a_key() -> None:
    """The same inputs always produce the same key, so caching is sound."""
    from michi.sweep import SweepCell

    first = SweepCell(model="rf", seed=3).key(data_sha="a", folds=5)
    second = SweepCell(model="rf", seed=3).key(data_sha="a", folds=5)
    assert first == second


# --- running ---------------------------------------------------------------


def test_sweep_runs_every_cell(tidy_csv: Path, tmp_path: Path) -> None:
    """Each cell produces an outcome and a manifest on disk."""
    plan = load_plan(_write_plan(tmp_path, tidy_csv))
    result = run_sweep(plan, sweep_dir=tmp_path / "cells")

    assert len(result.outcomes) == 4
    assert result.counts()["ran"] == 4
    assert len(list((tmp_path / "cells").glob("*.json"))) == 4


def test_a_second_run_reuses_recorded_cells(tidy_csv: Path, tmp_path: Path) -> None:
    """An interrupted sweep resumes rather than starting over."""
    plan = load_plan(_write_plan(tmp_path, tidy_csv))
    cells = tmp_path / "cells"
    run_sweep(plan, sweep_dir=cells)
    again = run_sweep(plan, sweep_dir=cells)

    assert again.counts()["cached"] == 4
    assert again.counts()["ran"] == 0


def test_force_reruns_everything(tidy_csv: Path, tmp_path: Path) -> None:
    """`--force` ignores recorded results."""
    plan = load_plan(_write_plan(tmp_path, tidy_csv))
    cells = tmp_path / "cells"
    run_sweep(plan, sweep_dir=cells)
    forced = run_sweep(plan, sweep_dir=cells, force=True)
    assert forced.counts()["ran"] == 4


def test_changing_the_data_invalidates_the_cache(
    tidy_csv: Path, tmp_path: Path
) -> None:
    """Caching is by content: different bytes mean different cells."""
    import pandas as pd

    plan_path = _write_plan(tmp_path, tidy_csv)
    cells = tmp_path / "cells"
    run_sweep(load_plan(plan_path), sweep_dir=cells)

    changed = tmp_path / "changed.csv"
    frame = pd.read_csv(tidy_csv)
    frame.loc[0, "feature_a"] = 999
    frame.to_csv(changed, index=False)

    second = run_sweep(load_plan(_write_plan(tmp_path, changed)), sweep_dir=cells)
    assert second.counts()["ran"] == 4


def test_changing_the_folds_invalidates_the_cache(
    tidy_csv: Path, tmp_path: Path
) -> None:
    """A different cross-validation is a different experiment."""
    cells = tmp_path / "cells"
    run_sweep(load_plan(_write_plan(tmp_path, tidy_csv)), sweep_dir=cells)
    second = run_sweep(
        load_plan(_write_plan(tmp_path, tidy_csv, folds=4)), sweep_dir=cells
    )
    assert second.counts()["ran"] == 4


def test_one_failing_cell_does_not_stop_the_sweep(
    tidy_csv: Path, tmp_path: Path
) -> None:
    """A cell that cannot run is recorded; the rest of the grid continues."""
    plan = load_plan(
        _write_plan(
            tmp_path,
            tidy_csv,
            grid={"models": ["linear", "lasso"], "seeds": [0]},
        )
    )
    result = run_sweep(plan, sweep_dir=tmp_path / "cells")
    assert result.counts()["failed"] >= 1
    assert result.counts()["ran"] >= 1


def test_missing_data_is_reported(tmp_path: Path) -> None:
    """A sweep pointing at data that does not exist fails before any work."""
    plan = SweepPlan(data=str(tmp_path / "absent.csv"), target="y", models=("linear",))
    with pytest.raises(RunError, match="not found"):
        run_sweep(plan, sweep_dir=tmp_path / "cells")


# --- results ---------------------------------------------------------------


def test_cells_are_ranked_by_score(tidy_csv: Path, tmp_path: Path) -> None:
    """The best cell heads the leaderboard."""
    plan = load_plan(_write_plan(tmp_path, tidy_csv))
    result = run_sweep(plan, sweep_dir=tmp_path / "cells")
    ranked = result.ranked()
    scores = [item.score for item in ranked if item.score is not None]
    assert scores == sorted(scores, reverse=True)


def test_manifests_record_the_cell_and_the_plan(tidy_csv: Path, tmp_path: Path) -> None:
    """A cell's result is traceable to the grid that produced it."""
    plan = load_plan(_write_plan(tmp_path, tidy_csv))
    result = run_sweep(plan, sweep_dir=tmp_path / "cells")
    manifest = next(item.manifest for item in result.outcomes if item.manifest)

    assert manifest.kind == "sweep"
    assert manifest.details["sweep_cell"]["model"]
    assert manifest.details["sweep_plan"]["grid"]["models"]
    assert manifest.details["sweep_key"]


def test_sweep_manifests_are_readable_by_report(tidy_csv: Path, tmp_path: Path) -> None:
    """Sweep cells feed straight into `michi report`."""
    from michi.report.runs import group_runs, load_manifests

    plan = load_plan(_write_plan(tmp_path, tidy_csv))
    cells = tmp_path / "cells"
    run_sweep(plan, sweep_dir=cells)

    groups = group_runs(load_manifests(cells))
    assert groups
    assert groups[0].manifests


def test_progress_is_reported_for_every_cell(tidy_csv: Path, tmp_path: Path) -> None:
    """A long sweep must never look like a hang."""
    seen: list[str] = []
    plan = load_plan(_write_plan(tmp_path, tidy_csv))
    run_sweep(
        plan,
        sweep_dir=tmp_path / "cells",
        progress=lambda index, total, cell, status: seen.append(status),
    )
    assert len(seen) == 4
