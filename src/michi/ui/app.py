"""A local, read-only viewer over recorded runs.

Design Principles
-----------------
- **Read-only, always.** The viewer has no route that writes, deletes, or
  trains anything. It shows files. A UI that can act is a platform, and
  michi is not one.
- **No database, no build step, no network.** Every request reads the runs
  directory; pages are server-rendered HTML with inline CSS. There is no
  JavaScript to bundle and nothing to fetch from a CDN, so the viewer works
  on an air-gapped machine and cannot rot when a frontend toolchain moves on.
- **Deletable.** Removing this package would remove convenience and not a
  single capability: everything shown here exists as a file, and
  ``michi report`` renders the same artifacts.
- Binds to localhost only. michi does not serve anything to a network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from michi.core.errors import ReportError, install_hint

__all__ = ["build_app"]


def build_app(runs_dir: Path) -> Any:
    """Build the viewer application over a runs directory.

    Parameters
    ----------
    runs_dir
        Directory of run manifests, re-read on every request so the page is
        never stale.

    Returns
    -------
    Any
        A FastAPI application.

    Raises
    ------
    ReportError
        If the ``ui`` extra is not installed.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except ImportError as err:
        msg = (
            f"the local viewer requires the ui extra: {install_hint('ui')}. "
            "Everything it shows is also available from `michi report`."
        )
        raise ReportError(msg) from err

    from michi.report.html import _environment

    app = FastAPI(
        title="michi",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    environment = _environment()

    def _groups() -> tuple[Any, ...]:
        from michi.report.runs import group_runs, load_manifests

        try:
            return group_runs(load_manifests(runs_dir))
        except ReportError:
            return ()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """List every recorded run, grouped by dataset and target."""
        groups = _groups()
        template = environment.get_template("ui_index.html.jinja")
        return str(
            template.render(
                groups=groups,
                runs_dir=str(runs_dir),
                total=sum(len(group.manifests) for group in groups),
            )
        )

    @app.get("/run/{run_id}", response_class=HTMLResponse)
    def run_detail(run_id: str) -> str:
        """Show one run in full: metrics, baselines, checks, environment."""
        for group in _groups():
            for manifest in group.manifests:
                if manifest.run_id == run_id:
                    template = environment.get_template("ui_run.html.jinja")
                    return str(
                        template.render(
                            manifest=manifest,
                            group=group,
                            charts=_charts(manifest),
                        )
                    )
        template = environment.get_template("ui_missing.html.jinja")
        return str(template.render(run_id=run_id, runs_dir=str(runs_dir)))

    return app


def _charts(manifest: Any) -> dict[str, str | None]:
    """Draw what this run recorded, and nothing it did not.

    Charts are rendered from the manifest alone: a viewer that recomputed
    anything could disagree with the terminal, and then two of michi's
    surfaces would describe the same run differently. Every entry may be
    ``None``, which the template reads as "do not draw this".
    """
    from michi.report.charts import (
        calibration_chart,
        confusion_chart,
        importance_chart,
        interval_chart,
        slice_chart,
    )

    details = manifest.details or {}
    rows = [
        (metric.name, metric.value, metric.ci_low, metric.ci_high)
        for metric in manifest.metrics
    ]
    return {
        "intervals": interval_chart(rows) if rows else None,
        "confusion": confusion_chart(
            list(details.get("classes") or []), list(details.get("confusion") or [])
        ),
        "calibration": calibration_chart(details.get("calibration")),
        "slices": slice_chart(list(details.get("slices") or [])),
        "importance": importance_chart(list(details.get("importance") or [])),
    }
