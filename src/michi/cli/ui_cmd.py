"""The ``michi ui`` command.

Design Principles
-----------------
- Binds to localhost only. michi does not serve anything to a network, and
  the flag to change that does not exist.
- The viewer is a convenience over files. If the ``ui`` extra is missing,
  michi says so and points at ``michi report``, which shows the same thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from michi.core.errors import MichiError

__all__ = ["ui_command"]


def ui_command(
    runs_dir: Annotated[Path, typer.Argument(help="Runs directory to view.")] = Path(
        "runs"
    ),
    port: Annotated[int, typer.Option("--port", help="Port to serve on.")] = 8731,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Do not open a browser.")
    ] = False,
) -> None:
    """Browse recorded runs in a local, read-only web view.

    Serves on localhost only. Nothing is uploaded, nothing is stored beyond
    the run manifests already on disk, and no page can change anything —
    everything shown is also available from 'michi report'.
    """
    console = Console()
    try:
        import uvicorn

        from michi.ui import build_app

        app = build_app(runs_dir)
    except ImportError as err:
        Console(stderr=True).print(
            "[bold red]error[/] the local viewer requires the ui extra: "
            "pip install 'michi[ui]'. Everything it shows is also available "
            "from `michi report`."
        )
        raise typer.Exit(code=2) from err
    except MichiError as err:
        Console(stderr=True).print(f"[bold red]error[/] {err}")
        raise typer.Exit(code=2) from err

    url = f"http://127.0.0.1:{port}"
    console.print()
    console.print(f"  [bold red]道[/] [bold]michi ui[/]  ·  {url}")
    console.print(f"  [dim]read-only view of {runs_dir} · Ctrl-C to stop[/]\n")

    if not no_open:
        import webbrowser

        webbrowser.open(url)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
