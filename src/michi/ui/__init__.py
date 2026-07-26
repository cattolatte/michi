"""The local viewer — the ``michi ui`` verb.

A read-only, offline web view over the runs directory.

Design Principles
-----------------
- Read-only: no route writes, deletes, or trains anything.
- No database, no build step, no network: server-rendered HTML with inline
  CSS, reading files on every request.
- Deletable: everything it shows exists as a file, and ``michi report``
  renders the same artifacts.
"""

from __future__ import annotations

from michi.ui.app import build_app

__all__ = ["build_app"]
