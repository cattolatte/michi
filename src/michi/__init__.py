"""Michi (道) — a local-first ML workbench.

The top-level package deliberately stays minimal so that ``import michi`` is
cheap and requires no optional dependencies: only the version lives here.
All functionality is in subpackages (``michi.core``, ``michi.inspection``,
``michi.evaluation``, …), each of which imports heavy dependencies lazily.

Design Principles
-----------------
- ``import michi`` never pulls in pandas, sklearn, or plotting libraries.
- The public API of each subpackage is its ``__all__``; anything else is
  private and may change without notice.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
