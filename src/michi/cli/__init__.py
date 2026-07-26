"""Command-line interface for michi.

Design Principles
-----------------
- Thin dispatch only: argument parsing and rendering. Domain logic lives in
  the domain packages and is imported lazily per command so that unrelated
  verbs never pay each other's import cost.
"""

from __future__ import annotations

__all__: list[str] = []
