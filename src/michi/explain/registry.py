"""Looking up the explanation attached to a finding.

Design Principles
-----------------
- **Content lives in data files, not in code.** Explanation text is edited and
  reviewed like documentation; this module only loads and validates it.
- **Explanations describe, they never recommend.** An entry states what an
  observation means and lists the options practitioners choose between. Which
  option is right is the user's judgement, not michi's.
- **A missing explanation is never fatal.** Findings render fine without one,
  so new detectors can ship before their prose is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = ["Explanation", "explanation_for", "explanations"]


@dataclass(frozen=True, slots=True)
class Explanation:
    """What a finding means, and the options it opens up.

    Examples
    --------
    >>> explanation_for("class-imbalance").title
    'Imbalanced classes'
    >>> len(explanation_for("class-imbalance").options) > 1
    True
    """

    kind: str
    title: str
    what: str
    options: tuple[str, ...] = ()
    caution: str | None = None


@lru_cache(maxsize=1)
def explanations() -> dict[str, Explanation]:
    """Load every explanation, keyed by finding kind.

    The content file is parsed once per process and cached.
    """
    from importlib.resources import files

    import yaml

    source = files("michi.explain.content").joinpath("findings.yaml")
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}

    loaded: dict[str, Explanation] = {}
    for kind, entry in payload.items():
        loaded[kind] = Explanation(
            kind=str(kind),
            title=str(entry["title"]).strip(),
            what=" ".join(str(entry["what"]).split()),
            options=tuple(
                " ".join(str(item).split()) for item in entry.get("options", ())
            ),
            caution=(
                " ".join(str(entry["caution"]).split())
                if entry.get("caution")
                else None
            ),
        )
    return loaded


def explanation_for(kind: str) -> Explanation | None:
    """Return the explanation for a finding kind, or ``None`` if unwritten.

    Examples
    --------
    >>> explanation_for("no-such-finding") is None
    True
    """
    return explanations().get(kind)
