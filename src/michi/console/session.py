"""Console state: the context every command inherits.

Design Principles
-----------------
- **The console holds no state the file system cannot.** Context *is* the
  ``michi.toml`` model, held in memory. ``set`` edits it, ``save`` writes it,
  and nothing dies silently when the session ends.
- **State is visible at all times.** The prompt shows the dataset and target,
  and ``show context`` prints everything with where each value came from.
  A console that hides its state produces "why did that happen?" bugs.
- **Every session is exportable.** The history of commands converts to a
  script of one-shot invocations, so exploration stays reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from michi.core.config import CONFIG_FILENAME, ProjectDefaults, load_defaults

__all__ = ["Session"]


@dataclass(slots=True)
class Session:
    """One console session's context and history.

    The fields mirror ``michi.toml`` exactly, because they are the same model:
    what the console remembers is what the file would record.
    """

    data: str | None = None
    target: str | None = None
    recipe: str | None = None
    runs_dir: str = "runs"
    models: str = "linear,rf,hist-gbm"
    seed: int = 0
    cv: int = 5
    columns: tuple[str, ...] = ()
    history: list[str] = field(default_factory=list)
    dirty: bool = False

    @classmethod
    def from_defaults(cls, defaults: ProjectDefaults | None = None) -> Session:
        """Build a session seeded from ``michi.toml``, if one exists."""
        resolved = defaults if defaults is not None else load_defaults()
        session = cls(
            data=resolved.data,
            target=resolved.target,
            recipe=resolved.recipe,
            runs_dir=resolved.runs_dir or "runs",
            models=resolved.models or "linear,rf,hist-gbm",
            seed=resolved.seed if resolved.seed is not None else 0,
            cv=resolved.cv if resolved.cv is not None else 5,
        )
        if session.data:
            session.load_columns()
        return session

    @property
    def prompt(self) -> str:
        """The prompt string, which always shows the current context."""
        if self.data is None:
            return "michi › "
        name = Path(self.data).name
        if self.target is None:
            return f"michi ({name}) › "
        return f"michi ({name} → {self.target}) › "

    def as_defaults(self) -> ProjectDefaults:
        """Render the context as project defaults, ready to write."""
        return ProjectDefaults(
            data=self.data,
            target=self.target,
            recipe=self.recipe,
            runs_dir=self.runs_dir,
            models=self.models,
            seed=self.seed,
            cv=self.cv,
        )

    def settings(self) -> dict[str, object]:
        """Every settable value, for ``show context``."""
        return {
            "data": self.data,
            "target": self.target,
            "recipe": self.recipe,
            "runs_dir": self.runs_dir,
            "models": self.models,
            "seed": self.seed,
            "cv": self.cv,
        }

    def load_columns(self) -> str | None:
        """Read the dataset's column names, for tab completion.

        Returns a description of the loaded data, or an error message. Only
        the header is needed, so this stays fast even on a very large file.
        """
        if self.data is None:
            return None
        path = Path(self.data)
        if not path.exists():
            self.columns = ()
            return f"no such file: {path}"

        try:
            import pandas as pd

            if path.suffix.lower() in {".parquet", ".pq"}:
                import pyarrow.parquet as pq

                parquet = pq.ParquetFile(str(path))
                self.columns = tuple(str(name) for name in parquet.schema.names)
                rows = int(parquet.metadata.num_rows)
            else:
                head = pd.read_csv(path, nrows=200)
                self.columns = tuple(str(name) for name in head.columns)
                rows = -1
        except Exception as err:  # third-party failure boundary
            self.columns = ()
            return f"could not read {path.name}: {err}"

        shape = f"{rows:,} rows × " if rows >= 0 else ""
        return f"loaded {path.name} — {shape}{len(self.columns)} columns"

    def record(self, command: str) -> None:
        """Add a command to the session history."""
        cleaned = command.strip()
        if cleaned:
            self.history.append(cleaned)

    def save(self, destination: Path | None = None) -> Path:
        """Write the context to ``michi.toml``."""
        path = destination or Path(CONFIG_FILENAME)
        self.as_defaults().write(path)
        self.dirty = False
        return path
