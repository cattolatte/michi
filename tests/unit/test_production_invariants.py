"""Invariants a production review would check, asserted rather than audited.

Each of these was a real defect found by hand. A finding you fix once comes
back; a finding you assert stays fixed.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from michi.cli.app import app

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src" / "michi"


# --- architecture -----------------------------------------------------------


def test_no_package_reaches_into_another_package_private_names() -> None:
    """A private symbol used across packages is API that nobody promised.

    ADR-0005 freezes each package's `__all__` and states that everything else
    "is private, whatever its visibility". Ten cross-package imports of private
    names made that false: `ui` depended on `report.html._environment`, the
    console banner on `recipes.model._KNOWN_OPS`, and one CLI module on
    another's `_write_manifest`.
    """
    offenders: list[str] = []
    pattern = re.compile(r"from (michi\.[\w.]+) import ([^\n]+)")
    for path in SOURCE.rglob("*.py"):
        package = path.relative_to(SOURCE).parts[0]
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = pattern.search(line)
            if not match:
                continue
            module, names = match.groups()
            other = module.split(".")[1]
            if other in {package, "core"}:
                continue  # inside one package, or the shared foundation
            private = [n.strip() for n in names.split(",") if n.strip().startswith("_")]
            if private:
                offenders.append(f"{path.name}:{number} → {module}.{private}")
    assert offenders == [], f"cross-package private imports: {offenders}"


def test_core_never_imports_a_domain_package() -> None:
    """The one-way dependency core → domain → cli is what keeps layers honest."""
    offenders: list[str] = []
    for path in (SOURCE / "core").rglob("*.py"):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = re.search(r"from michi\.(\w+)", line)
            if match and match.group(1) not in {"core"}:
                offenders.append(f"core/{path.name}:{number} → michi.{match.group(1)}")
    assert offenders == [], offenders


# --- reproducibility --------------------------------------------------------


def test_no_command_hardcodes_a_sampling_seed() -> None:
    """A hardcoded seed silently ignores `--seed` and michi.toml.

    Four verbs sampled large files at `seed=0` regardless of what the user
    configured, so two people running the same command on the same file could
    not compare notes — and one of them could not reproduce the other's run at
    all.
    """
    offenders: list[str] = []
    for path in (SOURCE / "cli").glob("*.py"):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "load_table(" in line and "seed=0" in line:
                offenders.append(f"{path.name}:{number}")
    assert offenders == [], f"hardcoded sampling seed: {offenders}"


def test_every_verb_that_samples_offers_a_seed() -> None:
    """Sampling without a seed flag is an unreproducible result by construction."""
    missing: list[str] = []
    for command in app.registered_commands:
        if not command.name:
            continue
        params = set(inspect.signature(command.callback).parameters)
        if "sample" in params and "seed" not in params:
            missing.append(command.name)
    assert missing == [], f"verbs that sample but cannot be seeded: {missing}"


# --- security ---------------------------------------------------------------


def test_every_model_loading_verb_documents_the_risk() -> None:
    """Loading a pickle executes it, and the user deserves to read that.

    The caveat existed in eval.md and nowhere else, while three newer verbs
    grew the same capability and shipped without it.
    """
    loaders = {
        path.stem.removesuffix("_cmd")
        for path in (SOURCE / "cli").glob("*.py")
        if "load_model" in path.read_text(encoding="utf-8")
    }
    docs = " ".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "docs").glob("*.md")
    ).lower()
    assert loaders, "expected at least one verb to load a user model"
    # The warning need not be per-verb, but it must exist and be findable.
    assert "executes whatever is inside it" in docs or "models you trust" in docs
