"""Content hashing for reproducibility.

Every artifact records the hash of the data it was derived from, so a profile,
run, or report can be traced back to exact bytes rather than a file name that
may have been overwritten.

Design Principles
-----------------
- SHA-256 everywhere: one algorithm, no configuration, no negotiation.
- Streaming reads with a bounded buffer, so hashing a 50 GB file costs
  constant memory.
- Hashes are of *bytes*, never of parsed in-memory objects, so results do not
  depend on the pandas or pyarrow version that happened to be installed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["hash_file", "hash_payload"]

_CHUNK_BYTES = 1024 * 1024


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents.

    Parameters
    ----------
    path
        File to hash. Read in binary mode in bounded chunks.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     target = pathlib.Path(tmp) / "d.txt"
    ...     _ = target.write_bytes(b"michi")
    ...     hash_file(target)[:8]
    'b3b4a0c0'
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(payload: Any) -> str:
    """Return a stable SHA-256 digest of any JSON-serialisable payload.

    Keys are sorted so that logically identical payloads hash identically
    regardless of construction order.

    Examples
    --------
    >>> hash_payload({"b": 1, "a": 2}) == hash_payload({"a": 2, "b": 1})
    True
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
