"""Canonical content hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible content."""
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
