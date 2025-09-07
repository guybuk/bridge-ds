from __future__ import annotations

from typing import Any, Dict, List


def validate_metadata(keys: List[str], metadata: Dict[str, Any] | None):
    if metadata is None:
        return
    dup_keys = set(metadata.keys()) & set(keys)
    assert dup_keys == set(), f"Metadata cannot contain reserved keys={dup_keys}"
