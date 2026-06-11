from __future__ import annotations

from typing import Any


def row_is_reusable_by_stable_input(row: Any) -> bool:
    marker = row.get("reusable_by_stable_input_config", False)
    stable_hash = row.get("stable_input_config_hash")
    return bool(marker) and stable_hash not in {None, "", "nan"}
