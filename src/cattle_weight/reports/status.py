from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cattle_weight.io import reports_dir, write_json


def write_headline_reproduction_status(
    config: dict[str, Any],
    status: str,
    stage: str,
    reason: str | None,
    metrics_allowed: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.get("run", {}).get("run_id", "run"),
        "config_hash": config.get("_config_hash"),
        "status": status,
        "stage": stage,
        "reason": reason,
        "metrics_allowed": metrics_allowed,
        "required_segmentation_source": "pvtv2_b2_frozen_saliency",
        "accepted_depth_provider": "depth_pro",
        "segmentation_model": config.get("segmentation", {}).get("frozen_pvtv2", {}).get("model_name", "pvt_v2_b2"),
        "segmentation_weights": "public_pretrained",
    }
    if details is not None:
        payload["details"] = details
    report_dir = reports_dir(config)
    write_json(payload, report_dir / "headline_reproduction_status.json")
    write_headline_reproduction_status_markdown(payload, report_dir / "headline_reproduction_status.md")
    return payload


def write_headline_reproduction_status_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Headline Reproduction Status",
        "",
        f"- Status: {payload['status']}",
        f"- Stage: {payload['stage']}",
        f"- Metrics allowed: {payload['metrics_allowed']}",
        f"- Required segmentation source: {payload['required_segmentation_source']}",
        f"- Segmentation model: {payload['segmentation_model']}",
        f"- Segmentation weights: {payload['segmentation_weights']}",
        f"- Accepted depth provider: {payload['accepted_depth_provider']}",
    ]
    if payload.get("reason"):
        lines.extend(["", "Reason:", "", str(payload["reason"])])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
