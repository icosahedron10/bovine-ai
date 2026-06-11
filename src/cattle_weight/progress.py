from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

LOGGER_NAME = "cattle_weight"

T = TypeVar("T")


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    root.setLevel(numeric_level)
    logging.getLogger(LOGGER_NAME).setLevel(numeric_level)


def logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


@dataclass
class ProgressLogger:
    stage: str
    total: int
    log_every: int = 25
    started: float = field(default_factory=time.perf_counter)
    last_logged: int = 0

    def log(
        self,
        completed: int,
        *,
        success: int = 0,
        failed: int = 0,
        skipped: int = 0,
        current: str | None = None,
        extra: str | None = None,
        force: bool = False,
    ) -> None:
        if self.total <= 0:
            return
        if not force and completed < self.total:
            interval = max(1, int(self.log_every or 1))
            if completed - self.last_logged < interval:
                return
        self.last_logged = completed
        elapsed = time.perf_counter() - self.started
        rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
        remaining = max(0, self.total - completed)
        eta = remaining / rate if rate > 0 else None
        pct = (completed / self.total) * 100.0
        parts = [
            f"{self.stage}: {completed}/{self.total} ({pct:.1f}%)",
            f"success={success}",
            f"failed={failed}",
            f"skipped={skipped}",
            f"elapsed={format_duration(elapsed)}",
        ]
        if rate > 0:
            parts.append(f"rate={rate:.3f}/s")
        if eta is not None:
            parts.append(f"eta={format_duration(eta)}")
        if current:
            parts.append(f"current={current}")
        if extra:
            parts.append(extra)
        logger().info(" | ".join(parts))


def log_stage_start(stage: str, **details: Any) -> float:
    parts = [f"Starting {stage}"]
    parts.extend(f"{key}={value}" for key, value in details.items() if value is not None)
    logger().info(" | ".join(parts))
    return time.perf_counter()


def log_stage_end(stage: str, started: float, **details: Any) -> None:
    elapsed = time.perf_counter() - started
    parts = [f"Finished {stage}", f"elapsed={format_duration(elapsed)}"]
    parts.extend(f"{key}={value}" for key, value in details.items() if value is not None)
    logger().info(" | ".join(parts))


def run_logged_stage(stage: str, func: Callable[[], T], **details: Any) -> T:
    started = log_stage_start(stage, **details)
    try:
        result = func()
    except Exception:
        logger().exception("Failed %s after %s", stage, format_duration(time.perf_counter() - started))
        raise
    log_stage_end(stage, started)
    return result


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
