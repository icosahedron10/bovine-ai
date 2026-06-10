from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cattle_weight.config import load_config
from cattle_weight.reports.depth import write_depth_completion_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a diagnostic Depth Pro completion summary CSV.")
    parser.add_argument("--config", default="configs/reproduction_b3_b4.yaml")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    summary = write_depth_completion_summary(config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
