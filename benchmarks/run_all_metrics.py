"""Convenience launcher for all offline benchmark scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str):
    cmd = [sys.executable, str(ROOT / "benchmarks" / script)]
    print("$ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    run("run_controller_metrics.py")
    run("run_yolo_tracker_metrics.py")
    run("run_reacquisition_metrics.py")
    run("run_distance_depth_metrics.py")
    run("run_extended_stability_metrics.py")
    print("All metric reports are in metrics_results/")


if __name__ == "__main__":
    main()
