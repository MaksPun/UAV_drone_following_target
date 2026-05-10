"""Compares PID and LQR results from the extended live AirSim suite."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.analyze_airsim_metrics import load_rows, summarize_group
from benchmarks.metrics_lib import RESULTS_DIR, mean, write_csv, write_markdown_table


def expand(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        matches = [Path(p) for p in glob.glob(str(path), recursive=True)]
        out.extend(matches or [path])
    return out


def load_manifest(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fp:
        return {row["label"]: row for row in csv.DictReader(fp)}


def group_key(row: dict):
    return (
        row.get("label", ""),
        row.get("scenario", ""),
        row.get("controller", ""),
        row.get("bytetrack", ""),
        row.get("_source", ""),
    )


def parse_label(label: str) -> dict:
    parts = label.split("_")
    if len(parts) < 6 or parts[0] != "air":
        return {"test": "unknown", "param_name": "none", "param_value": 0.0}
    controller_idx = next((i for i, p in enumerate(parts) if p in {"pid", "lqr"}), -1)
    if controller_idx <= 1:
        return {"test": "unknown", "param_name": "none", "param_value": 0.0}
    test = "_".join(parts[1:controller_idx])
    patterns = [
        (("none",), "none"),
        (("bbox", "noise", "px"), "bbox_noise_px"),
        (("latency", "ms"), "latency_ms"),
        (("speed", "scale"), "speed_scale"),
        (("occlusion", "duration", "s"), "occlusion_duration_s"),
        (("frame", "exit", "case"), "frame_exit_case"),
    ]
    name = "none"
    raw = "0"
    for pattern, candidate in patterns:
        n = len(pattern)
        for i in range(len(parts) - n + 1):
            if tuple(parts[i:i + n]) == pattern:
                name = candidate
                raw = parts[i + n] if candidate != "none" and len(parts) > i + n else "0"
                break
        if name == candidate:
            break
    try:
        value = float(raw.replace("p", ".").replace("m", "-"))
    except ValueError:
        value = 0.0
    return {"test": test, "param_name": name, "param_value": value}


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> float:
    return 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def aggregate(summaries: list[dict], manifest: dict[str, dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in summaries:
        meta = manifest.get(row["label"], {}) or parse_label(row["label"])
        test = meta.get("test", "unknown")
        param_name = meta.get("param_name", "none")
        try:
            param_value = float(meta.get("param_value", 0.0))
        except ValueError:
            param_value = 0.0
        key = (test, row["controller"], row["scenario"], param_name, param_value)
        grouped[key].append(row)

    metrics = [
        "mean_abs_dist_m",
        "mean_abs_lat_m",
        "mean_abs_alt_m",
        "rmse_img",
        "visible_ratio",
        "reacq_ratio",
        "lost_episodes",
        "mean_recovery_s",
        "id_switches",
        "settling_time_s",
        "p95_cmd_yaw_dps",
    ]
    out = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        agg = {
            "test": key[0],
            "controller": key[1],
            "scenario": key[2],
            "param_name": key[3],
            "param_value": key[4],
            "n_runs": len(rows),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in rows if float(r[metric]) >= 0 or metric not in {"mean_recovery_s", "settling_time_s"}]
            agg[f"{metric}_mean"] = mean(vals)
            agg[f"{metric}_std"] = stdev(vals)
            agg[f"{metric}_ci95"] = ci95(vals)
        out.append(agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path, help="AirSim extended metrics CSV files or glob patterns.")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=RESULTS_DIR / "airsim_extended_compare")
    args = ap.parse_args()

    paths = expand(args.logs)
    rows = load_rows(paths)
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    summaries = [summarize_group(items) for items in grouped.values()]
    manifest = load_manifest(args.manifest)
    aggregate_rows = aggregate(summaries, manifest)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "airsim_extended_run_summary.csv", summaries)
    write_csv(args.out_dir / "airsim_extended_aggregate_summary.csv", aggregate_rows)
    write_markdown_table(args.out_dir / "airsim_extended_aggregate_summary.md", "AirSim extended aggregate metrics", aggregate_rows)
    print(f"Wrote {args.out_dir / 'airsim_extended_aggregate_summary.md'}")


if __name__ == "__main__":
    main()
