"""Aggregates live AirSim CSV logs into summary tables."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics_lib import RESULTS_DIR, mean, p95, rmse, write_csv, write_markdown_table


def f(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    required = {"t", "label", "scenario", "controller", "target_visible", "dist_err"}
    expanded = []
    for path in paths:
        matches = [Path(p) for p in glob.glob(str(path))]
        expanded.extend(matches or [path])
    for path in expanded:
        if not path.exists() or path.stat().st_size == 0:
            print(f"Skipping empty metrics file: {path}")
            continue
        with path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                print(f"Skipping non-metrics CSV: {path}")
                continue
            for row in reader:
                row["_source"] = str(path)
                rows.append(row)
    return rows


def group_key(row):
    return (
        row.get("label", ""),
        row.get("scenario", ""),
        row.get("controller", ""),
        row.get("bytetrack", ""),
        row.get("_source", ""),
    )


def summarize_group(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: f(r, "t"))
    visible = [f(r, "target_visible") for r in rows]
    visible_rows = [r for r in rows if f(r, "target_visible") > 0.5]
    reacq_rows = [r for r in rows if r.get("reacq_mode", "NONE") != "NONE"]

    lost_episodes = 0
    recoveries = []
    lost_start = None
    prev_visible = True
    for row in rows:
        is_visible = f(row, "target_visible") > 0.5
        t = f(row, "t")
        if prev_visible and not is_visible:
            lost_episodes += 1
            lost_start = t
        if not prev_visible and is_visible and lost_start is not None:
            recoveries.append(t - lost_start)
            lost_start = None
        prev_visible = is_visible

    id_switches = 0
    prev_id = ""
    for row in rows:
        track_id = row.get("track_id", "")
        if track_id and prev_id and track_id != prev_id:
            id_switches += 1
        if track_id:
            prev_id = track_id

    img_errors = []
    for row in visible_rows:
        ix = f(row, "img_x")
        iy = f(row, "img_y")
        img_errors.append(math.sqrt(ix * ix + iy * iy))

    first = rows[0]
    settled = -1.0
    for row in rows:
        if abs(f(row, "dist_err")) < 0.75 and abs(f(row, "lat_err")) < 0.45 and abs(f(row, "alt_err")) < 0.35:
            settled = f(row, "t")
            break

    return {
        "label": first.get("label", ""),
        "scenario": first.get("scenario", ""),
        "controller": first.get("controller", ""),
        "bytetrack": first.get("bytetrack", ""),
        "duration_s": max(f(r, "t") for r in rows) if rows else 0.0,
        "mean_abs_dist_m": mean(abs(f(r, "dist_err")) for r in rows),
        "mean_abs_lat_m": mean(abs(f(r, "lat_err")) for r in rows),
        "mean_abs_alt_m": mean(abs(f(r, "alt_err")) for r in rows),
        "rmse_img": rmse(img_errors),
        "visible_ratio": mean(visible),
        "reacq_ratio": len(reacq_rows) / max(1, len(rows)),
        "lost_episodes": lost_episodes,
        "mean_recovery_s": mean(recoveries) if recoveries else -1.0,
        "p95_recovery_s": p95(recoveries) if recoveries else -1.0,
        "id_switches": id_switches,
        "settling_time_s": settled,
        "p95_cmd_yaw_dps": p95(abs(f(r, "cmd_yaw")) for r in rows),
        "det_noise_px": f(first, "det_noise_px"),
        "latency_ms": f(first, "latency_ms"),
        "scenario_speed_scale": f(first, "scenario_speed_scale", 1.0),
        "forced_occlusion_s": f(first, "forced_occlusion_s"),
        "source": first.get("_source", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path, help="One or more AirSim metrics CSV files.")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "airsim_summary")
    args = ap.parse_args()

    rows = load_rows(args.logs)
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    summary = [summarize_group(group_rows) for group_rows in grouped.values()]
    csv_path = args.out.with_suffix(".csv")
    md_path = args.out.with_suffix(".md")
    write_csv(csv_path, summary)
    write_markdown_table(md_path, "AirSim live tracking metrics", summary)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
