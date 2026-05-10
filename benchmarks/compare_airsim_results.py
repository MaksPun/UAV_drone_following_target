"""Compares PID and LQR results from the base live AirSim suite."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.analyze_airsim_metrics import load_rows, summarize_group
from benchmarks.metrics_lib import RESULTS_DIR, mean, write_csv, write_markdown_table


SCENARIOS = [
    "static",
    "lateral_sine",
    "depth_step",
    "vertical_step",
    "zigzag",
    "frame_exit_right",
    "frame_exit_left",
    "frame_exit_top",
    "frame_exit_bottom",
    "occlusion_like",
]

LOWER_IS_BETTER = {
    "mean_abs_dist_m",
    "mean_abs_lat_m",
    "mean_abs_alt_m",
    "rmse_img",
    "reacq_ratio",
    "lost_episodes",
    "mean_recovery_s",
    "id_switches",
    "settling_time_s",
    "p95_cmd_yaw_dps",
}
HIGHER_IS_BETTER = {"visible_ratio"}


def expand(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        matches = [Path(p) for p in glob.glob(str(path))]
        out.extend(matches or [path])
    return out


def group_runs(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get("label", ""),
            row.get("scenario", ""),
            row.get("controller", ""),
            row.get("bytetrack", ""),
            row.get("_source", ""),
        )
        grouped[key].append(row)
    return [summarize_group(items) for items in grouped.values()]


def aggregate_runs(run_summaries: list[dict]) -> dict[tuple[str, str], dict]:
    grouped = defaultdict(list)
    for row in run_summaries:
        scenario = row["scenario"]
        controller = row["controller"]
        if scenario not in SCENARIOS or controller not in ("pid", "lqr"):
            continue
        grouped[(scenario, controller)].append(row)

    aggregated = {}
    for key, rows in grouped.items():
        numeric_keys = [
            "duration_s",
            "mean_abs_dist_m",
            "mean_abs_lat_m",
            "mean_abs_alt_m",
            "rmse_img",
            "visible_ratio",
            "reacq_ratio",
            "lost_episodes",
            "mean_recovery_s",
            "p95_recovery_s",
            "id_switches",
            "settling_time_s",
            "p95_cmd_yaw_dps",
        ]
        agg = {"scenario": key[0], "controller": key[1], "n_runs": len(rows)}
        for metric in numeric_keys:
            vals = [float(r[metric]) for r in rows if float(r[metric]) >= 0 or metric not in {"mean_recovery_s", "p95_recovery_s", "settling_time_s"}]
            agg[metric] = mean(vals) if vals else -1.0
        aggregated[key] = agg
    return aggregated


def pct_change(pid_value: float, lqr_value: float) -> float:
    if abs(pid_value) < 1e-9:
        return 0.0 if abs(lqr_value) < 1e-9 else math.inf
    return (lqr_value - pid_value) / abs(pid_value) * 100.0


def winner(metric: str, pid_value: float, lqr_value: float) -> str:
    if pid_value < 0 and lqr_value < 0:
        return "n/a"
    if pid_value < 0:
        return "lqr"
    if lqr_value < 0:
        return "pid"
    if abs(pid_value - lqr_value) < 1e-9:
        return "tie"
    if metric in HIGHER_IS_BETTER:
        return "lqr" if lqr_value > pid_value else "pid"
    return "lqr" if lqr_value < pid_value else "pid"


def comparison_rows(aggregated: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
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
    for scenario in SCENARIOS:
        pid = aggregated.get((scenario, "pid"))
        lqr = aggregated.get((scenario, "lqr"))
        if not pid or not lqr:
            rows.append({
                "scenario": scenario,
                "status": "missing_pair",
                "pid_runs": pid["n_runs"] if pid else 0,
                "lqr_runs": lqr["n_runs"] if lqr else 0,
                "best_overall": "n/a",
                "summary": "missing PID or LQR run",
            })
            continue

        score = {"pid": 0, "lqr": 0}
        row = {
            "scenario": scenario,
            "status": "ok",
            "pid_runs": pid["n_runs"],
            "lqr_runs": lqr["n_runs"],
        }
        for metric in metrics:
            pv = float(pid[metric])
            lv = float(lqr[metric])
            w = winner(metric, pv, lv)
            if w in score:
                score[w] += 1
            row[f"pid_{metric}"] = pv
            row[f"lqr_{metric}"] = lv
            row[f"delta_{metric}_pct"] = pct_change(pv, lv)
            row[f"winner_{metric}"] = w

        if score["pid"] == score["lqr"]:
            best = "mixed"
        else:
            best = "pid" if score["pid"] > score["lqr"] else "lqr"
        row["best_overall"] = best
        row["summary"] = describe_scenario(row)
        rows.append(row)
    return rows


def compact_rows(rows: list[dict]) -> list[dict]:
    out = []
    keys = [
        "scenario",
        "status",
        "pid_runs",
        "lqr_runs",
        "best_overall",
        "pid_dist",
        "lqr_dist",
        "dist_change_pct",
        "pid_img",
        "lqr_img",
        "img_change_pct",
        "pid_visible",
        "lqr_visible",
        "pid_lost",
        "lqr_lost",
        "pid_reacq",
        "lqr_reacq",
        "pid_yaw_p95",
        "lqr_yaw_p95",
        "summary",
    ]
    for row in rows:
        if row["status"] != "ok":
            missing = {key: "" for key in keys}
            missing.update({
                "scenario": row["scenario"],
                "status": row["status"],
                "pid_runs": row["pid_runs"],
                "lqr_runs": row["lqr_runs"],
                "best_overall": "n/a",
                "summary": row["summary"],
            })
            out.append(missing)
            continue
        out.append({
            "scenario": row["scenario"],
            "status": row["status"],
            "pid_runs": row["pid_runs"],
            "lqr_runs": row["lqr_runs"],
            "best_overall": row["best_overall"],
            "pid_dist": row["pid_mean_abs_dist_m"],
            "lqr_dist": row["lqr_mean_abs_dist_m"],
            "dist_change_pct": row["delta_mean_abs_dist_m_pct"],
            "pid_img": row["pid_rmse_img"],
            "lqr_img": row["lqr_rmse_img"],
            "img_change_pct": row["delta_rmse_img_pct"],
            "pid_visible": row["pid_visible_ratio"],
            "lqr_visible": row["lqr_visible_ratio"],
            "pid_lost": row["pid_lost_episodes"],
            "lqr_lost": row["lqr_lost_episodes"],
            "pid_reacq": row["pid_reacq_ratio"],
            "lqr_reacq": row["lqr_reacq_ratio"],
            "pid_yaw_p95": row["pid_p95_cmd_yaw_dps"],
            "lqr_yaw_p95": row["lqr_p95_cmd_yaw_dps"],
            "summary": row["summary"],
        })
    return out


def describe_scenario(row: dict) -> str:
    parts = []
    dist = row["delta_mean_abs_dist_m_pct"]
    img = row["delta_rmse_img_pct"]
    yaw = row["delta_p95_cmd_yaw_dps_pct"]
    vis_delta = row["lqr_visible_ratio"] - row["pid_visible_ratio"]
    if math.isfinite(dist):
        parts.append(f"distance {'improved' if dist < 0 else 'worsened'} by {abs(dist):.1f}% on LQR vs PID")
    if math.isfinite(img):
        parts.append(f"image RMSE {'improved' if img < 0 else 'worsened'} by {abs(img):.1f}%")
    parts.append(f"visibility delta {vis_delta:+.3f}")
    if math.isfinite(yaw):
        parts.append(f"yaw p95 {'lower' if yaw < 0 else 'higher'} by {abs(yaw):.1f}%")
    return "; ".join(parts)


def write_narrative(path: Path, rows: list[dict]) -> None:
    lines = ["# AirSim PID vs LQR comparison", ""]
    ok_rows = [r for r in rows if r["status"] == "ok"]
    missing = [r for r in rows if r["status"] != "ok"]
    pid_wins = sum(1 for r in ok_rows if r["best_overall"] == "pid")
    lqr_wins = sum(1 for r in ok_rows if r["best_overall"] == "lqr")
    mixed = sum(1 for r in ok_rows if r["best_overall"] == "mixed")
    lines.append(f"Compared {len(ok_rows)} complete PID/LQR scenario pairs. PID wins: {pid_wins}; LQR wins: {lqr_wins}; mixed: {mixed}.")
    if missing:
        lines.append("Missing complete pairs: " + ", ".join(f"{r['scenario']} (pid={r['pid_runs']}, lqr={r['lqr_runs']})" for r in missing) + ".")
    lines.append("")
    lines.append("## Scenario notes")
    lines.append("")
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"- `{row['scenario']}`: missing complete PID/LQR pair, so no fair comparison yet.")
        else:
            lines.append(f"- `{row['scenario']}`: best overall `{row['best_overall']}`; {row['summary']}.")
    lines.append("")
    lines.append("## Reading the deltas")
    lines.append("")
    lines.append("Negative percentage is better for error metrics such as distance, image RMSE, reacquisition ratio, lost episodes, settling time, and yaw p95. Positive percentage is better only for visibility ratio.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path, help="AirSim metrics CSV files or glob patterns.")
    ap.add_argument("--out_dir", type=Path, default=RESULTS_DIR / "airsim_compare")
    args = ap.parse_args()

    paths = expand(args.logs)
    rows = load_rows(paths)
    summaries = group_runs(rows)
    aggregated = aggregate_runs(summaries)
    full = comparison_rows(aggregated)
    compact = compact_rows(full)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "airsim_run_summaries.csv", summaries)
    write_csv(args.out_dir / "airsim_controller_averages.csv", list(aggregated.values()))
    write_csv(args.out_dir / "airsim_pid_lqr_comparison_full.csv", full)
    write_csv(args.out_dir / "airsim_pid_lqr_comparison_compact.csv", compact)
    write_markdown_table(args.out_dir / "airsim_pid_lqr_comparison_compact.md", "AirSim PID vs LQR compact comparison", compact)
    write_narrative(args.out_dir / "airsim_pid_lqr_report.md", compact)
    print(f"Wrote {args.out_dir / 'airsim_pid_lqr_report.md'}")


if __name__ == "__main__":
    main()
