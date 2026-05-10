from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = [
    "mean_abs_dist_m",
    "mean_abs_lat_m",
    "mean_abs_alt_m",
    "rmse_img",
    "visible_ratio",
    "reacq_ratio",
    "lost_episodes",
    "mean_recovery_s",
    "settling_time_s",
    "p95_cmd_yaw_dps",
]

LOWER_IS_BETTER = {
    "mean_abs_dist_m",
    "mean_abs_lat_m",
    "mean_abs_alt_m",
    "rmse_img",
    "reacq_ratio",
    "lost_episodes",
    "mean_recovery_s",
    "settling_time_s",
    "p95_cmd_yaw_dps",
}
HIGHER_IS_BETTER = {"visible_ratio"}

TEST_ORDER = [
    "repeated_runs",
    "detection_noise_sweep",
    "latency_sweep",
    "target_speed_sweep",
    "occlusion_duration_sweep",
    "full_frame_exit_set",
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def as_float(value, default=0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt(value, digits=3) -> str:
    if value in ("", None):
        return ""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "inf"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.{digits}f}"


def pct(pid: float, lqr: float) -> float:
    if abs(pid) < 1e-9:
        return 0.0 if abs(lqr) < 1e-9 else math.inf
    return (lqr - pid) / abs(pid) * 100.0


def metric_winner(metric: str, pid: float, lqr: float) -> str:
    if pid < 0 <= lqr:
        return "LQR"
    if lqr < 0 <= pid:
        return "PID"
    if abs(pid - lqr) < 1e-9:
        return "tie"
    if metric in HIGHER_IS_BETTER:
        return "LQR" if lqr > pid else "PID"
    return "LQR" if lqr < pid else "PID"


def overall_winner(pid_row: dict, lqr_row: dict) -> str:
    score = {"PID": 0, "LQR": 0}
    for metric in METRICS:
        pid = as_float(pid_row.get(f"{metric}_mean", ""))
        lqr = as_float(lqr_row.get(f"{metric}_mean", ""))
        winner = metric_winner(metric, pid, lqr)
        if winner in score:
            score[winner] += 1
    if score["PID"] == score["LQR"]:
        return "mixed"
    return "PID" if score["PID"] > score["LQR"] else "LQR"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |")
    return "\n".join(out)


def pair_rows(aggregate: list[dict], test: str) -> list[dict]:
    grouped = defaultdict(dict)
    for row in aggregate:
        if row["test"] != test:
            continue
        key = (row["scenario"], row["param_name"], row["param_value"])
        grouped[key][row["controller"]] = row
    out = []
    for key, pair in sorted(grouped.items(), key=lambda item: (item[0][0], as_float(item[0][2]))):
        pid = pair.get("pid")
        lqr = pair.get("lqr")
        if not pid or not lqr:
            only = pid or lqr
            out.append(
                {
                    "scenario": key[0],
                    "param": f"{key[1]}={key[2]}",
                    "status": "missing_pair",
                    "controller_present": only["controller"] if only else "",
                }
            )
            continue
        pid_rmse = as_float(pid["rmse_img_mean"])
        lqr_rmse = as_float(lqr["rmse_img_mean"])
        pid_dist = as_float(pid["mean_abs_dist_m_mean"])
        lqr_dist = as_float(lqr["mean_abs_dist_m_mean"])
        pid_vis = as_float(pid["visible_ratio_mean"])
        lqr_vis = as_float(lqr["visible_ratio_mean"])
        pid_yaw = as_float(pid["p95_cmd_yaw_dps_mean"])
        lqr_yaw = as_float(lqr["p95_cmd_yaw_dps_mean"])
        pid_lost = as_float(pid["lost_episodes_mean"])
        lqr_lost = as_float(lqr["lost_episodes_mean"])
        out.append(
            {
                "scenario": key[0],
                "param": f"{key[1]}={key[2]}",
                "status": "ok",
                "n_pid": pid["n_runs"],
                "n_lqr": lqr["n_runs"],
                "pid_dist": pid_dist,
                "lqr_dist": lqr_dist,
                "dist_delta_pct": pct(pid_dist, lqr_dist),
                "pid_rmse": pid_rmse,
                "lqr_rmse": lqr_rmse,
                "rmse_delta_pct": pct(pid_rmse, lqr_rmse),
                "pid_visible": pid_vis,
                "lqr_visible": lqr_vis,
                "visible_delta": lqr_vis - pid_vis,
                "pid_lost": pid_lost,
                "lqr_lost": lqr_lost,
                "pid_yaw": pid_yaw,
                "lqr_yaw": lqr_yaw,
                "yaw_delta_pct": pct(pid_yaw, lqr_yaw),
                "winner": overall_winner(pid, lqr),
            }
        )
    return out


def compact_table_for_test(aggregate: list[dict], test: str) -> str:
    rows = []
    for row in pair_rows(aggregate, test):
        if row["status"] != "ok":
            rows.append([row["scenario"], row["param"], row["status"], row.get("controller_present", ""), "", "", "", "", "", "", ""])
            continue
        rows.append(
            [
                row["scenario"],
                row["param"],
                f"{row['n_pid']}/{row['n_lqr']}",
                fmt(row["pid_dist"]),
                fmt(row["lqr_dist"]),
                fmt(row["dist_delta_pct"], 1) + "%",
                fmt(row["pid_rmse"]),
                fmt(row["lqr_rmse"]),
                fmt(row["rmse_delta_pct"], 1) + "%",
                fmt(row["pid_visible"]) + " / " + fmt(row["lqr_visible"]),
                fmt(row["pid_yaw"]) + " / " + fmt(row["lqr_yaw"]),
                row["winner"],
            ]
        )
    return md_table(
        [
            "scenario",
            "parameter",
            "n PID/LQR",
            "PID dist",
            "LQR dist",
            "dist delta",
            "PID img RMSE",
            "LQR img RMSE",
            "img delta",
            "visible PID/LQR",
            "yaw p95 PID/LQR",
            "overall",
        ],
        rows,
    )


def raw_aggregate_table(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["test"],
                row["controller"],
                row["scenario"],
                f"{row['param_name']}={row['param_value']}",
                row["n_runs"],
                fmt(row["mean_abs_dist_m_mean"]),
                fmt(row["rmse_img_mean"]),
                fmt(row["visible_ratio_mean"]),
                fmt(row["lost_episodes_mean"]),
                fmt(row["mean_recovery_s_mean"]),
                fmt(row["p95_cmd_yaw_dps_mean"]),
            ]
        )
    return md_table(
        ["test", "ctrl", "scenario", "param", "n", "dist", "rmse_img", "visible", "lost", "recovery", "yaw_p95"],
        table_rows,
    )


def build_summary(args) -> str:
    aggregate = read_csv(args.aggregate)
    run_summary = read_csv(args.run_summary)
    manifest = read_csv(args.manifest)
    lines: list[str] = []

    lines.append("# AirSim Extended Metrics Summary For ChatGPT")
    lines.append("")
    lines.append("Цей файл є одним зведеним пакетом для аналізу результатів в іншому чаті ChatGPT. Він містить методику експериментів, пояснення метрик, інвентаризацію CSV та компактні таблиці PID vs LQR по кожному типу live AirSim-тесту.")
    lines.append("")
    lines.append("## Project Context")
    lines.append("")
    lines.append("- Система: візуальне супроводження цілі БПЛА в AirSim/PX4.")
    lines.append("- Pipeline: AirSim RGB camera -> YOLO detector -> optional ByteTrack -> PID або LQR controller -> MAVSDK offboard command -> PX4 SITL -> рух дрона.")
    lines.append("- Ціль: куб `BP_TargetCube_5` у AirSim.")
    lines.append("- Основне порівняння: PID vs LQR за однакових стартових умов.")
    lines.append("- Стартові позиції за замовчуванням: drone `0,0,0,0`, cube `22,0,-3.4,0`, формат `x,y,z,yaw_deg` у AirSim NED.")
    lines.append("- Перед кожним тестом і після кожного тесту `run_tracking.py` повертає дрон і куб у стартові позиції через `simSetVehiclePose` та `simSetObjectPose`.")
    lines.append("")

    lines.append("## Files Included")
    lines.append("")
    lines.append(md_table(
        ["file", "purpose"],
        [
            [str(args.manifest), "manifest усіх live-запусків"],
            [str(args.run_summary), "один рядок на кожен run після агрегації"],
            [str(args.aggregate), "головна агрегована таблиця mean/std/CI95"],
            ["metrics_results/airsim_extended/by_test/*.csv", "об'єднані покадрові CSV окремо для кожного типу тесту"],
            ["metrics_results/airsim_extended/AIRSIM_EXTENDED_TEST_PLAN.md", "детальний план тестів"],
        ],
    ))
    lines.append("")

    manifest_counts = defaultdict(int)
    for row in manifest:
        manifest_counts[row.get("test", "unknown")] += 1
    lines.append("## Dataset Inventory")
    lines.append("")
    lines.append(md_table(["test_type", "number_of_runs_in_manifest"], [[k, str(manifest_counts[k])] for k in sorted(manifest_counts)]))
    lines.append("")
    lines.append(f"Total manifest runs: `{len(manifest)}`. Aggregated rows: `{len(aggregate)}`. Run-summary rows: `{len(run_summary)}`.")
    lines.append("")

    lines.append("## Metrics Dictionary")
    lines.append("")
    lines.append(md_table(
        ["metric", "meaning", "better"],
        [
            ["mean_abs_dist_m", "середня абсолютна похибка дистанції до цілі, м", "lower"],
            ["mean_abs_lat_m", "середня абсолютна бокова похибка, м", "lower"],
            ["mean_abs_alt_m", "середня абсолютна вертикальна похибка, м", "lower"],
            ["rmse_img", "RMSE нормованої похибки центра bbox у кадрі", "lower"],
            ["visible_ratio", "частка часу, коли YOLO бачить ціль", "higher"],
            ["reacq_ratio", "частка часу, коли працює reacquisition-режим", "lower за умови схожої visible_ratio"],
            ["lost_episodes", "кількість переходів visible -> lost", "lower"],
            ["mean_recovery_s", "середній час повернення detection після втрати", "lower"],
            ["settling_time_s", "перший час входу у стабільну зону похибок", "lower"],
            ["p95_cmd_yaw_dps", "95-й перцентиль yaw-команди, deg/s", "lower за умови схожого tracking quality"],
        ],
    ))
    lines.append("")
    lines.append("Note: якщо `n_runs=1`, стандартне відхилення та CI95 дорівнюють 0. Для статистично сильнішого висновку repeated runs треба запускати з `--repeats 5` або більше.")
    lines.append("")

    test_descriptions = {
        "repeated_runs": "Повторення базових сценаріїв для PID і LQR. Перевіряє загальну стабільність контролерів.",
        "detection_noise_sweep": "До bbox додається Gaussian noise. Перевіряє стійкість до шумного detection.",
        "latency_sweep": "Керування отримує затримане detection/pose. Перевіряє вплив inference/command latency.",
        "target_speed_sweep": "Швидкість цілі масштабується через speed_scale. Перевіряє межу line-of-sight.",
        "occlusion_duration_sweep": "Detection примусово вимикається на різний час. Перевіряє PredictiveReacq.",
        "full_frame_exit_set": "Ціль виходить за кадр у різні сторони та діагоналі. Перевіряє return-to-frame logic.",
    }

    lines.append("## PID vs LQR Compact Tables By Test")
    lines.append("")
    for test in TEST_ORDER:
        if not any(row["test"] == test for row in aggregate):
            continue
        lines.append(f"### {test}")
        lines.append("")
        lines.append(test_descriptions.get(test, ""))
        lines.append("")
        lines.append(compact_table_for_test(aggregate, test))
        lines.append("")

    lines.append("## Raw Aggregate Table")
    lines.append("")
    lines.append("Ця таблиця містить усі агреговані рядки без PID-vs-LQR зведення. Її можна використовувати для побудови власних графіків.")
    lines.append("")
    lines.append(raw_aggregate_table(aggregate))
    lines.append("")

    lines.append("## Suggested Thesis Interpretation Prompts")
    lines.append("")
    lines.append("Можна дати іншому ChatGPT такий запит:")
    lines.append("")
    lines.append("> На основі цього файлу сформуй академічний аналіз результатів AirSim-експериментів для бакалаврської роботи. Порівняй PID та LQR по кожному сценарію, поясни trade-off між image RMSE, visible ratio, yaw smoothness, lost episodes і recovery time. Окремо виділи, де LQR кращий, де PID кращий, і які обмеження має експеримент, якщо n_runs=1.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", type=Path, default=Path("metrics_results") / "airsim_extended_compare" / "airsim_extended_aggregate_summary.csv")
    ap.add_argument("--run_summary", type=Path, default=Path("metrics_results") / "airsim_extended_compare" / "airsim_extended_run_summary.csv")
    ap.add_argument("--manifest", type=Path, default=Path("metrics_results") / "airsim_extended" / "suite_manifest.csv")
    ap.add_argument("--out", type=Path, default=Path("metrics_results") / "airsim_extended" / "AIRSIM_EXTENDED_CHATGPT_SUMMARY.md")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_summary(args), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
