"""Rebuilds the extended AirSim manifest from generated CSV logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TEST_NAMES = (
    "repeated_runs",
    "detection_noise_sweep",
    "latency_sweep",
    "target_speed_sweep",
    "occlusion_duration_sweep",
    "full_frame_exit_set",
)

PARAM_PATTERNS = {
    "none": re.compile(r"_none_(?P<value>[^_]+)_"),
    "bbox_noise_px": re.compile(r"_bbox_noise_px_(?P<value>[^_]+)_"),
    "latency_ms": re.compile(r"_latency_ms_(?P<value>[^_]+)_"),
    "speed_scale": re.compile(r"_speed_scale_(?P<value>[^_]+)_"),
    "occlusion_duration_s": re.compile(r"_occlusion_duration_s_(?P<value>[^_]+)_"),
    "frame_exit_case": re.compile(r"_frame_exit_case_(?P<value>[^_]+)_"),
}


def parse_value(text: str) -> float:
    try:
        return float(text.replace("p", ".").replace("m", "-"))
    except ValueError:
        return 0.0


def parse_log(path: Path) -> dict:
    label = path.stem
    test = next((name for name in TEST_NAMES if label.startswith(f"air_{name}_")), "unknown")
    tail = label[len(f"air_{test}_"):] if test != "unknown" else label
    parts = tail.split("_")
    controller = next((p for p in parts if p in {"pid", "lqr"}), "")
    controller_idx = parts.index(controller) if controller else -1

    param_name = "none"
    param_value = 0.0
    param_start = len(parts)
    for name, pattern in PARAM_PATTERNS.items():
        match = pattern.search("_" + tail + "_")
        if match:
            param_name = name
            param_value = parse_value(match.group("value"))
            param_start = tail[: match.start()].count("_")
            break

    scenario = "_".join(parts[controller_idx + 1:param_start]) if controller_idx >= 0 else ""
    repeat = 0
    repeat_match = re.search(r"_rep(?P<repeat>\d+)_", label)
    if repeat_match:
        repeat = int(repeat_match.group("repeat"))
    return {
        "label": label,
        "test": test,
        "controller": controller,
        "scenario": scenario,
        "repeat": repeat,
        "param_name": param_name,
        "param_value": param_value,
        "log_path": str(path),
        "status": "existing",
    }


def combine_by_test(out_dir: Path, rows: list[dict]) -> None:
    by_test_dir = out_dir / "by_test"
    by_test_dir.mkdir(parents=True, exist_ok=True)
    tests = sorted({row["test"] for row in rows if row["test"] != "unknown"})
    for test in tests:
        items = [row for row in rows if row["test"] == test]
        combined = []
        fieldnames = []
        for item in items:
            path = Path(item["log_path"])
            with path.open("r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                if not reader.fieldnames:
                    continue
                extra_fields = ["test_type", "test_param_name", "test_param_value", "repeat", "source_log"]
                for name in list(reader.fieldnames) + extra_fields:
                    if name not in fieldnames:
                        fieldnames.append(name)
                for csv_row in reader:
                    csv_row["test_type"] = item["test"]
                    csv_row["test_param_name"] = item["param_name"]
                    csv_row["test_param_value"] = item["param_value"]
                    csv_row["repeat"] = item["repeat"]
                    csv_row["source_log"] = item["log_path"]
                    combined.append(csv_row)
        out = by_test_dir / f"{test}.csv"
        with out.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)
        manifest_out = by_test_dir / f"{test}_manifest.csv"
        with manifest_out.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(items[0].keys()))
            writer.writeheader()
            writer.writerows(items)
        print(f"Combined {test} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("metrics_results") / "airsim_extended")
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()
    paths = sorted(p for p in args.out_dir.glob("**/air_*.csv") if p.is_file() and p.stat().st_size > 200)
    rows = [parse_log(path) for path in paths]
    out = args.manifest or args.out_dir / "suite_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    combine_by_test(args.out_dir, rows)
    print(f"Rebuilt manifest -> {out} ({len(rows)} logs)")


if __name__ == "__main__":
    main()
