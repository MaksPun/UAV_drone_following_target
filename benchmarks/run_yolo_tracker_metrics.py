from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics_lib import RESULTS_DIR, mean, write_csv, write_markdown_table, add_project_root_to_path
from benchmarks.metrics_lib import install_airsim_stub_for_offline_metrics, install_cv2_stub_for_offline_metrics

add_project_root_to_path()
install_airsim_stub_for_offline_metrics()
install_cv2_stub_for_offline_metrics()

from uav_tracking.vision import Det, YoloDetector


def detector_stub(bytetrack: bool):
    det = object.__new__(YoloDetector)
    det.bytetrack = bytetrack
    det.sel_id = None
    return det


def make_det(cx, cy, conf, track_id, size=42):
    return Det((int(cx - size), int(cy - size), int(cx + size), int(cy + size)), conf, track_id)


def sequence_crossing_distractor(n=240, seed=7):
    rng = random.Random(seed)
    for i in range(n):
        x_true = 250 + i * 3.1
        y_true = 360 + rng.gauss(0, 5)
        x_false = 990 - i * 2.8
        y_false = 360 + rng.gauss(0, 5)
        true_conf = 0.72 + rng.gauss(0, 0.05)
        false_conf = 0.68 + rng.gauss(0, 0.08)
        if 90 < i < 145:
            false_conf += 0.18
        dets = [
            make_det(x_true, y_true, true_conf, 1),
            make_det(x_false, y_false, false_conf, 2),
        ]
        yield i, dets, 1


def sequence_occlusion_confdrop(n=240, seed=11):
    rng = random.Random(seed)
    for i in range(n):
        dets = []
        if not (95 <= i <= 125):
            conf = 0.84 - (0.28 if 60 <= i <= 150 else 0.0) + rng.gauss(0, 0.05)
            dets.append(make_det(640 + rng.gauss(0, 8), 360 + rng.gauss(0, 8), conf, 4))
        if 50 <= i <= 170:
            dets.append(make_det(650 + rng.gauss(0, 12), 370 + rng.gauss(0, 12), 0.76 + rng.gauss(0, 0.07), 9))
        yield i, dets, 4


SCENARIOS = {
    "crossing_distractor": sequence_crossing_distractor,
    "occlusion_confdrop": sequence_occlusion_confdrop,
}
TRUE_IDS = {
    "crossing_distractor": 1,
    "occlusion_confdrop": 4,
}


def run_case(scenario_name: str, bytetrack: bool):
    picker = detector_stub(bytetrack)
    selected_ids = []
    lost = 0
    for _, dets, true_id in SCENARIOS[scenario_name]():
        picked = picker.pick(dets)
        if picked is None:
            selected_ids.append(None)
            lost += 1
        else:
            selected_ids.append(picked.track_id)

    true_hits = sum(1 for sid in selected_ids if sid == TRUE_IDS[scenario_name])
    switches = 0
    prev = None
    for sid in selected_ids:
        if sid is not None and prev is not None and sid != prev:
            switches += 1
        if sid is not None:
            prev = sid
    return {
        "scenario": scenario_name,
        "selector": "yolo_bytetrack" if bytetrack else "yolo_confidence_only",
        "true_target_ratio": true_hits / len(selected_ids),
        "lost_frames": lost,
        "id_switches": switches,
        "selected_frames": sum(1 for sid in selected_ids if sid is not None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    rows = []
    for scenario in SCENARIOS:
        rows.append(run_case(scenario, False))
        rows.append(run_case(scenario, True))
    write_csv(args.out / "yolo_tracker_summary.csv", rows)
    write_markdown_table(args.out / "yolo_tracker_summary.md", "YOLO selection with and without ByteTrack", rows)
    print(f"Wrote {args.out / 'yolo_tracker_summary.md'}")


if __name__ == "__main__":
    main()
