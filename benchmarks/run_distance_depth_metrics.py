"""Offline comparison of distance estimation methods."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics_lib import RESULTS_DIR, mean, p95, rmse, write_csv, write_markdown_table


def synthetic_depth_roi(true_dist_m: float, rng: random.Random, pixels=121):
    values = []
    for i in range(pixels):
        if rng.random() < 0.08:
            values.append(true_dist_m + rng.uniform(-4.0, 4.0))
        else:
            values.append(true_dist_m + rng.gauss(0.0, 0.18 + 0.012 * true_dist_m))
    values.sort()
    return values[len(values) // 2]


def synthetic_depth_center(true_dist_m: float, rng: random.Random):
    if rng.random() < 0.18:
        return true_dist_m + rng.uniform(-5.0, 5.0)
    return true_dist_m + rng.gauss(0.0, 0.35 + 0.02 * true_dist_m)


def synthetic_bbox_distance(true_dist_m: float, rng: random.Random):
    # Toy inverse-size estimator. It becomes weak when the target rotates or bbox is noisy.
    apparent = 850.0 / max(true_dist_m, 0.5)
    apparent *= rng.uniform(0.78, 1.24)
    apparent += rng.gauss(0.0, 4.0)
    return 850.0 / max(apparent, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = []
    summary = []
    methods = {
        "airsim_pose_ground_truth": lambda d: d,
        "depth_roi_median": lambda d: synthetic_depth_roi(d, rng),
        "depth_center_pixel": lambda d: synthetic_depth_center(d, rng),
        "bbox_inverse_size": lambda d: synthetic_bbox_distance(d, rng),
    }
    errors = {name: [] for name in methods}
    for i in range(args.samples):
        true_dist = rng.uniform(5.0, 32.0)
        for name, fn in methods.items():
            est = fn(true_dist)
            err = est - true_dist
            errors[name].append(err)
            rows.append({"sample": i, "method": name, "true_dist_m": true_dist, "estimated_dist_m": est, "error_m": err})

    for name, vals in errors.items():
        summary.append(
            {
                "method": name,
                "mae_m": mean(abs(v) for v in vals),
                "rmse_m": rmse(vals),
                "bias_m": mean(vals),
                "p95_abs_error_m": p95(abs(v) for v in vals),
            }
        )
    write_csv(args.out / "distance_depth_samples.csv", rows)
    write_csv(args.out / "distance_depth_summary.csv", summary)
    write_markdown_table(args.out / "distance_depth_summary.md", "Distance estimation methods", summary)
    print(f"Wrote {args.out / 'distance_depth_summary.md'}")


if __name__ == "__main__":
    main()
