from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import airsim
import cv2
import numpy as np

from benchmarks.metrics_lib import mean, p95, rmse, write_markdown_table
from uav_tracking.common import world_to_body
from uav_tracking.vision import YoloDetector, get_scene_bgr


def get_depth_m(client: airsim.MultirotorClient, cam: str):
    req = [airsim.ImageRequest(cam, airsim.ImageType.DepthPerspective, True, False)]
    response = client.simGetImages(req)[0]
    if response.width == 0 or response.height == 0 or not response.image_data_float:
        return None
    arr = np.array(response.image_data_float, dtype=np.float32)
    return arr.reshape(response.height, response.width)


def roi_from_det(det, shape, margin_ratio: float):
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in det.xyxy]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    return (
        max(0, x1 + mx),
        max(0, y1 + my),
        min(w, x2 - mx),
        min(h, y2 - my),
    )


def depth_roi_stats(depth, det, margin_ratio: float):
    x1, y1, x2, y2 = roi_from_det(det, depth.shape, margin_ratio)
    if x2 <= x1 or y2 <= y1:
        return None, 0
    roi = depth[y1:y2, x1:x2]
    vals = roi[np.isfinite(roi)]
    vals = vals[(vals > 0.2) & (vals < 200.0)]
    if vals.size == 0:
        return None, 0
    return float(np.median(vals)), int(vals.size)


def summarize(rows: list[dict]) -> list[dict]:
    valid = [r for r in rows if r["depth_roi_m"] != ""]
    body_err = [float(r["depth_minus_body_x_m"]) for r in valid]
    dist_err = [float(r["depth_minus_euclid_m"]) for r in valid]
    return [
        {
            "samples": len(rows),
            "valid_depth_samples": len(valid),
            "valid_ratio": len(valid) / max(1, len(rows)),
            "mae_depth_vs_body_x_m": mean(abs(v) for v in body_err),
            "rmse_depth_vs_body_x_m": rmse(body_err),
            "bias_depth_vs_body_x_m": mean(body_err),
            "p95_abs_depth_vs_body_x_m": p95(abs(v) for v in body_err),
            "mae_depth_vs_euclid_m": mean(abs(v) for v in dist_err),
            "rmse_depth_vs_euclid_m": rmse(dist_err),
            "bias_depth_vs_euclid_m": mean(dist_err),
            "p95_abs_depth_vs_euclid_m": p95(abs(v) for v in dist_err),
        }
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", default=r"C:\Users\User\PycharmProjects\AirSim\runs\detect\train4\weights\best.pt")
    ap.add_argument("--cam", default="0")
    ap.add_argument("--cube_name", default="BP_TargetCube_5")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--device", default="0")
    ap.add_argument("--bytetrack", action="store_true")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--interval", type=float, default=0.10)
    ap.add_argument("--roi_margin_ratio", type=float, default=0.22, help="Ignore bbox border pixels to reduce background leakage.")
    ap.add_argument("--out", type=Path, default=Path("metrics_results") / "airsim_depth_roi" / "depth_roi_live.csv")
    args = ap.parse_args()

    detector = YoloDetector(args.weights, args.imgsz, args.conf, args.iou, args.device, args.bytetrack)
    client = airsim.MultirotorClient()
    client.confirmConnection()

    rows: list[dict] = []
    for idx in range(args.samples):
        t0 = time.time()
        frame = get_scene_bgr(client, args.cam)
        depth = get_depth_m(client, args.cam)
        drone_pose = client.simGetVehiclePose()
        cube_pose = client.simGetObjectPose(args.cube_name)
        dp = drone_pose.position
        cp = cube_pose.position
        dx = float(cp.x_val - dp.x_val)
        dy = float(cp.y_val - dp.y_val)
        dz = float(cp.z_val - dp.z_val)
        bx, by, bz = world_to_body(drone_pose, dx, dy, dz)
        euclid = math.sqrt(dx * dx + dy * dy + dz * dz)

        det = None
        depth_roi = None
        roi_count = 0
        if frame is not None and depth is not None:
            det = detector.pick(detector.infer(frame))
            if det is not None:
                depth_roi, roi_count = depth_roi_stats(depth, det, args.roi_margin_ratio)

        rows.append(
            {
                "sample": idx,
                "wall_time": t0,
                "detected": int(det is not None),
                "track_id": det.track_id if det is not None and det.track_id is not None else "",
                "conf": det.conf if det is not None else "",
                "bbox": det.xyxy if det is not None else "",
                "roi_valid_pixels": roi_count,
                "depth_roi_m": depth_roi if depth_roi is not None else "",
                "body_x_m": bx,
                "body_y_m": by,
                "body_z_m": bz,
                "euclid_dist_m": euclid,
                "depth_minus_body_x_m": depth_roi - bx if depth_roi is not None else "",
                "depth_minus_euclid_m": depth_roi - euclid if depth_roi is not None else "",
            }
        )

        elapsed = time.time() - t0
        if args.interval > elapsed:
            time.sleep(args.interval - elapsed)

    write_csv(args.out, rows)
    summary = summarize(rows)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    md_path = args.out.with_name(args.out.stem + "_summary.md")
    write_csv(summary_path, summary)
    write_markdown_table(md_path, "Live AirSim depth ROI distance metrics", summary)
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
