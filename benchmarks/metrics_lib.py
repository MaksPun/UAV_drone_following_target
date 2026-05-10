"""Shared helpers for CSV, Markdown and offline benchmark statistics."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


RESULTS_DIR = Path("metrics_results")


def ensure_results_dir(path: Path = RESULTS_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def rmse(values):
    values = list(values)
    return math.sqrt(mean(v * v for v in values)) if values else 0.0


def p95(values):
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return values[idx]


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_results_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    ensure_results_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_markdown_table(path: Path, title: str, rows: list[dict]) -> None:
    ensure_results_dir(path.parent)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    out = [f"# {title}", ""]
    out.append("| " + " | ".join(keys) + " |")
    out.append("| " + " | ".join("---" for _ in keys) + " |")
    for row in rows:
        out.append("| " + " | ".join(format_cell(row[k]) for k in keys) + " |")
    out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def add_project_root_to_path() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def install_airsim_stub_for_offline_metrics() -> None:
    import sys
    import types

    if "airsim" in sys.modules:
        return
    try:
        __import__("airsim")
        return
    except ModuleNotFoundError:
        pass

    airsim = types.ModuleType("airsim")
    airsim.Pose = object
    airsim.MultirotorClient = object
    airsim.Vector3r = lambda x=0.0, y=0.0, z=0.0: types.SimpleNamespace(
        x_val=float(x), y_val=float(y), z_val=float(z)
    )

    class ImageType:
        Scene = 0
        DepthPerspective = 2

    airsim.ImageType = ImageType
    airsim.ImageRequest = lambda *args, **kwargs: types.SimpleNamespace(args=args, kwargs=kwargs)
    sys.modules["airsim"] = airsim


def install_cv2_stub_for_offline_metrics() -> None:
    import sys
    import types

    if "cv2" in sys.modules:
        return
    try:
        __import__("cv2")
        return
    except ModuleNotFoundError:
        pass

    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.imdecode = lambda *args, **kwargs: None
    sys.modules["cv2"] = cv2
