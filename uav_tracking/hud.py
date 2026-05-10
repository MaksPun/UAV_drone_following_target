from __future__ import annotations

import math

import cv2
import numpy as np
import pygame

from .config import CFG
from .speed import AdaptiveSpeedManager, SpeedLimits
from .vision import Det


def draw_hud(img, lines, det: Det | None, lim: SpeedLimits, asp: "AdaptiveSpeedManager"):
    H, W = img.shape[:2]
    for r in range(3):
        for c in range(3):
            x0, y0 = c * W // 3, r * H // 3
            cv2.rectangle(img, (x0, y0), ((c + 1) * W // 3, (r + 1) * H // 3), (45, 45, 45), 1)
    if det is not None:
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 80), 2)
        cv2.circle(img, (int(det.cx), int(det.cy)), 5, (0, 255, 80), -1)
        col_i = min(2, int(det.cx / (W // 3)))
        row_i = min(2, int(det.cy / (H // 3)))
        cv2.rectangle(
            img,
            (col_i * W // 3, row_i * H // 3),
            ((col_i + 1) * W // 3, (row_i + 1) * H // 3),
            (0, 180, 255),
            2,
        )
    ay = int(H * CFG.aim_y_ratio)
    cv2.line(img, (W // 2 - 18, ay), (W // 2 + 18, ay), (255, 200, 0), 1)
    cv2.line(img, (W // 2, ay - 10), (W // 2, ay + 10), (255, 200, 0), 1)
    bar_w = int(W * asp.speed_scale)
    cv2.rectangle(img, (0, H - 8), (bar_w, H), (0, 200, 100), -1)
    cv2.putText(
        img,
        f"cube {asp.cube_speed_ms:.1f}m/s  vx_lim={lim.vx:.1f} vy={lim.vy:.1f} yaw={lim.yaw:.0f}",
        (8, H - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (200, 255, 180),
        1,
        cv2.LINE_AA,
    )
    ov = img.copy()
    lh = 16
    px, py = 10, 10
    mw = 0
    for ln in lines:
        tw, _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0]
        mw = max(mw, tw)
    cv2.rectangle(ov, (px - 4, py - 4), (px + mw + 8, py + lh * len(lines) + 12), (14, 14, 14), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    yy = py + 11
    for ln in lines:
        cv2.putText(img, ln, (px, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
        yy += lh


def to_surface(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")


BLUE = (255, 145, 35)
GREEN = (50, 240, 70)
RED = (35, 65, 255)
YELLOW = (0, 240, 240)
WHITE = (230, 230, 230)
GRAY = (155, 155, 155)
DARK = (8, 9, 10)


def _put(img, text, org, scale=0.46, color=WHITE, thick=1):
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _panel(img, x, y, w, h, title=None, alpha=0.68):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), DARK, -1)
    cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (95, 95, 95), 1)
    if title:
        _put(img, title, (x + 10, y + 22), 0.48, BLUE, 1)


def _fit(img, w, h):
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = min(w / max(1, iw), h / max(1, ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), cv2.INTER_AREA)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out


def _draw_det(img, det: Det | None, src_shape, x, y, w, h, label=True):
    if det is None or src_shape is None:
        return
    src_h, src_w = src_shape[:2]
    sx, sy = w / max(1, src_w), h / max(1, src_h)
    x1, y1, x2, y2 = det.xyxy
    p1 = (x + int(x1 * sx), y + int(y1 * sy))
    p2 = (x + int(x2 * sx), y + int(y2 * sy))
    cv2.rectangle(img, p1, p2, GREEN, 2)
    cx, cy = x + int(det.cx * sx), y + int(det.cy * sy)
    cv2.circle(img, (cx, cy), 4, GREEN, -1)
    if label:
        tid = det.track_id if det.track_id is not None else 1
        txt = f"target (ID:{tid}) {det.conf:.2f}"
        tw, th = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
        bx1, by1 = p1[0], max(y + 4, p1[1] - th - 10)
        cv2.rectangle(img, (bx1, by1), (bx1 + tw + 8, by1 + th + 8), (0, 120, 0), -1)
        _put(img, txt, (bx1 + 4, by1 + th + 3), 0.46, GREEN, 1)


def _crosshair(img, x, y, w, h):
    cx, cy = x + w // 2, y + int(h * CFG.aim_y_ratio)
    color = (90, 255, 120)
    cv2.line(img, (cx - 18, cy), (cx + 18, cy), color, 1)
    cv2.line(img, (cx, cy - 14), (cx, cy + 14), color, 1)
    for dx, dy in [(0, -165), (0, 165), (-180, 0), (180, 0)]:
        px, py = cx + dx, cy + dy
        cv2.line(img, (px - 10, py), (px + 10, py), color, 1)
        cv2.line(img, (px, py - 10), (px, py + 10), color, 1)


def _quat_euler_deg(pose):
    try:
        q = pose.orientation
        x, y, z, w = float(q.x_val), float(q.y_val), float(q.z_val), float(q.w_val)
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)
        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
    except Exception:
        return 0.0, 0.0, 0.0


def _compass(img, x, y, yaw, pitch, roll):
    r = 54
    c = (x + r, y + r)
    cv2.circle(img, c, r, (85, 85, 85), 1)
    for a, lab in [(0, "E"), (90, "N"), (180, "W"), (270, "S")]:
        rad = math.radians(a)
        px = int(c[0] + math.cos(rad) * (r - 8))
        py = int(c[1] - math.sin(rad) * (r - 8))
        _put(img, lab, (px - 5, py + 5), 0.42, WHITE, 1)
    hdg = math.radians(90.0 - yaw)
    tip = (int(c[0] + math.cos(hdg) * 33), int(c[1] - math.sin(hdg) * 33))
    left = (int(c[0] + math.cos(hdg + 2.45) * 12), int(c[1] - math.sin(hdg + 2.45) * 12))
    right = (int(c[0] + math.cos(hdg - 2.45) * 12), int(c[1] - math.sin(hdg - 2.45) * 12))
    cv2.fillConvexPoly(img, np.array([tip, left, right], np.int32), RED)
    cv2.circle(img, c, 6, WHITE, -1)
    _put(img, f"Yaw:  {yaw:5.1f}", (x + 130, y + 34), 0.47, WHITE, 1)
    _put(img, f"Pitch:{pitch:5.1f}", (x + 130, y + 58), 0.47, WHITE, 1)
    _put(img, f"Roll: {roll:5.1f}", (x + 130, y + 82), 0.47, WHITE, 1)


def _depth_view(depth, w, h):
    if depth is None:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        _put(out, "Depth camera unavailable", (18, h // 2), 0.45, GRAY, 1)
        return out
    d = np.asarray(depth, dtype=np.float32)
    valid = d[np.isfinite(d) & (d > 0.05)]
    if valid.size == 0:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        lo, hi = np.percentile(valid, [2, 98])
        hi = max(hi, lo + 0.1)
        norm = np.clip((d - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)
    return cv2.resize(colored, (w, h), cv2.INTER_AREA)


def _metric(history, key, default=0.0):
    vals = [float(h[key]) for h in history if h.get(key) is not None]
    return vals if vals else [default]


def _running_metrics(history):
    if not history:
        return {
            "mean_abs_dist_m": 0.0,
            "rmse_img_px": 0.0,
            "visible_ratio": 0.0,
            "lost_episodes": 0,
            "mean_recovery_s": 0.0,
            "p95_cmd_yaw_dps": 0.0,
        }
    dist = _metric(history, "dist_err_m")
    img = _metric(history, "img_err_px")
    yaw = [abs(v) for v in _metric(history, "cmd_yaw_dps")]
    visible = [1.0 if h.get("visible") else 0.0 for h in history]
    lost_eps = 0
    prev = True
    rec = []
    lost_start = None
    for h in history:
        vis = bool(h.get("visible"))
        if not vis and prev:
            lost_eps += 1
            lost_start = float(h.get("t", 0.0))
        if vis and not prev and lost_start is not None:
            rec.append(max(0.0, float(h.get("t", 0.0)) - lost_start))
            lost_start = None
        prev = vis
    return {
        "mean_abs_dist_m": float(np.mean(np.abs(dist))),
        "rmse_img_px": float(math.sqrt(np.mean(np.square(img)))),
        "visible_ratio": float(np.mean(visible)) if visible else 0.0,
        "lost_episodes": int(lost_eps),
        "mean_recovery_s": float(np.mean(rec)) if rec else 0.0,
        "p95_cmd_yaw_dps": float(np.percentile(yaw, 95)) if yaw else 0.0,
    }


def _plot_panel(img, x, y, w, h, title, series, colors, labels, y_min, y_max, unit=""):
    _panel(img, x, y, w, h, title, alpha=0.83)
    gx, gy, gw, gh = x + 48, y + 32, w - 62, h - 50
    cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (55, 55, 55), 1)
    cv2.line(img, (gx, gy + gh // 2), (gx + gw, gy + gh // 2), (45, 45, 45), 1)
    _put(img, f"{y_max:g}", (x + 12, gy + 6), 0.36, GRAY, 1)
    _put(img, "0", (x + 22, gy + gh // 2 + 5), 0.36, GRAY, 1)
    _put(img, f"{y_min:g}", (x + 8, gy + gh + 5), 0.36, GRAY, 1)
    for vals, color in zip(series, colors):
        if len(vals) < 2:
            continue
        clipped = vals[-180:]
        pts = []
        for i, v in enumerate(clipped):
            px = gx + int(i * gw / max(1, len(clipped) - 1))
            n = (float(v) - y_min) / max(1e-6, y_max - y_min)
            py = gy + gh - int(max(0.0, min(1.0, n)) * gh)
            pts.append((px, py))
        cv2.polylines(img, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)
    lx = x + w - 112
    for i, (lab, color) in enumerate(zip(labels, colors)):
        cv2.line(img, (lx, y + 34 + i * 18), (lx + 24, y + 34 + i * 18), color, 2)
        _put(img, lab, (lx + 30, y + 39 + i * 18), 0.35, WHITE, 1)
    if unit:
        _put(img, unit, (x + 12, y + h - 13), 0.36, GRAY, 1)


def draw_operator_hud(
    frame,
    *,
    depth=None,
    det: Det | None = None,
    lim: SpeedLimits,
    asp: AdaptiveSpeedManager,
    history=None,
    logs=None,
    vehicle_state="",
    mode="AUTO",
    ctrl_name="LQR",
    tracker_name="YOLO",
    reacq_mode="",
    state_est="Kalman",
    bx=0.0,
    by=0.0,
    bz=0.0,
    dist=0.0,
    z_now=0.0,
    cmd=None,
    lost_s=0.0,
    drone_pose=None,
    metrics_log="",
    app_time=0.0,
    seq=0,
    safety_status="OK",
):
    history = list(history or [])
    logs = list(logs or [])
    H, W = 900, 1400
    right_w, status_h, bottom_h = 280, 42, 245
    main_w, main_h = W - right_w, H - status_h - bottom_h
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    main = cv2.resize(frame, (main_w, main_h), cv2.INTER_AREA)
    canvas[0:main_h, 0:main_w] = main
    _crosshair(canvas, 0, 0, main_w, main_h)
    _draw_det(canvas, det, frame.shape, 0, 0, main_w, main_h, True)

    ov = canvas.copy()
    cv2.rectangle(ov, (main_w, 0), (W, H - status_h), DARK, -1)
    cv2.rectangle(ov, (0, main_h), (W, H - status_h), DARK, -1)
    cv2.rectangle(ov, (0, H - status_h), (W, H), (2, 3, 4), -1)
    cv2.addWeighted(ov, 0.90, canvas, 0.10, 0, canvas)

    _panel(canvas, 10, 12, 208, 205, "AIRSIM HUD", 0.62)
    armed = vehicle_state.upper() in {"TAKEOFF", "OFFBOARD", "LANDING"}
    left_lines = [
        ("Collided:", "False"),
        ("Mode:", "Guided" if mode.startswith("AUTO") else "Manual"),
        ("Armed:", str(armed)),
        ("GPS:", "3D Fix"),
        ("Speed:", f"{asp.cube_speed_ms:.2f} m/s"),
        ("Alt (rel):", f"{abs(z_now):.1f} m"),
        ("Alt (abs):", f"{abs(z_now) + 100.0:.1f} m"),
        ("Battery:", "92 %"),
    ]
    yy = 62
    for k, v in left_lines:
        _put(canvas, k, (22, yy), 0.44, WHITE, 1)
        _put(canvas, v, (130, yy), 0.44, WHITE, 1)
        yy += 22

    _panel(canvas, 866, 12, 240, 170, "METRICS (running)", 0.68)
    rm = _running_metrics(history)
    yy = 62
    for k, v in [
        ("mean_abs_dist_m:", f"{rm['mean_abs_dist_m']:.2f}"),
        ("rmse_img (px):", f"{rm['rmse_img_px']:.1f}"),
        ("visible_ratio:", f"{rm['visible_ratio']:.2f}"),
        ("lost_episodes:", f"{rm['lost_episodes']}"),
        ("mean_recovery_s:", f"{rm['mean_recovery_s']:.2f}"),
        ("p95_cmd_yaw_dps:", f"{rm['p95_cmd_yaw_dps']:.1f}"),
    ]:
        _put(canvas, k, (880, yy), 0.42, WHITE, 1)
        _put(canvas, v, (1038, yy), 0.42, WHITE, 1)
        yy += 21

    follow_txt = f"Follow mode:  {ctrl_name}"
    tw = cv2.getTextSize(follow_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    _panel(canvas, main_w // 2 - tw // 2 - 12, 10, tw + 24, 34, None, 0.55)
    _put(canvas, "Follow mode:", (main_w // 2 - tw // 2, 33), 0.55, WHITE, 1)
    _put(canvas, ctrl_name, (main_w // 2 + tw // 2 - 44, 33), 0.55, GREEN, 2)

    yaw, pitch, roll = _quat_euler_deg(drone_pose)
    _panel(canvas, 15, main_h - 178, 278, 146, None, 0.45)
    _compass(canvas, 35, main_h - 162, yaw, pitch, roll)
    _put(canvas, "FPS:  30.0", (42, main_h - 38), 0.48, YELLOW, 1)

    _panel(canvas, 848, main_h - 190, 250, 150, "COMMANDS (to PX4)", 0.72)
    cmd_yaw = getattr(cmd, "yaw", 0.0)
    cmd_side = getattr(cmd, "vy", 0.0)
    cmd_vertical = getattr(cmd, "vz", 0.0)
    cmd_fwd = getattr(cmd, "vx", 0.0)
    cmd_lines = [
        ("cmd_yaw_rate (dps):", cmd_yaw, RED),
        ("cmd_forward (m/s):", cmd_fwd, YELLOW),
        ("cmd_lateral (m/s):", cmd_side, (255, 170, 0)),
        ("cmd_vertical (m/s):", cmd_vertical, GREEN),
    ]
    yy = main_h - 130
    for k, v, c in cmd_lines:
        _put(canvas, k, (860, yy), 0.40, WHITE, 1)
        _put(canvas, f"{v:6.2f}", (1028, yy), 0.40, c, 1)
        yy += 25

    rx = main_w
    _put(canvas, "SYSTEM STATUS", (rx + 22, 26), 0.50, BLUE, 1)
    yy = 56
    for k, v, c in [
        ("Controller:", ctrl_name, GREEN),
        ("Tracker:", tracker_name, GREEN),
        ("Reacq:", reacq_mode, GREEN),
        ("State Est.:", state_est, GREEN),
    ]:
        _put(canvas, k, (rx + 22, yy), 0.41, WHITE, 1)
        _put(canvas, v, (rx + 130, yy), 0.41, c, 1)
        yy += 24
    cv2.line(canvas, (rx + 18, yy + 6), (W - 14, yy + 6), (90, 90, 90), 1)

    yy += 34
    _put(canvas, "TARGET INFO", (rx + 22, yy), 0.50, BLUE, 1)
    yy += 28
    tid = det.track_id if det is not None and det.track_id is not None else (1 if det is not None else "-")
    bbox = det.xyxy if det is not None else ("-", "-", "-", "-")
    center = (int(det.cx), int(det.cy)) if det is not None else ("-", "-")
    tlines = [
        ("ID:", f"{tid}", WHITE),
        ("Class:", "target", WHITE),
        ("Conf:", f"{det.conf:.2f}" if det is not None else "-", WHITE),
        ("BBox:", f"{bbox}", WHITE),
        ("Center (px):", f"{center}", WHITE),
        ("Depth (m):", f"{dist:.1f}", WHITE),
        ("Visible:", str(det is not None), GREEN if det is not None else RED),
    ]
    for k, v, c in tlines:
        _put(canvas, k, (rx + 22, yy), 0.39, WHITE, 1)
        _put(canvas, v, (rx + 130, yy), 0.39, c, 1)
        yy += 24
    cv2.line(canvas, (rx + 18, yy + 6), (W - 14, yy + 6), (90, 90, 90), 1)

    yy += 34
    _put(canvas, "CONTROL MODE", (rx + 22, yy), 0.50, BLUE, 1)
    yy += 30
    for k, v in [("Mode:", mode), ("Controller:", ctrl_name)]:
        _put(canvas, k, (rx + 22, yy), 0.41, WHITE, 1)
        _put(canvas, v, (rx + 130, yy), 0.41, GREEN, 1)
        yy += 24
    cv2.line(canvas, (rx + 18, yy + 8), (W - 14, yy + 8), (90, 90, 90), 1)

    yy += 38
    _put(canvas, "RECOVERY", (rx + 22, yy), 0.50, BLUE, 1)
    yy += 30
    for k, v, c in [
        ("Status:", safety_status, GREEN),
        ("Last lost (s ago):", f"{lost_s:.1f}", WHITE),
        ("Reacq attempts:", f"{rm['lost_episodes']}", WHITE),
    ]:
        _put(canvas, k, (rx + 22, yy), 0.39, WHITE, 1)
        _put(canvas, v, (rx + 162, yy), 0.39, c, 1)
        yy += 24

    yb = main_h + 8
    ph = bottom_h - 16
    widths = [266, 296, 256, 276, 282]
    xs = [4]
    for width in widths[:-1]:
        xs.append(xs[-1] + width + 4)
    _panel(canvas, xs[0], yb, widths[0], ph, "RGB CAMERA", 0.86)
    rgb_small = _fit(frame, widths[0] - 16, ph - 44)
    canvas[yb + 30 : yb + 30 + rgb_small.shape[0], xs[0] + 8 : xs[0] + 8 + rgb_small.shape[1]] = rgb_small
    _draw_det(canvas, det, frame.shape, xs[0] + 8, yb + 30, widths[0] - 16, ph - 44, False)

    _panel(canvas, xs[1], yb, widths[1], ph, "DEPTH (meters)", 0.86)
    dview = _depth_view(depth, widths[1] - 62, ph - 44)
    canvas[yb + 30 : yb + 30 + dview.shape[0], xs[1] + 8 : xs[1] + 8 + dview.shape[1]] = dview
    _draw_det(canvas, det, frame.shape, xs[1] + 8, yb + 30, widths[1] - 62, ph - 44, False)
    gx = xs[1] + widths[1] - 44
    gy = yb + 30
    cv2.rectangle(canvas, (gx, gy), (gx + 14, gy + ph - 44), (80, 80, 80), 1)
    for i in range(ph - 45):
        val = 255 - int(255 * i / max(1, ph - 45))
        color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()
        cv2.line(canvas, (gx + 1, gy + i), (gx + 13, gy + i), color, 1)
    _put(canvas, "40", (gx + 20, gy + 10), 0.35, WHITE, 1)
    _put(canvas, "20", (gx + 20, gy + (ph - 44) // 2), 0.35, WHITE, 1)
    _put(canvas, "0", (gx + 20, gy + ph - 48), 0.35, WHITE, 1)

    ex = [h.get("err_x_px", 0.0) for h in history]
    ey = [h.get("err_y_px", 0.0) for h in history]
    _plot_panel(canvas, xs[2], yb, widths[2], ph, "ERROR (image plane)", [ex, ey], [RED, GREEN], ["ex", "ey"], -120, 120, "time (s)")

    yaw_vals = [h.get("cmd_yaw_dps", 0.0) for h in history]
    _plot_panel(canvas, xs[3], yb, widths[3], ph, "COMMAND YAW RATE (dps)", [yaw_vals], [YELLOW], ["cmd_yaw_rate"], -90, 90, "time (s)")

    _panel(canvas, xs[4], yb, widths[4], ph, "LOG (last messages)", 0.86)
    yy = yb + 48
    for line in logs[-8:]:
        _put(canvas, line, (xs[4] + 14, yy), 0.37, WHITE, 1)
        yy += 22

    cv2.line(canvas, (0, H - status_h), (W, H - status_h), (105, 105, 105), 1)
    _put(canvas, "Connection:", (16, H - 17), 0.39, WHITE, 1)
    _put(canvas, "OK (MAVSDK)", (112, H - 17), 0.39, GREEN, 1)
    _put(canvas, "Vehicle: PX4 (SITL)", (300, H - 17), 0.39, WHITE, 1)
    mins, secs = int(app_time // 60), int(app_time % 60)
    _put(canvas, f"Time: {mins:02d}:{secs:02d}", (505, H - 17), 0.39, WHITE, 1)
    _put(canvas, f"Seq: {seq}", (680, H - 17), 0.39, WHITE, 1)
    log_txt = metrics_log if metrics_log else "no metrics log"
    if len(log_txt) > 46:
        log_txt = "..." + log_txt[-43:]
    _put(canvas, f"Log file: {log_txt}", (808, H - 17), 0.39, WHITE, 1)
    cv2.circle(canvas, (1245, H - 21), 6, RED, 2)
    _put(canvas, "RECORDING" if metrics_log else "LIVE", (1264, H - 17), 0.37, WHITE, 1)
    return canvas
