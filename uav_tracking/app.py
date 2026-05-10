"""Main application loop for AirSim, YOLO, controllers, HUD and metrics."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
import time
from collections import deque

import airsim
import cv2
import pygame

from .airsim_scenarios import SCENARIO_NAMES, ScenarioState
from .common import Cmd, CubeCmd, clamp, pose_from_body_offset
from .config import CFG
from .controllers import LQRTracker, PIDTracker
from .estimation import BodyMotionTracker, ImageMotionTracker, KalmanCube, SpatialGrid27
from .hud import draw_hud, draw_operator_hud, to_surface
from .metrics import MetricsLogger
from .reacquisition import PredictiveReacq, ReacquireFSM
from .safety import SafetyMonitor
from .simulation import move_cube
from .speed import AdaptiveSpeedManager
from .vehicle import MavsdkVehicle
from .vision import Det, YoloDetector, estimate_body_from_depth, get_depth_m, get_scene_bgr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("UAV")


def jitter_detection(det: Det | None, noise_px: float, rng: random.Random, frame_shape) -> Det | None:
    if det is None or noise_px <= 0.0 or frame_shape is None:
        return det
    h, w = frame_shape[:2]
    dx = rng.gauss(0.0, noise_px)
    dy = rng.gauss(0.0, noise_px)
    x1, y1, x2, y2 = det.xyxy
    nx1 = int(max(0, min(w - 1, x1 + dx)))
    ny1 = int(max(0, min(h - 1, y1 + dy)))
    nx2 = int(max(1, min(w, x2 + dx)))
    ny2 = int(max(1, min(h, y2 + dy)))
    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)
    return Det((nx1, ny1, nx2, ny2), det.conf, det.track_id)


def delayed_measurement(history: list[dict], now: float, latency_s: float) -> dict:
    if not history or latency_s <= 0.0:
        return history[-1]
    target_t = now - latency_s
    chosen = history[0]
    for item in history:
        if item["t"] <= target_t:
            chosen = item
        else:
            break
    return chosen


def parse_xyzyaw(text: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("pose must be formatted as x,y,z,yaw_deg")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose values must be numeric") from exc


def yaw_quat(yaw_deg: float):
    yaw = math.radians(float(yaw_deg))
    return airsim.Quaternionr(0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def make_pose_xyzyaw(spec: tuple[float, float, float, float]) -> airsim.Pose:
    x, y, z, yaw_deg = spec
    pose = airsim.Pose()
    pose.position = airsim.Vector3r(float(x), float(y), float(z))
    pose.orientation = yaw_quat(yaw_deg)
    return pose


def reset_test_poses(client, cube_name: str, drone_pose_spec, cube_pose_spec, settle_s: float, label: str):
    drone_pose = make_pose_xyzyaw(drone_pose_spec)
    cube_pose = make_pose_xyzyaw(cube_pose_spec)
    try:
        client.simSetVehiclePose(drone_pose, ignore_collision=True)
        log.info("%s drone pose reset -> x=%.2f y=%.2f z=%.2f yaw=%.1f",
                 label, drone_pose_spec[0], drone_pose_spec[1], drone_pose_spec[2], drone_pose_spec[3])
    except Exception as exc:
        log.warning("%s drone pose reset failed: %s", label, exc)
    try:
        client.simSetObjectPose(cube_name, cube_pose, True)
        log.info("%s cube pose reset -> %s x=%.2f y=%.2f z=%.2f yaw=%.1f",
                 label, cube_name, cube_pose_spec[0], cube_pose_spec[1], cube_pose_spec[2], cube_pose_spec[3])
    except Exception as exc:
        log.warning("%s cube pose reset failed: %s", label, exc)
    if settle_s > 0:
        time.sleep(settle_s)


async def async_main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights",default=r"C:\Users\User\PycharmProjects\AirSim\runs\detect\train4\weights\best.pt")
    ap.add_argument("--cam",default="0")
    ap.add_argument("--cube_name",default="BP_TargetCube")
    ap.add_argument("--mavsdk_address",default="udpin://0.0.0.0:14560")
    ap.add_argument("--imgsz",type=int,default=1280)
    ap.add_argument("--conf",type=float,default=0.25)
    ap.add_argument("--iou",type=float,default=0.60)
    ap.add_argument("--device",default="0")
    ap.add_argument("--bytetrack",action="store_true")
    ap.add_argument("--infer_hz",type=float,default=CFG.infer_hz)
    ap.add_argument("--ctrl",choices=["pid","lqr"],default="lqr",
                    help="Controller type: pid or lqr")
    ap.add_argument("--metrics_log",default="",
                    help="CSV path for live AirSim metrics. Empty disables logging.")
    ap.add_argument("--metrics_label",default="",
                    help="Label stored in metrics CSV.")
    ap.add_argument("--metrics_scenario",choices=SCENARIO_NAMES,default="manual",
                    help="Scripted target motion scenario for AirSim metric runs.")
    ap.add_argument("--metrics_duration",type=float,default=0.0,
                    help="Auto-stop after N seconds. 0 means run until ESC/L.")
    ap.add_argument("--metrics_det_noise_px",type=float,default=0.0,
                    help="Add Gaussian noise to detection bbox center during metric runs.")
    ap.add_argument("--metrics_latency_ms",type=float,default=0.0,
                    help="Use delayed detection/target pose for control to emulate inference or command latency.")
    ap.add_argument("--metrics_occlusion_start",type=float,default=-1.0,
                    help="Scenario time when detection is forced lost. Negative disables forced occlusion.")
    ap.add_argument("--metrics_occlusion_duration",type=float,default=0.0,
                    help="Forced detection loss duration in seconds.")
    ap.add_argument("--metrics_scenario_speed_scale",type=float,default=1.0,
                    help="Multiplier applied to scripted AirSim target scenario velocities.")
    ap.add_argument("--connect_timeout",type=float,default=90.0,
                    help="Fail the run if MAVSDK/PX4 does not become armable in this many seconds.")
    ap.add_argument("--landing_wait",type=float,default=8.0,
                    help="Seconds to wait after land command before disarm and process shutdown.")
    ap.add_argument("--reset_poses",action="store_true",
                    help="Reset drone and target cube to fixed test poses before and after the metric run.")
    ap.add_argument("--drone_start_pose",type=parse_xyzyaw,default=parse_xyzyaw("0,0,0,0"),
                    help="Initial drone pose as x,y,z,yaw_deg in AirSim NED coordinates.")
    ap.add_argument("--cube_start_pose",type=parse_xyzyaw,default=parse_xyzyaw("22,0,-3.4,0"),
                    help="Initial cube pose as x,y,z,yaw_deg in AirSim NED coordinates.")
    ap.add_argument("--pose_settle_s",type=float,default=1.0,
                    help="Seconds to wait after resetting AirSim poses.")
    ap.add_argument("--land_on_finish",action="store_true",
                    help="Land before disarm when the app finishes.")
    ap.add_argument("--hud_style",choices=["operator","classic"],default="operator",
                    help="Visual UI style. operator is the dashboard HUD shown during experiments.")
    args=ap.parse_args()

    # Runtime blocks are created once and reused every frame.
    detector  = YoloDetector(args.weights,args.imgsz,args.conf,
                              args.iou,args.device,args.bytetrack)
    pid_tracker = PIDTracker()
    lqr_tracker = LQRTracker(dt_d=0.10)
    kalman    = KalmanCube()
    asp       = AdaptiveSpeedManager()
    grid27    = SpatialGrid27()
    imt       = ImageMotionTracker()
    body_hist = BodyMotionTracker()
    reacq_fsm = ReacquireFSM()
    reacq     = PredictiveReacq()
    safety    = SafetyMonitor()
    scenario  = ScenarioState(args.metrics_scenario, speed_scale=args.metrics_scenario_speed_scale)
    noise_rng = random.Random(3110)
    measurement_history: list[dict] = []

    ctrl_name = args.ctrl.upper()
    log.info("Controller: %s", ctrl_name)
    if args.metrics_scenario!="manual":
        log.info("Metrics scenario: %s", args.metrics_scenario)

    client=airsim.MultirotorClient()
    client.confirmConnection(); log.info("AirSim connected")
    if args.reset_poses:
        reset_test_poses(client,args.cube_name,args.drone_start_pose,args.cube_start_pose,args.pose_settle_s,"Initial")

    vehicle=MavsdkVehicle()
    try:
        await asyncio.wait_for(vehicle.connect(args.mavsdk_address), timeout=max(5.0, args.connect_timeout))
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"PX4 did not become armable within {args.connect_timeout:.1f}s") from exc
    await vehicle.arm_takeoff_offboard(CFG.takeoff_alt)
    safety.arm_home(client.simGetVehiclePose())

    frame=None
    while frame is None: frame=get_scene_bgr(client,args.cam); await asyncio.sleep(.08)
    H0,W0=frame.shape[:2]; log.info("Camera %dx%d",W0,H0)

    pygame.init()
    screen=pygame.display.set_mode((W0,H0),pygame.RESIZABLE)
    pygame.display.set_caption(f"PX4+MAVSDK+AirSim | {ctrl_name}+AdaptiveSpeed+27Zone")
    clock=pygame.time.Clock()

    mode="AUTO"; hover_now=False
    last_infer_t=0.; last_det:Det|None=None; last_ctrl_det_t=0.
    prev={k:False for k in "123hl"}; last_t=time.time(); running=True
    app_start_t=time.time(); seq=0
    hud_history=deque(maxlen=900)
    hud_logs=deque(maxlen=8)
    hud_logs.append(f"{time.strftime('%H:%M:%S')} INFO  Controller: {ctrl_name}")
    hud_logs.append(f"{time.strftime('%H:%M:%S')} INFO  Tracker: YOLO{' + ByteTrack' if args.bytetrack else ''}")
    last_target_visible=False
    depth_frame=None; last_depth_t=0.0
    metrics=None
    if args.metrics_log:
        label=args.metrics_label or f"{args.ctrl}_{args.metrics_scenario}_{int(time.time())}"
        metrics=MetricsLogger(
            args.metrics_log,label,args.metrics_scenario,args.ctrl,args.bytetrack,
            det_noise_px=args.metrics_det_noise_px,
            latency_ms=args.metrics_latency_ms,
            scenario_speed_scale=args.metrics_scenario_speed_scale,
            forced_occlusion_s=args.metrics_occlusion_duration,
        )
        log.info("Metrics log -> %s", args.metrics_log)
    scenario.start(time.time())
    lim=asp.current

    while running:
        now=time.time(); dt=clamp(now-last_t,.001,.25); last_t=now
        if args.metrics_duration>0. and metrics is not None and now-metrics.t0>=args.metrics_duration:
            running=False

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
        keys=pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]: running=False

        def edge(kc,ks):
            c=keys[kc]; f=c and not prev[ks]; prev[ks]=c; return f

        if edge(pygame.K_1,"1"):
            mode="AUTO"; hover_now=False
            pid_tracker.reset(); lqr_tracker.reset(); reacq.reset(); asp.reset()
        if edge(pygame.K_2,"2"):
            mode="MANUAL_DRONE"; hover_now=False
            pid_tracker.reset(); lqr_tracker.reset(); reacq.reset()
        if edge(pygame.K_3,"3"):
            mode="AUTO_CUBE"; hover_now=False
            pid_tracker.reset(); lqr_tracker.reset(); reacq.reset(); asp.reset()
        if edge(pygame.K_h,"h"):
            hover_now=not hover_now
            pid_tracker.reset(); lqr_tracker.reset(); reacq.reset()
        if edge(pygame.K_l,"l"):
            await vehicle.land(); await asyncio.sleep(3.)
            await vehicle.disarm(); running=False; break

        frame=get_scene_bgr(client,args.cam)
        if frame is None: clock.tick(30); await asyncio.sleep(0); continue

        if now-last_depth_t>=0.12:
            try:
                depth_frame=get_depth_m(client,args.cam)
            except Exception as exc:
                if depth_frame is None:
                    hud_logs.append(f"{time.strftime('%H:%M:%S')} WARN  Depth unavailable")
                log.debug("Depth frame unavailable: %s", exc)
            last_depth_t=now

        # YOLO is rate-limited so control and HUD rendering stay responsive.
        if now-last_infer_t>=1./max(args.infer_hz,1e-3):
            dets=detector.infer(frame); last_det=detector.pick(dets)
            last_infer_t=now

        drone_pose=client.simGetVehiclePose()

        # Metric options can emulate detector noise, latency and occlusion.
        scenario_t = scenario.elapsed(now) if scenario.enabled else 0.0
        forced_occlusion = (
            args.metrics_occlusion_start >= 0.0
            and args.metrics_occlusion_duration > 0.0
            and args.metrics_occlusion_start <= scenario_t < args.metrics_occlusion_start + args.metrics_occlusion_duration
        )
        current_target = None if forced_occlusion else last_det
        current_target = jitter_detection(current_target, args.metrics_det_noise_px, noise_rng, frame.shape)
        body_est=estimate_body_from_depth(
            current_target,depth_frame,frame.shape,CFG.depth_fov_deg,CFG.depth_roi_shrink,
            CFG.depth_min_m,CFG.depth_max_m)
        est_cube_pose=None
        if body_est is not None:
            bx,by,bz,dist=body_est
            est_cube_pose=pose_from_body_offset(drone_pose,bx,by,bz)
            kalman.update(est_cube_pose,now)
        else:
            bx=by=bz=dist=0.0
        measurement_history.append({
            "t": now,
            "target": current_target if est_cube_pose is not None else None,
            "cube_pose": est_cube_pose,
            "body_xyz": (bx,by,bz) if est_cube_pose is not None else None,
        })
        if len(measurement_history) > 240:
            measurement_history = measurement_history[-240:]
        control_measurement = delayed_measurement(measurement_history, now, args.metrics_latency_ms / 1000.0)
        target = control_measurement["target"]
        control_cube_pose = control_measurement["cube_pose"]
        control_body_xyz = control_measurement["body_xyz"]
        if target is not None:
            last_ctrl_det_t = now
        lost_s=now-last_ctrl_det_t if last_ctrl_det_t>0 else 9999.
        target_visible=current_target is not None
        if target_visible and not last_target_visible:
            tid=current_target.track_id if current_target and current_target.track_id is not None else 1
            hud_logs.append(f"{time.strftime('%H:%M:%S')} INFO  Target acquired ID={tid}")
        if not target_visible and last_target_visible:
            hud_logs.append(f"{time.strftime('%H:%M:%S')} WARN  Target lost")
        last_target_visible=target_visible

        z_now=float(drone_pose.position.z_val)

        if body_est is not None:
            grid27.push(bx,by,bz,now)
            body_hist.push(bx,by,bz,now)
        if current_target is not None: imt.push(current_target,frame.shape,now)

        lim = asp.update(kalman, target,
                         frame.shape if target is not None else None,
                         drone_pose)

        safety.check(drone_pose)
        boost=1.8 if(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.

        cmd=Cmd(); cube_cmd=CubeCmd()

        if hover_now:
            pid_tracker.reset(); lqr_tracker.reset(); reacq.reset()

        elif mode in("AUTO","AUTO_CUBE"):
            if target is not None and lost_s<0.16:
                if args.ctrl=="pid":
                    cmd=pid_tracker.step(target,frame.shape,drone_pose,control_cube_pose,dt,kalman,lim,
                                         body_xyz=control_body_xyz)
                    lqr_tracker.reset()
                else:
                    cmd=lqr_tracker.step(target,frame.shape,drone_pose,control_cube_pose,dt,kalman,lim,
                                         body_xyz=control_body_xyz)
                    pid_tracker.reset()
                reacq.reset()
            else:
                # Reacquisition takes over when visual detection is temporarily lost.
                cmd=reacq.step(drone_pose,kalman,grid27,imt,body_hist,reacq_fsm,now,dt)
                pid_tracker.reset(); lqr_tracker.reset()

        elif mode=="MANUAL_DRONE":
            sp=CFG.manual_speed*boost; yw=CFG.manual_yaw*boost
            if keys[pygame.K_w]: cmd.vx+=sp
            if keys[pygame.K_s]: cmd.vx-=sp
            if keys[pygame.K_d]: cmd.vy+=sp
            if keys[pygame.K_a]: cmd.vy-=sp
            if keys[pygame.K_r]: cmd.vz-=sp
            if keys[pygame.K_f]: cmd.vz+=sp
            if keys[pygame.K_q]: cmd.yaw-=yw
            if keys[pygame.K_e]: cmd.yaw+=yw

        cmd.vz=safety.altitude_correction(cmd.vz,drone_pose)

        if mode=="AUTO_CUBE":
            sp=CFG.cube_speed*boost
            if keys[pygame.K_UP]:     cube_cmd.vx+=sp
            if keys[pygame.K_DOWN]:   cube_cmd.vx-=sp
            if keys[pygame.K_RIGHT]:  cube_cmd.vy+=sp
            if keys[pygame.K_LEFT]:   cube_cmd.vy-=sp
            if keys[pygame.K_PAGEUP]:   cube_cmd.vz-=sp
            if keys[pygame.K_PAGEDOWN]: cube_cmd.vz+=sp
            move_cube(client,args.cube_name,cube_cmd,dt)
        elif scenario.enabled and mode in("AUTO","AUTO_CUBE"):
            cube_cmd=scenario.step(now)
            move_cube(client,args.cube_name,cube_cmd,dt)

        await vehicle.send(cmd)
        if metrics is not None:
            metrics.row(now=now,mode=mode,target=current_target if body_est is not None else None,frame_shape=frame.shape,
                        lost_s=lost_s,reacq=reacq,bx=bx,by=by,bz=bz,dist=dist,
                        cmd=cmd,lim=lim,cube_cmd=cube_cmd)
        seq+=1
        if current_target is not None:
            fh, fw=frame.shape[:2]
            err_x=float(current_target.cx-fw*0.5)
            err_y=float(current_target.cy-fh*CFG.aim_y_ratio)
            img_err=math.sqrt(err_x*err_x+err_y*err_y)
        else:
            err_x=0.0; err_y=0.0; img_err=0.0
        hud_history.append({
            "t": now-app_start_t,
            "visible": body_est is not None,
            "dist_err_m": dist-CFG.target_dist_m if body_est is not None else None,
            "err_x_px": err_x,
            "err_y_px": err_y,
            "img_err_px": img_err,
            "cmd_yaw_dps": cmd.yaw,
        })
        if seq % 30 == 0:
            hud_logs.append(f"{time.strftime('%H:%M:%S')} INFO  Depth: {dist:.1f} m")
            hud_logs.append(f"{time.strftime('%H:%M:%S')} INFO  Visible: {current_target is not None}")

        intent=reacq.intent
        rb_yaw,rb_vy,rb_vz,rb_fx,rb_edge=reacq.return_bias
        lines=[
            f"state:{vehicle.state.name} mode:{mode} ctrl:{ctrl_name} hover:{hover_now}",
            f"target:{'YES' if current_target else 'NO '} lost:{lost_s:.2f}s reacq:{reacq.mode.name}",
            f"zone:{grid27.zone_label} edge:{imt.last_edge} reason:{reacq.reason}",
            f"intent: side:{intent.side} vert:{intent.vertical} up_guard:{intent.upward_guard}",
            f"return: yaw={rb_yaw:+.1f} vy={rb_vy:+.2f} vz={rb_vz:+.2f} fx={rb_fx:.2f} e={rb_edge:.2f}",
            f"body bx={bx:+.2f} by={by:+.2f} bz={bz:+.2f} dist={dist:.1f} z={z_now:.2f}",
            f"adapt: scale={asp.speed_scale:.2f} cube_v={asp.cube_speed_ms:.1f}m/s eb_y={asp.edge_boost_y:.2f}",
            f"lim: vx={lim.vx:.1f} vy={lim.vy:.1f} vz={lim.vz:.1f} yaw={lim.yaw:.0f}",
            f"cmd: vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} vz={cmd.vz:+.2f} yaw={cmd.yaw:+.1f}",
            f"cube vx={cube_cmd.vx:+.2f} vy={cube_cmd.vy:+.2f} vz={cube_cmd.vz:+.2f}",
            "1 AUTO | 2 MANUAL | 3 AUTO+CUBE | H hover | L land | ESC quit",
        ]
        if args.hud_style=="operator":
            vis=draw_operator_hud(
                frame,
                depth=depth_frame,
                det=current_target,
                lim=lim,
                asp=asp,
                history=hud_history,
                logs=hud_logs,
                vehicle_state=vehicle.state.name,
                mode=mode,
                ctrl_name=ctrl_name,
                tracker_name=f"YOLO{' + ByteTrack' if args.bytetrack else ''}",
                reacq_mode=reacq.mode.name,
                bx=bx,
                by=by,
                bz=bz,
                dist=dist,
                z_now=z_now,
                cmd=cmd,
                lost_s=lost_s,
                drone_pose=drone_pose,
                metrics_log=args.metrics_log,
                app_time=now-app_start_t,
                seq=seq,
                safety_status="OK" if not hover_now else "HOVER",
            )
        else:
            vis=frame.copy()
            draw_hud(vis,lines,current_target,lim,asp)
        ww,wh=screen.get_size()
        vis_s=(cv2.resize(vis,(ww,wh),cv2.INTER_LINEAR)
               if (ww,wh)!=(vis.shape[1],vis.shape[0]) else vis)
        screen.blit(to_surface(vis_s),(0,0)); pygame.display.flip(); clock.tick(30)
        await asyncio.sleep(0)

    log.info("Shutting down ...")
    if args.land_on_finish:
        try:
            await vehicle.land()
            await asyncio.sleep(max(0.0, args.landing_wait))
            await vehicle.wait_landed(timeout_s=max(3.0, args.landing_wait))
        except Exception: pass
    else:
        try: await vehicle.hold()
        except Exception: pass
    try: await vehicle.disarm()
    except Exception: pass
    if args.reset_poses:
        reset_test_poses(client,args.cube_name,args.drone_start_pose,args.cube_start_pose,args.pose_settle_s,"Final")
    if metrics is not None:
        metrics.close()
    pygame.quit()

def main():
    asyncio.run(async_main())

if __name__=="__main__":
    main()
