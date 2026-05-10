# Tracking Architecture Notes

This file is a compact technical note. The main setup and launch instructions are in `README.md`.

## Core Pipeline

1. AirSim provides RGB and optional depth frames.
2. YOLO detects the target in the RGB image.
3. ByteTrack can keep a stable target ID between frames.
4. AirSim object poses provide simulation ground truth for target distance.
5. Kalman and motion-history modules estimate short-term target motion.
6. PID or LQR converts image/body-frame errors into body velocity and yaw-rate commands.
7. MAVSDK sends commands to PX4 in offboard mode.
8. The metrics logger writes frame-by-frame CSV rows for later comparison.

## Main Modules

- `uav_tracking/app.py` - runtime orchestration.
- `uav_tracking/config.py` - all tuning parameters.
- `uav_tracking/controllers/pid.py` - PID controller.
- `uav_tracking/controllers/lqr.py` - LQR controller.
- `uav_tracking/reacquisition.py` - target return-to-frame logic.
- `uav_tracking/estimation.py` - Kalman, image motion, body motion and spatial prediction.
- `uav_tracking/speed.py` - adaptive command limits.
- `uav_tracking/hud.py` - operator and classic HUD views.

## Distance Estimate

During live AirSim tracking the main distance estimate is computed from AirSim poses:

- drone pose: `simGetVehiclePose()`;
- target pose: `simGetObjectPose(cube_name)`;
- body-frame distance: transformed world delta between target and drone.

The depth camera is used for visualization and for separate validation tests. This is intentional: AirSim pose gives stable simulation ground truth, while depth ROI tests show how the system could move toward a real sensor-based distance estimate.

## Reacquisition

When YOLO temporarily loses the target, `PredictiveReacq` combines:

- Kalman prediction of target position;
- body-frame motion history;
- image-plane motion history;
- 27-zone spatial prediction;
- last known frame edge.

The output is a conservative command that tries to return the target to the camera frame instead of continuing blindly forward.
