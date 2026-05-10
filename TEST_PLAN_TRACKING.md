# Tracking metric test plan

This project currently measures target distance from AirSim object poses:

- `drone_pose = client.simGetVehiclePose()`
- `cube_pose = client.simGetObjectPose(args.cube_name)`
- body-frame distance is computed from the difference between those poses.

So the live tracker does **not** currently use the built-in depth camera for the
main distance estimate. The depth benchmark is included to justify this choice
and to compare possible future estimators: ground-truth pose, ROI median depth,
center-pixel depth, and inverse bbox size.

## Run all offline metrics

```powershell
python benchmarks/run_all_metrics.py
```

Outputs are written to `metrics_results/` as `.csv` and `.md` tables.

## Run live AirSim metrics

Single controlled AirSim run:

```powershell
python scripts/run_tracking.py --ctrl lqr --cube_name BP_TargetCube_5 --metrics_scenario lateral_sine --metrics_duration 20 --metrics_label lqr_lateral_sine_run1 --metrics_log metrics_results/airsim_runs/lqr_lateral_sine_run1.csv
```

The same scenario with PID:

```powershell
python scripts/run_tracking.py --ctrl pid --cube_name BP_TargetCube_5 --metrics_scenario lateral_sine --metrics_duration 20 --metrics_label pid_lateral_sine_run1 --metrics_log metrics_results/airsim_runs/pid_lateral_sine_run1.csv
```

Batch suite for multiple scenarios:

```powershell
python scripts/run_airsim_metric_suite.py --cube_name BP_TargetCube_5 --duration 20 --controllers pid lqr
```

Resume only LQR runs after PID logs were already collected:

```powershell
python scripts/run_airsim_metric_suite.py --cube_name BP_TargetCube_5 --duration 20 --controllers lqr
```

Analyze live AirSim logs:

```powershell
python benchmarks/analyze_airsim_metrics.py metrics_results/airsim_runs/*.csv --out metrics_results/airsim_summary
```

AirSim metric scenarios currently available:

- `static`
- `lateral_sine`
- `depth_step`
- `vertical_step`
- `zigzag`
- `frame_exit_right`
- `frame_exit_left`
- `frame_exit_top`
- `frame_exit_bottom`
- `occlusion_like`

For YOLO tracker comparison, run the same AirSim scenario twice: once without
`--bytetrack`, once with `--bytetrack`. The analyzer reports `id_switches`,
`visible_ratio`, `lost_episodes`, and recovery time.

## Experiments

### 1. PID vs LQR controller tracking

```powershell
python benchmarks/run_controller_metrics.py
```

Metrics:

- mean absolute distance error;
- mean lateral/altitude error;
- image RMSE;
- visible-frame ratio;
- settling time;
- p95 yaw command.

Scenarios:

- static offset;
- lateral sine movement;
- forward/backward depth step;
- vertical step;
- zigzag target movement.

### 2. YOLO selection with and without ByteTrack

```powershell
python benchmarks/run_yolo_tracker_metrics.py
```

Metrics:

- true target ratio;
- lost frames;
- ID switches;
- selected frames.

Scenarios:

- crossing distractor;
- occlusion and confidence drop.

### 3. Target return to frame

```powershell
python benchmarks/run_reacquisition_metrics.py
```

Compared policies:

- scan only;
- edge-only return;
- predictive return used by `PredictiveReacq`.

Metrics:

- success;
- recovery time;
- mean image error;
- p95 horizontal/vertical image error.

### 4. Distance estimation

```powershell
python benchmarks/run_distance_depth_metrics.py
```

Compared methods:

- AirSim pose ground truth;
- simulated ROI median depth;
- simulated center-pixel depth;
- inverse bbox-size estimate.

For the thesis, this table helps explain why pose-based distance is convenient
in simulation, while a real drone should prefer calibrated depth/stereo/range
sensors or a fused estimator.
