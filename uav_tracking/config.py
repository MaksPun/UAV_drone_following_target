from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CFG:
    # Flight and safety.
    takeoff_alt:       float = 3.0
    arm_retry:         int   = 25
    takeoff_wait:      float = 7.0
    alt_floor_m:       float = 1.0
    geofence_r_m:      float = 90.0

    # Desired target geometry in the camera/body frame.
    target_dist_m:     float = 12.0
    target_alt_off:    float = 0.0
    aim_y_ratio:       float = 0.55

    # Baseline command limits for slow or static targets.
    base_vx:           float = 2.5
    base_vy:           float = 1.1
    base_vz:           float = 1.2
    base_yaw_dps:      float = 30.0

    # Upper command limits for aggressive target motion.
    max_vx:            float = 4.5
    max_vy:            float = 2.2
    max_vz:            float = 2.0
    max_yaw_dps:       float = 55.0

    # Adaptive speed scaling.
    cube_v_ref:        float = 4.0
    cube_a_ref:        float = 2.0
    edge_thr:          float = 0.55
    edge_yaw_boost:    float = 2.2
    edge_vy_boost:     float = 1.8
    edge_vz_boost:     float = 1.7
    edge_vx_boost:     float = 1.3
    adapt_alpha_up:    float = 0.35
    adapt_alpha_dn:    float = 0.08

    # PID gains and output shaping.
    pid_x:     tuple = (0.62, 0.040, 0.18)
    pid_y:     tuple = (0.36, 0.000, 0.10)
    pid_z:     tuple = (0.70, 0.040, 0.18)
    pid_yaw:   tuple = (24.0, 0.000, 5.2)
    pid_imgz:  tuple = (0.28, 0.000, 0.08)
    int_max_x: float = 0.70
    int_max_z: float = 0.60
    pid_deriv_alpha: float = 0.28
    pid_int_leak:    float = 0.992
    pid_aw_margin:   float = 0.92
    pid_smooth_vx: float = 0.30
    pid_smooth_vy: float = 0.30
    pid_smooth_vz: float = 0.27
    pid_smooth_yaw: float = 0.28
    pid_slew_vx:   float = 3.8
    pid_slew_vy:   float = 2.5
    pid_slew_vz:   float = 3.0
    pid_slew_yaw:  float = 46.0

    # LQR weights: (position error, error velocity, command effort).
    lqr_x:     tuple = (5.0, 2.5, 6.0)
    lqr_y:     tuple = (4.0, 2.0, 8.0)
    lqr_z:     tuple = (6.0, 2.8, 5.0)
    lqr_yaw:   tuple = (8.0, 2.0, 4.0)
    lqr_imgz:  tuple = (3.5, 1.5, 7.0)
    ki_x:      float = 0.032
    ki_z:      float = 0.040
    int_max:   float = 0.55
    int_decay_locked: float = 0.94
    lqr_recompute_dt_eps: float = 0.025
    lqr_adaptive_dt: bool = False

    # State observer.
    obs_pole:      float = 0.55
    obs_alpha_img: float = 0.16

    # Kalman-based feed-forward.
    ff_vel_scale:  float = 0.22
    max_ff_x:      float = 0.60
    max_ff_y:      float = 0.20
    max_ff_z:      float = 0.25
    max_ff_yaw:    float = 9.0
    ff_body:       float = 0.28
    max_ff_x_pid:  float = 0.65
    max_ff_y_pid:  float = 0.22
    max_ff_z_pid:  float = 0.28
    max_ff_yaw_pid:float = 10.0
    ff_vel_alpha:  float = 0.22
    ff_yaw_alpha:  float = 0.18

    # Small errors are ignored to reduce command jitter.
    db_x:    float = 0.28
    db_y:    float = 0.09
    db_z:    float = 0.07
    db_exi:  float = 0.015
    db_eyi:  float = 0.015

    # Lock zone near the desired target position.
    lock_near_m:   float = 0.55
    lock_near_y:   float = 0.10
    lock_near_z:   float = 0.10
    lock_near_xi:  float = 0.022
    lock_near_yi:  float = 0.022
    gain_near_scale: float = 0.20
    img_edge_gain_yaw: float = 0.55
    img_edge_gain_vy:  float = 0.35
    img_edge_gain_vz:  float = 0.30
    image_forward_brake: float = 0.55

    vx_near_damp:  float = 0.18
    vy_near_damp:  float = 0.10
    vz_near_damp:  float = 0.42
    yaw_near_damp: float = 0.16

    # Output filters.
    smooth_vx:  float = 0.20
    smooth_vy:  float = 0.20
    smooth_vz:  float = 0.18
    smooth_yaw: float = 0.18
    slew_vx:    float = 2.2
    slew_vy:    float = 1.4
    slew_vz:    float = 1.8
    slew_yaw:   float = 28.0

    # Reacquisition policy.
    reacq_short_s:    float = 0.30
    reacq_edge_s:     float = 0.80
    reacq_long_s:     float = 2.00
    reacq_max_vx:     float = 2.2
    reacq_max_vy:     float = 1.4
    reacq_max_vz:     float = 0.9
    reacq_max_yaw:    float = 38.0
    reacq_yaw_prio:   float = 18.0
    reacq_intercept:  float = 1.20
    reacq_orbit_yaw:  float = 1.25
    reacq_edge_boost: float = 1.25
    reacq_return_deadband: float = 0.08
    reacq_return_yaw_gain: float = 0.70
    reacq_return_vy_gain:  float = 0.45
    reacq_return_vz_gain:  float = 0.70
    reacq_forward_brake:   float = 0.60
    reacq_scan_yaw:        float = 0.42

    # Heuristics for targets leaving through the top of the frame.
    upward_guard_s:    float = 0.95
    up_min_climb:      float = 0.38
    up_strong_climb:   float = 0.58
    top_exit_thr:      float = -0.40
    side_exit_thr:     float = 0.42
    up_img_vel_thr:    float = -0.22
    up_body_vel_thr:   float = -0.35
    up_world_vel_thr:  float = -0.25
    dn_img_vel_thr:    float = 0.22
    dn_body_vel_thr:   float = 0.35
    near_up_bonus:     float = 0.18
    corner_yaw_bonus:  float = 7.0

    # Body-frame spatial bins used by predictive reacquisition.
    zone_x_near: float = 6.0
    zone_x_far:  float = 20.0
    zone_y_lat:  float = 2.5
    zone_z_alt:  float = 1.2

    kalman_proc: float = 0.30
    kalman_meas: float = 0.05

    infer_hz:    float = 10.0
    img_hist:    int   = 14
    body_hist_n: int   = 14

    manual_speed: float = 2.7
    manual_yaw:   float = 46.0
    cube_speed:   float = 3.2

    body_err_alpha: float = 0.22
    img_err_alpha:  float = 0.20


CFG = CFG()
