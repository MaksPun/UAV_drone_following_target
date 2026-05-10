"""Compatibility entry point for the refactored PX4/YOLO/MAVSDK tracker.

The implementation now lives in the uav_tracking package.
The original monolithic script is preserved in legacy/px4_yolo_mavsdk_final_monolith.py.
"""

from uav_tracking.app import main


if __name__ == "__main__":
    main()
