import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_tracking.app import main

if "--ctrl" not in sys.argv:
    sys.argv.extend(["--ctrl", "lqr"])

if __name__ == "__main__":
    main()
