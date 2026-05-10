# GitHub Export Manifest

Upload these files and folders:

- `.gitignore`
- `README.md`
- `requirements.txt`
- `settings.json`
- `TEST_PLAN_TRACKING.md`
- `README_tracking.md`
- `px4_yolo_mavsdk_final.py`
- `uav_tracking/`
- `scripts/`
- `benchmarks/`

Do not upload:

- `.venv/`
- `.idea/`
- `runs/`, `runs_target_drone_rf/`
- `*.pt`, `*.onnx`, `*.engine`
- `dataset_target_drone_fast_precise/`, `yolo_redcube_fast/`, `train/`, `valid/`, `test/`, `train_split/`
- `metrics_results/`
- `thesis_output/`
- `tools/`
- local thesis templates and generated `.docx` files
- old one-file prototype scripts from the root directory

Recommended first commit:

```powershell
git init
git add .gitignore README.md requirements.txt settings.json TEST_PLAN_TRACKING.md README_tracking.md uav_tracking scripts benchmarks
git commit -m "Prepare UAV tracking project for thesis demo"
```
