# Spleen Segmentation MLOps Pipeline

[![CI](https://github.com/Adityajaiswal9050/spleen-segmentation-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Adityajaiswal9050/spleen-segmentation-pipeline/actions/workflows/ci.yml)

A solo, hands-on MLOps/robotics-perception project: a 3D U-Net (MONAI) trained
to segment the spleen from abdominal CT, wrapped in the full lifecycle a real
deployment needs — serving, explainability, uncertainty, robustness testing,
versioning, containerization, ROS2 integration, and CI. Built for interview
prep, not a class assignment — every number below is from a real run, not a
projection, including the ones that came out worse than hoped.

## Architecture

```mermaid
flowchart TB
    subgraph training["Training (src/)"]
        A["train_spleen.py<br/>CPU, 50 epochs, 4-level UNet"] -->|"val Dice 0.4603"| CKPT[("Production checkpoint")]
        A2["train_spleen_colab.py<br/>GPU, 5-level UNet, 300-epoch schedule"] -.->|"100/300 epochs run,<br/>val Dice 0.2127, worse, not shipped"| FAIL["docs/FAILURE_ANALYSIS.md<br/>(GPU retrain writeup lives in<br/>experiments/gpu_run_100ep/)"]
        A3["kfold_colab.py (src/) and<br/>train_spleen_ddp_kaggle.py (src/)<br/>GPU robustness + DDP demo"] -.->|"pending real run"| PEND[("Not yet executed")]
    end

    CKPT --> EVAL["src/evaluate_spleen.py"] --> LOG[("results/benchmark_log.csv")]
    LOG --> VER["scripts/version_and_benchmark.py"] --> VSTATE[("version_state.json")]
    VSTATE --> CHART["results/dice_vs_version.png"]

    CKPT --> API["src/serve_api.py<br/>FastAPI predict + metrics"]
    CKPT --> ONNX["analysis/onnx_export_benchmark.py<br/>ONNX Runtime export"]
    CKPT --> GRAD["analysis/gradcam_spleen.py"]
    CKPT --> MC["analysis/mc_dropout_uncertainty.py"]
    CKPT --> FAILCASE["analysis/failure_case_analysis.py"]
    CKPT --> NOISE["analysis/noise_robustness_test.py"]

    subgraph ros2_stack["ROS2 (Docker: ros-jazzy-ros-base)"]
        CAM["image_publisher.py<br/>1.0 Hz"] --> BRIDGE["segmentation_bridge_node.py"]
        CKPT --> BRIDGE
        BRIDGE --> RQT["rqt_graph and rqt_image_view"]
    end

    subgraph automation["Automation"]
        CRON["cron, 2am daily"] --> NIGHTLY["scripts/run_nightly_eval.sh"] --> DOCKER["Docker: spleen-eval image"] --> EVAL
    end

    subgraph ci_cd["CI/CD"]
        GH["GitHub Actions"] --> T1["tests/test_version_and_benchmark.py"]
        GH --> T2["tests/test_model_inference.py"]
    end

    subgraph versioning["Versioning"]
        DVC["DVC: data.dvc, checkpoints.dvc"] -.-> CKPT
    end
```

## Results — real, measured, not rounded up

| Phase / item | Metric | Result |
|---|---|---|
| Camera calibration (OpenCV) | Reprojection error | **0.4087** (13/14 sample images; 1 corrupted) |
| CPU baseline training (`src/train_spleen.py`) | Val Dice, 50 epochs | **0.4603** — production checkpoint |
| GPU retrain attempt (`src/train_spleen_colab.py`) | Val Dice, 100/300 epochs | **0.2127** — worse, not shipped ([why](experiments/gpu_run_100ep/FAILURE_ANALYSIS.md)) |
| Noise robustness (`analysis/noise_robustness_test.py`) | Dice @ noise std 0 / 0.05 / 0.10 / 0.20 / 0.30 | 0.4603 / 0.4573 / 0.4496 / 0.3764 / 0.2784 |
| K-fold cross-validation (`src/kfold_colab.py`) | Mean ± std Dice, 3 folds | **Pending** — script written, not yet run (needs Colab GPU) |
| ONNX export (`analysis/onnx_export_benchmark.py`) | Latency, single 64³ patch | PyTorch 316ms vs. ONNX Runtime 68.6ms (**4.61x**), both under the ROS2 node's 1.0Hz budget |
| CI (`tests/test_version_and_benchmark.py` + `tests/test_model_inference.py`) | Tests passing | **8/8**, real inference smoke test included, not just versioning arithmetic |
| Semantic versioning | Current version | `1.0.2` (see `results/dice_vs_version.png`) |
| Distributed training demo (`src/train_spleen_ddp_kaggle.py`) | Real 2-GPU DDP run | **Pending** — script written, not yet run (needs Kaggle dual-T4) |
| Cloud deployment | Public endpoint | **Not attempted** — deliberately deferred, not claimed |

The GPU retrain is the most important honest result here: bigger model,
finer spacing, and heavier augmentation *underperformed* the small CPU
baseline because the run was cut off at 100 of a 300-epoch planned LR
schedule. The CPU checkpoint (0.4603) remains what's actually in production.

## Phase-by-phase evidence

**Phase 2 — Camera calibration**
| Detected corners | Undistortion comparison |
|---|---|
| ![corners](calibration/detected_corners_sample.jpg) | ![undistort](calibration/undistortion_comparison.jpg) |

Per-image reprojection error: ![reprojection error](calibration/reprojection_error_per_image.png)

**Phase 3 — Segmentation predictions** (`src/visualize_predictions.py`, CPU baseline checkpoint)
![prediction case 1](results/prediction_case1.png)

**Phase 4 — ROS2 integration** (`ros2/camera_calib_pkg/segmentation_bridge_node.py`; runs the real 3D CT model
against the 2D camera feed with a documented input-format caveat — see the
module docstring in that file for why that mismatch is disclosed, not hidden)
| rqt_graph | rqt_image_view |
|---|---|
| ![rqt graph](ros2/ros2_rqt_graph.png) | ![rqt image view](ros2/ros2_rqt_image_view.png) |

**Phase 5 — Docker + cron automation**
| `docker ps` | `docker images` | nightly log |
|---|---|---|
| ![docker ps](automation_evidence/evidence_docker_ps.png) | ![docker images](automation_evidence/evidence_docker_images.png) | ![nightly log](automation_evidence/evidence_nightly_log.png) |

**Phase 6 — Semantic versioning tied to benchmarks**
![dice vs version](results/dice_vs_version.png)

**Tier 2 — Grad-CAM explainability**
![gradcam case 1](results/gradcam_case1.png)

**Tier 2 — MC dropout uncertainty**
![mc dropout case 1](results/mc_dropout_case1.png)

**Tier 2 — Honest failure-case analysis** (worst real Dice cases, not cherry-picked —
see [docs/FAILURE_ANALYSIS.md](docs/FAILURE_ANALYSIS.md) for the full writeup, including a
hypothesis about spleen size that the data itself disproved)
![failure case 1](results/failure_case_1.png)

**Tier 2 — Noise robustness**
![noise robustness](results/noise_robustness.png)

**Tier 2 — GPU retrain failure analysis**
![gpu retrain training curve](experiments/gpu_run_100ep/training_curve.png)

**Tier 2 — FastAPI serving + monitoring**: `src/serve_api.py` exposes `/predict`
(real inference + Grad-CAM-style preview), `/health`, and `/metrics`
(request/error/drift counts, latency), plus a drift check on incoming CT
intensity stats against the real training distribution (mean 0.1757,
measured across the 33 training cases, not guessed).

## What's not done, honestly

- **K-fold cross-validation** and the **DDP distributed-training demo** are
  written (`src/kfold_colab.py`, `src/train_spleen_ddp_kaggle.py`) but not yet
  executed — both need a GPU this VM doesn't have. Results will be added
  once run on Colab/Kaggle and brought back.
- **Cloud deployment** (Render/Fly.io) was deliberately skipped for now
  rather than rushed — no public endpoint currently exists.
- The **W&B run URL** from the GPU retrain session wasn't preserved, so it
  isn't linked here — the run's real outputs (`experiments/gpu_run_100ep/training_log.csv`,
  `experiments/gpu_run_100ep/training_curve.png`) are committed and are what the Dice
  number above is based on.

## Setup

```bash
# clone + create the venv (has MONAI, torch, fastapi, onnxruntime, etc.)
python3 -m venv monai-env && source monai-env/bin/activate
pip install monai torch nibabel fastapi uvicorn python-multipart matplotlib onnxruntime pytest

# pull data/checkpoints (DVC-tracked, not in git)
dvc pull

# run the CPU baseline eval (writes to results/benchmark_log.csv)
python src/evaluate_spleen.py

# run the API (--app-dir makes src/ importable without a src/__init__.py)
uvicorn serve_api:app --app-dir src --reload
curl http://127.0.0.1:8000/health

# run tests (same as CI; pytest.ini puts src/, scripts/, analysis/ on
# sys.path so tests/ can import sibling modules like version_and_benchmark)
pytest tests/ -v
```

## Repo layout

- `src/` — training (`train_spleen.py`, `train_spleen_colab.py`, `train_spleen_ddp_kaggle.py`, `kfold_colab.py`), evaluation (`evaluate_spleen.py`), visualization (`visualize_predictions.py`), and serving (`serve_api.py`)
- `scripts/` — operational glue: `run_nightly_eval.sh`, `version_and_benchmark.py` (semantic versioning gated on real Dice), `plot_dice_vs_version.py`
- `analysis/` — explainability, uncertainty, and robustness: `gradcam_spleen.py`, `mc_dropout_uncertainty.py`, `failure_case_analysis.py`, `onnx_export_benchmark.py`, `noise_robustness_test.py`
- `tests/` — `test_version_and_benchmark.py`, `test_model_inference.py` (run via `pytest tests/`)
- `results/` — generated outputs (`.png`/`.csv`/`.txt`): benchmark log, version history, Dice-vs-version chart, Grad-CAM/MC-dropout/failure-case/prediction images, ONNX benchmark output
- `docs/` — `FAILURE_ANALYSIS.md` (CPU baseline's honest failure-case writeup)
- `experiments/gpu_run_100ep/` — the failed GPU retrain: checkpoint, training log/curve, its own failure analysis and eval script
- `ros2/` — camera publisher, undistortion node, model-serving bridge node (`camera_calib_pkg/` layout untouched)
- `calibration/` — OpenCV chessboard calibration
- `.github/workflows/ci.yml` — versioning + real model-inference tests on every push
- `Dockerfile` / `Dockerfile.api` — eval and serving images (code/data mounted at runtime via `-v`, not copied in)
