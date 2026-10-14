# SGoF
## [Paper] SGoF: Distilling Semantic-Geometric Occupancy Fields from Multimodal Scene Understanding for Large-Scale Gaussian Splatting
These are the official code releases.
- Graphical Abstract:
<img src="" alt="Teaser Image">


## Highlights


## TODO
- [x] Model Files
- [x] Configuration Files
- [x] Train and Eval Codes


## Features

- **Geometry:** Fuse extracted point features into voxel volumes and handle camera poses in the camera coordinate frame.
- **Models:** SaZ-Mod semantic modulation, an occupancy decoder, and a ViT-style vision-only student.
- **Training:** Decoder calibration, teacher target generation, student distillation, local weight loading, and checkpoint resumption.
- **Initialization:** Sample Gaussians from predicted occupancy, project colors, and export PLY files and camera tensors.
- **Evaluation:** PSNR, SSIM, LPIPS, and regional image metrics.


## Installation

Python 3.10 or newer is required. Use a dedicated environment and run the following commands from the repository root. On Linux, use `python3` if required by your environment.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

On Windows, if the repository path contains non-ASCII characters, set UTF-8 mode before installation and subsequent commands to avoid editable-install path decoding issues in older Python versions:

```powershell
$env:PYTHONUTF8="1"
```

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m sgof --help
```

For CUDA training, install a PyTorch build compatible with your hardware and driver. Student training in `configs/paper.json` defaults to `cuda`. For CPU use, set `train.device` to `cpu` and reduce the workload as needed.

Install the optional dependencies when using LPIPS:

```bash
python -m pip install -r requirements-optional.txt
```

LPIPS requires its official pretrained weights. This project does not automatically download VGGT, CLIP, or LLaVA weights.


## Start

### 1. Prepare Real Data and Geometry Caches

Prepare images, split lists, context manifests, pretrained geometry exports, and actual text embeddings according to the [Data Contract](docs/DATA_CONTRACT.md). Teacher feature extraction must use training images only; held-out test images must remain separate.

```bash
python tools/build_scene_split.py --scene Building --images data/Building/images --train-list data/Building/train.txt --test-list data/Building/test.txt --output splits/Building.json
python tools/prepare_geometry_cache.py --geometry-npz exports/train.npz --context contexts/train_window.json --text-embedding exports/clip.pt --config configs/paper.json --vggt-checkpoint-id YOUR_VGGT_ID --clip-checkpoint-id YOUR_CLIP_ID --output caches/Building.pt
```

The geometry archive `train.npz` must contain `points`, `features`, `depth`, `c2w`, and `intrinsics_normalized`. Point features must correspond to the geometry point by point, and the feature channel count must match the configuration. Text embeddings must come from an actual frozen pretrained model. Input caches are not included in this repository.

Example context manifest, with image paths relative to the JSON file:

```json
{
  "scene_id": "Building",
  "images": ["../data/Building/images/0001.png", "../data/Building/images/0002.png"],
  "bounds": [[-1.0, -1.0, 0.1], [1.0, 1.0, 4.0]],
  "uses_test_views": false,
  "intrinsics_normalized": [[0.8, 0.8, 0.5, 0.5], [0.8, 0.8, 0.5, 0.5]]
}
```

These bounds and intrinsics illustrate the schema only. Replace them with valid calibration and coordinate bounds for your experiment. `bounds` is expressed in the first-camera frame and must not be selected using held-out reference geometry.

### 2. Calibrate the Decoder and Generate Teacher Targets

```bash
python -m sgof calibrate-decoder --caches caches/Building.pt --config configs/paper.json --steps 100 --device cuda --output runs/decoder
python -m sgof fit-teacher --caches caches/Building.pt --decoder runs/decoder/decoder.pt --config configs/paper.json --steps 100 --device cuda --output runs/teacher
```

Teacher fitting writes `teacher.pt`, target caches such as `target_000.pt`, and training reports. Decoder calibration uses the geometric proxy `support/(support+1)` and is a supplementary implementation choice. The current teacher rendering path is intended for small-scale research checks. Large-scene experiments require integrating and validating an appropriate CUDA rendering backend.

### 3. Distill the Student

Create `targets/manifest.json` and list the generated teacher targets. Cache paths are relative to this manifest. A scene must not appear in both the distillation training and test splits.

```json
{
  "samples": [
    {"id": "Building_window_000", "scene": "Building", "split": "train", "cache": "../runs/teacher/target_000.pt"}
  ]
}
```

```bash
python -m sgof train-student --manifest targets/manifest.json --config configs/paper.json --output runs/student
```

Use `--encoder-weights weights/vit_tiny.pth` to load supported local ViT-Tiny encoder weights, or `--resume runs/student/student.pt` to resume training. Without supplied weights, training starts from scratch. 

### 4. Run Inference and Export Gaussian Initialization

```bash
python -m sgof initialize --checkpoint runs/student/student.pt --context contexts/new_scene.json --device cuda --tau 0.5 --output runs/initialization
```

This writes `initial_gaussians.ply`, `initialization.pt`, and `initialization_report.json`. Use `--max-points` to cap the number of primitives. 


## Comparison

The comparison includes these five external methods:

| Method | Year recorded in the project | Role |
|---|---:|---|
| 3D Gaussian Splatting | 2023 | Classic baseline |
| CityGaussianV2 | 2025 | Large-scene baseline |
| AnySplat | 2025 | Pose-free feed-forward baseline |
| HiCo-GS | 2026 | Large-scene baseline |
| TopoGS | 2026 | Large-scene baseline |

See [Official Baselines](docs/OFFICIAL_BASELINES.md), [Sources](docs/SOURCES.md), and `configs/methods.json` for the recorded publication and repository information. Run each baseline using its official implementation and environment.


## Image Evaluation

Within each protocol, all methods and seeds must evaluate identical scene/view sets. Optional fields include `sky_mask`, `far_mask`, `boundary_mask`, and `category`. Region masks should come from independent annotations or geometry.

```bash
python -m sgof evaluate --manifest runs/renders.csv --output runs/metrics
```

Add `--lpips --device cuda` to `evaluate` when perceptual metrics are needed. Evaluation produces per-view and per-scene/seed JSON results.


## Citation

No placeholder BibTeX entry is supplied because the exact bibliographic information has not been established in this release.
