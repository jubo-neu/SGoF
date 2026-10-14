# Integrating the Five Official Baselines

This project manages run records and evaluation. External baselines use their official implementations; the reference renderer is not a substitute for a baseline. A Windows CPU environment does not replace the Linux/CUDA environments required by these methods.

| Method | Official source | Integration notes |
|---|---|---|
| 3DGS | https://github.com/graphdeco-inria/gaussian-splatting | Uses `train.py` and `render.py`; the default evaluation split does not automatically match the shared manifest. |
| CityGaussianV2 | https://github.com/Linketic/CityGaussian | Confirm the version through the V2 project page, retain its 2D Gaussian pipeline, and avoid accidentally using the older version. |
| AnySplat | https://github.com/OpenRobotLab/AnySplat | Follow official weight and context requirements. Configure actual commands after checking the local version; do not invent entry points. |
| HiCo-GS | https://github.com/WZ-CS/HiCo-GS | Recorded entry points: `scripts/train.sh` and `scripts/render_and_evaluate.sh`; the reviewed public configuration uses four RTX 4090 GPUs and 30,000 steps. |
| TopoGS | https://github.com/WZ-CS/TopoGS | Recorded entry points: `scripts/train_mill19.sh` and `scripts/render.sh`; the reviewed public example uses four processes and 100,000 steps. |

Public configuration values are not local measurements. Using one A800 or changing the GPU count changes the resource configuration and must not be described as an identical official run.

## Workflow

1. Prepare each official repository, CUDA extensions, and weights in a separate environment. Pin the commit and record modifications.
2. Use shared training/test manifests and preserve dataset-loader adaptation patches to avoid differences between default splits.
3. Copy `configs/external_task.example.json` and fill in actual absolute paths, explicit `argv`, hardware, and budgets.
4. Preview the task with `run-external`, then add `--execute` to launch it.
5. Render the shared test views, create the CSV described in `DATA_CONTRACT.md`, and evaluate using this project's common metrics.

Controlled initialization requires explicitly adapting the original 3DGS entry point: load `initial_gaussians.ply`, use SH degree 0 or correctly expand the coefficients, supply matching cameras and calibration, and retain optimization, densification, and pruning. The PLY contains Gaussians only and cannot replace standard camera files. Preserve and validate the CUDA adaptation patch. This integration has not been validated on the local machine; successful export alone does not establish successful completion of an official 3DGS experiment.
