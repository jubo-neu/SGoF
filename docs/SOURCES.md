# Comparison Literature Review Record (2026-10-14)

The following information is translated from the project's existing source review record. Publication details, repository availability, and entry points were not independently rechecked during this documentation translation.

| Method | Recorded review sources | Year notes | Recorded public implementation status |
|---|---|---|---|
| 3DGS | https://arxiv.org/abs/2308.04079 | 2023 | Official repository: https://github.com/graphdeco-inria/gaussian-splatting |
| CityGaussianV2 | https://arxiv.org/abs/2411.00771 ; project page: https://dekuliutesla.github.io/CityGaussianV2/ | First preprint in 2024; arXiv Comments identifies ICLR 2025, so the publication year is recorded as 2025 | The project page links to https://github.com/Linketic/CityGaussian |
| AnySplat | https://arxiv.org/abs/2505.23716 ; project page: https://city-super.github.io/anysplat/ | First submission recorded as 2025-05-29 | The project page links to https://github.com/OpenRobotLab/AnySplat |
| HiCo-GS | https://arxiv.org/abs/2608.14136 ; official repository: https://github.com/WZ-CS/HiCo-GS | 2026; the official repository identifies ACM MM 2026 | The recorded web directory review found training, rendering, configurations, and CUDA submodules; official GPU experiments were not run locally. |
| TopoGS | https://arxiv.org/abs/2609.27868 ; official repository: https://github.com/WZ-CS/TopoGS | 2026; the previous review records an arXiv submission date of 2026-08-20, a September announcement index, and official README news dated September 24. These date details should be checked against the source before citation. | The recorded web directory review found `train.py`, `render.py`, and `scripts`; official GPU experiments were not run locally. |

Later candidates were recorded in the review but excluded from the main comparison under the reproducibility-first selection policy: CompactSplat (https://arxiv.org/abs/2610.13177, recorded date 2026-10-09), DeltaSplat (https://arxiv.org/abs/2610.09853, recorded date 2026-10-07), EffGS (https://arxiv.org/abs/2609.39553), and Budgeted-GS (https://arxiv.org/abs/2610.03162). The first two were described as more oriented toward sparse views; their official implementations were not verified in the recorded review. The selection criterion was work from 2026 with matching datasets and public implementations, rather than exhaustive coverage of all the latest work as of the review date.

At the time of the recorded review, the official SGoF repository at https://github.com/jubo-neu/SGoF contained only `README.md`. Models, configurations, datasets, weights, and training/testing code were listed as TODO. All SGoF modules in this project are independent reference implementations based on the available manuscript.

The recorded public experiments for HiCo-GS and TopoGS include Residence and Sci-Art from UrbanScene3D. The reference manuscript uses ArtSci and PolyTech; PolyTech must not be assumed to be Residence. Before formal experiments, confirm scene directories, training/test splits, and camera files with the dataset publisher or authors. This project retains the manuscript scene names and treats identity verification as a prerequisite.

## Supporting References

- VGGT: https://arxiv.org/abs/2503.11651
- Continuous 6D rotation representation: https://arxiv.org/abs/1812.07035
- LPIPS: https://arxiv.org/abs/1801.03924

The separately prepared English and Chinese manuscript sections use local references. When integrating those sections into the original paper, deduplicate them against the authors' existing bibliography. Those manuscript sections are separate artifacts and are not included in this code release.
