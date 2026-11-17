# GraphMem results — closed-world OSNOM benchmark

Setting: the benchmark's own task as reported by Whareformer and LMK. Ground-truth object boxes, the released per-box
features, the official test splits, and the official metrics (`src/whareformer_eval/`): **mPCL** and **IDF1**.
GraphMem uses **no 3D reconstruction, camera pose or depth**; Whareformer and LMK use offline SfM / mesh 3D locations.
GraphMem's settings were chosen on EPIC-KITCHENS train videos and applied unchanged to every dataset.

| Dataset (test split) | Method | 3D input | mPCL | IDF1 |
|---|---|---|---|---|
| EPIC-KITCHENS (54 videos) | **GraphMem** | none | **72.1** | **84.0** |
| | Whareformer | offline SfM + mesh | 72.2 | 84.0 |
| | LMK (published) | offline SfM + mesh | 53.0 | 70.0 |
| HD-EPIC (156 videos) | **GraphMem** | none | **37.5** | **51.6** |
| | Whareformer | offline 3D | 41.6 | 51.6 |
| IT3DEgo (50 videos) | **GraphMem** | none | **31.5** | **76.9** |
| | Whareformer | offline 3D | 53.7 | 87.9 |

Paired 95 % bootstrap intervals over videos, GraphMem − Whareformer:

| Dataset | Δ mPCL | Δ IDF1 |
|---|---|---|
| EPIC-KITCHENS | −0.1 [−4.5, +4.4] | +0.0 [−3.2, +3.1] |
| HD-EPIC | −4.1 [−5.6, −2.6] | +0.0 [−0.9, +0.8] |
| IT3DEgo | −22.2 [−26.1, −12.7] | −11.0 [−14.4, −7.2] |

- On EPIC-KITCHENS GraphMem matches Whareformer on both metrics without any 3D input, and it ties on HD-EPIC IDF1.
- IT3DEgo scenes contain many identical copies of the same object that only position can separate; there an
  appearance-only memory is at a disadvantage.
- `closed_world_epic_per_video.csv`: per-video mPCL and IDF1 of GraphMem and Whareformer on the EPIC-KITCHENS test split.

Reproduce (EPIC-KITCHENS, closed world): see `src/analysis/milestones/m3/README.md`.
