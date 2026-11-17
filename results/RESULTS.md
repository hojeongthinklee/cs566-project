# GraphMem: Object Persistence in Egocentric Video Without 3D

**Summary.** GraphMem is an online appearance-memory graph that keeps every object's identity across a single
egocentric video using no camera pose, depth or 3D reconstruction. On the OSNOM benchmark's closed-world setting
(ground-truth boxes, as reported by Whareformer and LMK) it matches Whareformer, which relies on offline SfM and a
mesh, on EPIC-KITCHENS (72.1 mPCL / 84.0 IDF1 vs 72.2 / 84.0) and on HD-EPIC IDF1, and clearly exceeds LMK.
With SAM 3.1 detections instead of ground-truth boxes it ties offline-3D Whareformer and outperforms every
streaming-3D baseline, while running 3.8× faster with 5.1× less memory on the longest video.

## 1. Task

One egocentric video at a time: every object must keep the same identity whenever it is seen again, after minutes
out of view and after being moved. Labels are produced online from the past only, with at most a short delay.
Whareformer, the state of the art on OSNOM, needs a camera pose and a 3D location for every detection. GraphMem
asks whether a 2D memory with no 3D at all can persist identities as well.

## 2. Method

- **Identity nodes** hold a reservoir of up to 256 appearance views of the object.
- **Delayed decision:** a short-term track is matched to memory, or becomes a new identity, after K observations;
  until then its observations carry a provisional label.
- **Cannot-link:** identities visible in the same frame are never merged.
- **Closed world (ground-truth boxes):** this core alone, K = 100; a candidate is scored by its best-matching view.
- **Open world (SAM 3.1 detections, 17 text prompts, DINOv2-giant masked-crop features):** in addition,
  - a learned short-term linker (trained on train videos) builds tracklets across 2 fps keyframes;
  - co-visibility edges, place nodes and the detector prompt as a soft object kind enter the score as small bonuses;
  - hierarchical search: places are indexed by the object kinds of their identities and searched first, falling
    back to the full memory when the local match is not clear;
  - K = 10 observations (about 5 s at 2 fps).

All settings were fixed on train videos before any test run; the same settings are used on every dataset.

## 3. Closed world — the benchmark setting

Ground-truth boxes, released per-box features, official test splits and the official scorer
(`src/whareformer_eval/`). Brackets: paired 95 % bootstrap intervals over videos, GraphMem − Whareformer.

| Dataset (test split) | Method | 3D input | mPCL | IDF1 |
|---|---|---|---|---|
| EPIC-KITCHENS (54 videos) | **GraphMem** | none | **72.1** | **84.0** |
| | Whareformer | offline SfM + mesh | 72.2 | 84.0 |
| | LMK (published) | offline SfM + mesh | 53.0 | 70.0 |
| HD-EPIC (156 videos) | **GraphMem** | none | **37.5** | **51.6** |
| | Whareformer | offline 3D | 41.6 | 51.6 |
| IT3DEgo (50 videos) | **GraphMem** | none | **31.5** | **76.9** |
| | Whareformer | offline 3D | 53.7 | 87.9 |

| Dataset | Δ mPCL | Δ IDF1 |
|---|---|---|
| EPIC-KITCHENS | −0.1 [−4.5, +4.4] | +0.0 [−3.2, +3.1] |
| HD-EPIC | −4.1 [−5.6, −2.6] | +0.0 [−0.9, +0.8] |
| IT3DEgo | −22.2 [−26.1, −12.7] | −11.0 [−14.4, −7.2] |

- With no 3D input, GraphMem ties Whareformer on EPIC-KITCHENS and on HD-EPIC IDF1.
- When Whareformer's offline 3D is replaced by streaming reconstruction, GraphMem leads on EPIC-KITCHENS:
  vs TTT3R 84.0 / 72.1 against 81.8 / 66.7 (mPCL +2.6 to +8.5), vs CUT3R against 80.4 / 65.5 (IDF1 +1.3 to +5.8).
- IT3DEgo scenes contain many identical copies of an object that only position can separate (see §6).
- Per-video scores on EPIC-KITCHENS: `closed_world_epic_per_video.csv`.

## 4. Open world — SAM 3.1 detections

Every method runs on the same detections. IDF1_cov is IDF1 on the ground-truth boxes a detection covers (coverage is
identical across methods); mPCL uses the benchmark's clean object locations, so it measures identity only.
Brackets: baseline − GraphMem.

**EPIC-KITCHENS (45 test videos)**

| Method | 3D input | IDF1_cov | mPCL | HOTA (AssA) |
|---|---|---|---|---|
| **GraphMem** | none | **76.7** | **45.5** | **19.1 (33.8)** |
| Whareformer, mesh | offline SfM + mesh | 76.3 [−2.5, +1.6] | 47.0 [−0.6, +3.6] | 19.2 (34.0) |
| Whareformer, fine-tuned on SAM detections | offline | 74.9 (per kitchen [−3.9, −0.6]) | 46.1 | 19.1 (33.9) |
| Whareformer, TTT3R | streaming | 72.7 [−6.8, −1.0] | 42.8 [−4.6, −0.7] | 18.7 (32.4) |
| LMK, mesh | offline | 73.1 [−5.9, −0.9] | 42.5 [−5.4, −0.5] | 17.7 (29.3) |

GraphMem ties offline-3D Whareformer on all three metrics and beats every streaming-3D baseline and LMK.
Asked "when did I last see this object?" (511,275 queries), GraphMem answers within 5 s in 26.6 % of cases vs
23.0 % for Whareformer (+3.6 [+1.1, +6.1]). Lowering the detector threshold from 0.5 to 0.4 (chosen on train) raises
coverage from 77 % to 82 % and gives IDF1 over all boxes +3.3 and mPCL +1.7 [+0.3, +2.8].

**HD-EPIC (27 videos, settings unchanged)**

| Method | IDF1_cov | mPCL | HOTA (AssA) |
|---|---|---|---|
| **GraphMem** | **72.0** | **21.5** | **7.2 (20.0)** |
| Whareformer, TTT3R | 74.2 [−0.3, +4.8] | 21.9 [−0.7, +1.8] | 6.9 (18.1) |

## 5. Delay and cost

GraphMem labels every observation online; the final label arrives when its track is decided.

| Lag allowed | 0 s | 1 s | 2 s | 5 s | final |
|---|---|---|---|---|---|
| Open world EPIC, IDF1_cov | 70.9 | 71.0 | 73.9 | 75.7 | 76.7 |
| Closed world EPIC, IDF1 | 80.1 | 81.8 | 83.6 | 83.8 | 84.0 |

Association cost (single CPU process, SAM inputs):

| Video | GraphMem | Whareformer (offline locations) | Speed-up | Memory |
|---|---|---|---|---|
| P04_26 | 2.7 s / 159 MB | 4.6 s / 346 MB | 1.7× | 2.2× |
| P04_114 | 7.9 s / 160 MB | 17.1 s / 567 MB | 2.2× | 3.6× |
| P04_03 | 30.9 s / 184 MB | 116 s / 829 MB | 3.8× | 4.5× |
| P04_02 (longest) | 72 s / 226 MB | 275 s / 1,162 MB | 3.8× | 5.1× |

The gap grows with video length, and Whareformer additionally needs its 3D input (offline SfM and a mesh, or a
streaming reconstructor on a GPU).

## 6. Limits

- Where position is the only cue, an appearance memory is at a disadvantage: on IT3DEgo, with many identical object
  instances, Whareformer leads (IDF1 87.9 vs 76.9). On HD-EPIC it keeps a 4-point mPCL edge at equal IDF1.
- The open-world accuracy assumes about 5 s of decision delay.
- Open-world results use one detector (SAM 3.1) and the EPIC-KITCHENS / HD-EPIC datasets.
- LMK open-world numbers come from our port of the method; closed-world LMK numbers are the published ones.

Reproduce: `src/analysis/milestones/m3/README.md`.
