# GraphMem — object persistence in egocentric video without 3D

GraphMem keeps an appearance memory graph of every object identity seen so far in one video and links each new
observation to it online. It uses no camera pose, depth or 3D reconstruction.

- **Identity nodes** hold a reservoir of up to 256 appearance views; a candidate is scored by its best-matching view.
- **Delayed decision:** a short-term track is assigned to an identity (or becomes a new one) after K observations.
- **Cannot-link:** identities visible in the same frame are never merged.

## Setup

1. Python 3.13, `pip install -r requirements.txt` (plus `motmetrics` and `pyyaml` for scoring).
2. Clone the official Whareformer repository next to this directory as `../Whareformer` (relative to this folder) and
   download its released OSNOM observations (per-box features and annotations) as described in its README.
   The official metrics are computed with Whareformer's own evaluation code (a copy is in `src/whareformer_eval/`).

## Run — closed world (benchmark ground-truth boxes; the setting reported by Whareformer and LMK)

    python graphmem2.py <out_dir> <video_ids,comma-separated> 0.4 0 100 0.1 30

Arguments: match threshold 0.4, decision window K = 100 observations, short-term gap 0.1 s, recent window 30 s.
The same settings are used for EPIC-KITCHENS, HD-EPIC (`OSNOM_DS=hd_epic`) and IT3DEgo (`OSNOM_DS=it3dego`).
The run prints mPCL and IDF1 computed with the official scorer; per-video counts come from `wf_common.rescore_run`.

## Run — open world (detector input, optional)

With SAM 3.1 detections and DINOv2-giant features in Whareformer's observation format (`GM_OBS=<dir>`):

    GM_OBS=<dir> GM_NOSCORE=1 GM_STC=0.3 GM_SEMCL=1 GM_CLOV=0.5 GM_CLOVOUT=big GM_REL=0.1 \
    GM_CHB=0.1 GM_TOPK=3 GM_SEMP=0.1 GM_LEARN=lo_0.9.pkl \
    GM_SEARCH=kind GM_KTOP=3 GM_KMARGIN=0.2 python graphmem2.py <out_dir> <video_ids> 0.4 0 10 1 30

`lo_0.9.pkl` is the learned short-term linker (trained on EPIC-KITCHENS train videos only).

## Files

- `graphmem2.py`: the tracker (memory graph, delayed decision, search).
- `graphmem_tracker.py`: shared helpers (box IoU, reservoir size).
- `wf_common.py`: data access and the official mPCL / IDF1 rescoring, wrapping Whareformer's evaluation code.
- `ow_comb.py`, `ow_fs.py`, `sp_common.py`, `sp_posegraph.py`, `sp_relembed.py`: optional modules imported only by
  non-default options.
- `lo_0.9.pkl`: learned short-term linker; `graph_train_vids.txt`, `ow_all_vids.txt`: video lists.

Results: `results/RESULTS.md` at the repository root.
