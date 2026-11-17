"""Shared helpers for design-free diagnostics on Whareformer / OSNOM (EPIC split).

- load_video(vid, loc_type): per-detection observations in time order
- rescore_run(run_dir, vids, clean_locs): per-video PCL counts (mirrors
  VideoInstance.evaluate_tracker) + per-video IDF1 counts (mirrors evaluate.py),
  optionally replacing output 3D locations with clean 'aligned' locations so that
  runs with perturbed input geometry are scored against unperturbed references.
- aggregate / bootstrap utilities.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

WF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Whareformer')
sys.path.insert(0, os.path.join(WF, 'src'))
from utils.evaluation import (get_tracking_data, get_dense_key_frames,  # noqa: E402
                              get_timescale_accuracy, get_sparse_key_frames)

DS = os.environ.get('OSNOM_DS', 'epic')   # 'epic' (default) | 'hd_epic'
_ROOT = {'epic': ('epic_osnom', 'EPIC_100_video_info.csv'), 'hd_epic': ('hd_epic_osnom', 'HD_EPIC_video_info.csv'), 'it3dego': ('it3dego_osnom', 'IT3DEgo_video_info.csv'), 'adt': ('adt_osnom', 'ADT_video_info.csv')}[DS]   # adt: standalone 3D study (src/spatial3d)
OBS = os.path.join(WF, 'data', _ROOT[0], 'observations')
OBS = os.environ.get('OSNOM_OBS', OBS)        # run on another observation root (e.g. SAM detections)
INFO = pd.read_csv(os.path.join(WF, 'data', _ROOT[0], _ROOT[1])).set_index('video_id')
DENSE = DS == 'epic'                       # Whareformer: dense key frames on EPIC, sparse on HD-EPIC


def video_meta(vid):
    fps = int(INFO.loc[vid, 'fps'])
    fps = 60 if fps == 59 else fps          # same rule as utils/config.py
    return fps, int(INFO.loc[vid, 'frames'])


def test_videos():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(WF, 'src', 'config', DS, 'whareformer_config.yaml')))
    return cfg['videos']['test']


def load_video(vid, loc_type='aligned', with_app=True):
    """Returns DataFrame rows: frame, name, det_idx, loc(3,), app(256,) ordered by frame."""
    ann = json.load(open(os.path.join(OBS, vid, 'object_annotations.json')))['video_annotations']
    locs = pickle.load(open(os.path.join(OBS, vid, f'3D_feat_{loc_type}.pkl'), 'rb'))
    apps = pickle.load(open(os.path.join(OBS, vid, '2D_feat_masked_PCA_256d.pkl'), 'rb')) if with_app else None
    rows = []
    for e in ann:
        fn = e['frame_name']
        if fn not in locs:
            continue
        f = int(fn.split('_')[1])
        for i, a in enumerate(e.get('annotations', [])):
            rows.append(dict(frame=f, frame_name=fn, name=a['name'], det_idx=i,
                             loc=locs[fn][i], app=(apps[fn][i] if with_app else None)))
    rows.sort(key=lambda r: (r['frame'], r['det_idx']))
    return rows


def _clean_loc_map(vid):
    locs = pickle.load(open(os.path.join(OBS, vid, '3D_feat_aligned.pkl'), 'rb'))
    ann = json.load(open(os.path.join(OBS, vid, 'object_annotations.json')))['video_annotations']
    m = {}
    for e in ann:
        fn = e['frame_name']
        if fn not in locs:
            continue
        for i, a in enumerate(e.get('annotations', [])):
            m.setdefault((fn, a['name']), locs[fn][i])
    return m


def pcl_counts(results, vid, min_count=3):
    """Exact mirror of VideoInstance.evaluate_tracker (dense eval, r=0.3)."""
    fps, num_frames = video_meta(vid)
    timescales = [fps * 5 * i for i in range(145)]
    obj_sets, obj_track_ids, obj_locs, track_locs, track_lifespans = get_tracking_data(results)
    key_frames = get_dense_key_frames(obj_sets, min_count=min_count) if DENSE else get_sparse_key_frames(obj_sets, fps=fps)
    correct, total = np.zeros(145), np.zeros(145)
    for k, ts in enumerate(timescales):
        acc, c, t = get_timescale_accuracy(key_frames, ts, 0.3, obj_track_ids, obj_locs, track_locs,
                                           track_lifespans, final_frame=f"frame_{num_frames:010d}",
                                           dense_eval=DENSE)
        if acc != -1:
            correct[k], total[k] = c, t
    return correct, total


def idf1_counts(results):
    import motmetrics as mm
    from importlib import util
    spec = util.spec_from_file_location('wf_eval', os.path.join(WF, 'src', 'scripts', 'evaluate.py'))
    ev = util.module_from_spec(spec); spec.loader.exec_module(ev)
    acc = mm.MOTAccumulator(auto_id=True)
    for gt_ids, pred_ids, dist in ev.process_idf1_worker(results):
        acc.update(gt_ids, pred_ids, dist)
    s = mm.metrics.create().compute(acc, metrics=['idtp', 'idfp', 'idfn'])
    return float(s['idtp'].iloc[0]), float(s['idfp'].iloc[0]), float(s['idfn'].iloc[0])


def rescore_video(run_dir, vid, clean_locs=True):
    res = pickle.load(open(os.path.join(run_dir, vid, 'tracking_outputs.pkl'), 'rb'))
    if clean_locs:
        m = _clean_loc_map(vid)
        for fn, r in res.items():
            r['track_3d_locs'] = [m.get((fn, n), l) for n, l in zip(r['track_obj_names'], r['track_3d_locs'])]
    c, t = pcl_counts(res, vid)
    idtp, idfp, idfn = idf1_counts(res)
    return dict(video=vid, correct=c, total=t, idtp=idtp, idfp=idfp, idfn=idfn)


def rescore_run(run_dir, vids=None, clean_locs=True, workers=6):
    from concurrent.futures import ProcessPoolExecutor
    vids = vids or sorted(v for v in os.listdir(run_dir) if os.path.exists(os.path.join(run_dir, v, 'tracking_outputs.pkl')))
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(rescore_video, [run_dir] * len(vids), vids, [clean_locs] * len(vids)))
    return out


def aggregate(per_video):
    C = sum(p['correct'] for p in per_video); T = sum(p['total'] for p in per_video)
    mpcl = 100 * np.mean(C / np.where(T == 0, 1, T))
    tp = sum(p['idtp'] for p in per_video); fp = sum(p['idfp'] for p in per_video); fn = sum(p['idfn'] for p in per_video)
    idf1 = 100 * 2 * tp / (2 * tp + fp + fn)
    return mpcl, idf1


def bootstrap(per_video, n=2000, seed=0, paired_with=None):
    """95% CI over videos. If paired_with is given (same video order), CI of the difference."""
    rng = np.random.default_rng(seed)
    k = len(per_video); stats = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        a = aggregate([per_video[i] for i in idx])
        if paired_with is not None:
            b = aggregate([paired_with[i] for i in idx])
            a = (a[0] - b[0], a[1] - b[1])
        stats.append(a)
    s = np.array(stats)
    return np.percentile(s, [2.5, 97.5], axis=0)  # rows: lo, hi ; cols: mpcl, idf1
