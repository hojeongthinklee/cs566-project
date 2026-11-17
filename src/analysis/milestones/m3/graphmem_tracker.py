"""GraphMem: standalone 2D-only object identity tracker (no base tracker, no 3D).

Input per frame: the OSNOM detections (boxes + DINOv2 PCA features) - the same observations WF/LMK consume,
minus the 3D locations.
- Short term: tracklets by Hungarian on (1 - IoU) + (1 - cos), gated; a tracklet ends after GAP_S unseen.
- Long term: when a tracklet reaches K observations (or ends earlier), it is matched against identities that
  are not currently visible:
    level 1 (local): identities seen within LOCAL_S seconds (or in the current chunk)  -> threshold TAU
    level 2 (older): every other identity                                               -> threshold TAU + DELTA
  score = max cosine between the tracklet mean feature and the identity's feature reservoir.
  Below threshold -> new identity. The tracklet's earlier frames are relabelled (decision delay = K frames).
Output: <out>/<vid>/tracking_outputs.pkl in the Whareformer format (3D locs are filled from the clean map
only for scoring, never used by the tracker).
usage: graphmem_tracker.py <out_dir> <vids,comma> [TAU DELTA K GAP_S LOCAL_S]
"""
import os, sys, json, pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
import wf_common as w

RES = int(os.environ.get("GM_RES", 256))
EPI_TAR = os.environ.get('GM_EPI')          # dir with <vid>.tar frames (2-fps subset) -> enables epipolar check
EPI_BETA = float(os.environ.get('GM_EPI_BETA', 0.05))
EPI_TOP = int(os.environ.get('GM_EPI_TOP', 3))


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run(args):
    vid, out_dir, TAU, DELTA, K, GAP_S, LOCAL_S = args
    fps, _ = w.video_meta(vid)
    ann = json.load(open(os.path.join(w.OBS, vid, 'object_annotations.json')))['video_annotations']
    apps = pickle.load(open(os.path.join(w.OBS, vid, '2D_feat_masked_PCA_256d.pkl'), 'rb'))
    rng = np.random.default_rng(0)
    FR = None
    if EPI_TAR and os.path.exists(os.path.join(EPI_TAR, vid + '.tar')):
        import graph_r1f as f1
        FR = f1.Frames(os.path.join(EPI_TAR, vid + '.tar'))
    epi_stats = dict(checked=0, consistent=0, inconsistent=0, changed=0)
    ids = {}            # identity -> dict(res=list of feats, last=t)
    trk = {}            # tracklet -> dict(feat=last, box, last=t, obs=[(fn, idx)], feats=[], ident=None, decided)
    out = {}
    next_t = next_id = 0

    def new_identity():
        nonlocal next_id
        ids[next_id] = dict(res=[], last=-1e9); next_id += 1
        return next_id - 1

    def decide(tid, t):
        tr = trk[tid]
        q = np.mean(tr['feats'], 0); q /= np.linalg.norm(q) + 1e-8
        visible = {x['ident'] for k, x in trk.items() if k != tid and x['ident'] is not None and t - x['last'] <= GAP_S}
        for fn, i in tr['obs']:          # identities already used in the frames this tracklet will be relabelled in
            visible |= {u for jj, u in enumerate(out[fn]['track_ids']) if jj != i and u is not None}
        best, bs = None, -1.0
        cand = []
        for iid, d in ids.items():
            if iid in visible or not d['res']:
                continue
            s = float(np.max(np.stack(d['res']) @ q))
            thr = TAU if t - d['last'] <= LOCAL_S else TAU + DELTA
            cand.append((s - thr, iid))
        if FR is not None and cand:
            # epipolar check of the top candidates: tracklet's kept view vs the identity's last kept view
            cand.sort(reverse=True)
            tv = [(fn, i) for fn, i in tr['obs'] if fn in FR.idx]
            if tv and len(cand) > 1 and cand[0][0] > -EPI_BETA:
                fq, iq = tv[-1]; bq = out[fq]['track_bboxes'][iq]
                b0 = cand[0][1]
                adj = []
                for m_, iid in cand[:EPI_TOP]:
                    lv = ids[iid].get('view')
                    if lv is None:
                        adj.append((m_, iid)); continue
                    r = f1.epi_pair(FR, lv[0], fq, {0: lv[1]}, {0: bq}) if lv[0] != fq else {}
                    el = r.get(0, (0, None, None))[2] if r else None
                    epi_stats['checked'] += 1
                    if el is not None and el < 0.5:
                        m_ += EPI_BETA; epi_stats['consistent'] += 1
                    elif el is not None and el > 1.0:
                        m_ -= EPI_BETA; epi_stats['inconsistent'] += 1
                    adj.append((m_, iid))
                cand = sorted(adj + cand[EPI_TOP:], reverse=True)
                if cand[0][1] != b0:
                    epi_stats['changed'] += 1
        for m_, iid in cand:
            if m_ >= 0 and m_ > bs:
                best, bs = iid, m_
        iid = best if best is not None else new_identity()
        tr['ident'] = iid; tr['decided'] = True
        for fn, i in tr['obs']:
            out[fn]['track_ids'][i] = iid
        absorb(tid, iid, t)

    def absorb(tid, iid, t):
        d = ids[iid]
        for v in trk[tid]['feats_new']:
            if len(d['res']) < RES:
                d['res'].append(v)
            else:
                d['res'][rng.integers(RES)] = v
        trk[tid]['feats_new'] = []
        d['last'] = max(d['last'], t)
        if FR is not None:
            kv = [(fn, i) for fn, i in trk[tid]['obs'] if fn in FR.idx]
            if kv:
                fn, i = kv[-1]; d['view'] = (fn, out[fn]['track_bboxes'][i])

    for e in sorted(ann, key=lambda e: int(e['frame_name'].split('_')[1])):
        fn = e['frame_name']; A = e.get('annotations', [])
        if fn not in apps or not A:
            continue
        t = int(fn.split('_')[1]) / fps
        F = np.asarray(apps[fn], np.float32); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
        boxes = [a['bounding_box'] for a in A]
        # close stale tracklets (decide if never decided)
        for k in [k for k, x in trk.items() if t - x['last'] > GAP_S]:
            if not trk[k]['decided']:
                decide(k, trk[k]['last'])
            del trk[k]
        act = list(trk)
        C = np.full((len(A), len(act)), 9.0)
        for i in range(len(A)):
            for j, k in enumerate(act):
                c = float(F[i] @ trk[k]['feat']); o = iou(boxes[i], trk[k]['box'])
                if c >= 0.3 or o >= 0.3:
                    C[i, j] = (1 - c) + (1 - o)
        rows, cols = linear_sum_assignment(C) if len(act) else ([], [])
        match = {r: act[c_] for r, c_ in zip(rows, cols) if C[r, c_] < 9.0}
        out[fn] = dict(track_ids=[None] * len(A), track_obj_names=[a['id'] if w.DS == 'hd_epic' else a['name'] for a in A], track_bboxes=boxes,
                       track_3d_locs=[np.zeros(3, np.float32)] * len(A))
        for i in range(len(A)):
            if i in match:
                k = match[i]
            else:
                k = next_t; next_t += 1
                trk[k] = dict(feat=F[i], box=boxes[i], last=t, obs=[], feats=[], feats_new=[], ident=None, decided=False)
            x = trk[k]
            x['feat'] = F[i]; x['box'] = boxes[i]; x['last'] = t; x['obs'].append((fn, i))
            x['feats'].append(F[i]); x['feats_new'].append(F[i])
            if x['decided']:
                out[fn]['track_ids'][i] = x['ident']
                if len(x['feats_new']) >= 10:
                    absorb(k, x['ident'], t)
            elif len(x['obs']) >= K:
                decide(k, t)
    for k in list(trk):
        if not trk[k]['decided']:
            decide(k, trk[k]['last'])
        else:
            absorb(k, trk[k]['ident'], trk[k]['last'])
    for fn, r in out.items():           # safety: unique ids per frame
        seen = set()
        for i, u in enumerate(r['track_ids']):
            if u is None or u in seen:
                r['track_ids'][i] = new_identity()
            seen.add(r['track_ids'][i])
    os.makedirs(os.path.join(out_dir, vid), exist_ok=True)
    pickle.dump(out, open(os.path.join(out_dir, vid, 'tracking_outputs.pkl'), 'wb'))
    if FR is not None:
        print('EPI', vid, epi_stats, flush=True)
    return vid, next_id


if __name__ == '__main__':
    out_dir, vids = sys.argv[1], sys.argv[2].split(',')
    TAU, DELTA, K, GAP_S, LOCAL_S = [float(x) for x in (sys.argv[3:8] + ['0.5', '0.2', '10', '0.5', '120'][len(sys.argv[3:8]):])]
    with Pool(int(os.environ.get('GM_WORKERS', 7))) as p:
        for v, n in p.imap_unordered(run, [(v, out_dir, TAU, DELTA, int(K), GAP_S, LOCAL_S) for v in vids]):
            print(v, 'identities', n, flush=True)
    per = w.rescore_run(out_dir, vids, clean_locs=True)
    A = w.aggregate(per)
    print(f'GraphMem tau={TAU} delta={DELTA} K={int(K)} gap={GAP_S} local={LOCAL_S}: mPCL {A[0]:.1f} IDF1 {A[1]:.1f}')
