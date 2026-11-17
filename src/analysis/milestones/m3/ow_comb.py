"""Round OW, Phase 1: cue-bank features for (tracklet, candidate identity) decisions, and a trained combiner.

Model-free cues come from ow_cues.py (per video pickle). Indexing: cue bank dets are in the same order as the SAM obs
annotations (both follow sam31_dets/<vid>.json), so (frame_name, det index) addresses both.
"""
import os, pickle, numpy as np, cv2
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
_KPC = {}


def kp_place_match(CB, fa, fb):
    """same physical spot? ORB matches + RANSAC homography inliers between two keyframes (cached)"""
    key = (fa, fb) if fa < fb else (fb, fa)
    if key in _KPC: return _KPC[key]
    a, b = CB.get(fa), CB.get(fb); r = (0.0, 0.0)
    if a is not None and b is not None and a['desc'] is not None and b['desc'] is not None and len(a['kp']) >= 8 and len(b['kp']) >= 8:
        ms = _BF.match(a['desc'], b['desc'])
        if len(ms) >= 8:
            p0 = np.float32([a['kp'][m.queryIdx] for m in ms]); p1 = np.float32([b['kp'][m.trainIdx] for m in ms])
            Hm, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 4.0)
            r = (float(np.log1p(int(inl.sum()) if inl is not None else 0)), 1.0)
    if len(_KPC) > 200000: _KPC.clear()
    _KPC[key] = r; return r

FEATS = ['app_max', 'app_top5', 'app_gap', 'rel', 'ord_agree', 'n_shared', 'size_d', 'aspect_d', 'sol_d', 'hsv_int',
         'prompt_sim', 'pred_d', 'pred_ok', 'thumb_cos', 'ghsv_int', 'log_gap', 'log_res', 'n_look', 'log_len', 'border',
         'score_t', 'kp_inl', 'kp_ok', 'res_div', 'app_hubz', 'rev_rank', 'neg_sim', 'neg_gap', 'cand_rank', 'hop10', 'hop30', 'hop3', 'hop5', 'hop20', 'top3', 'top10', 'neg_hop10', 'hop_gap', 'place_rel', 'mv_last']


def load_cb(cdir, vid):
    p = os.path.join(cdir, f'{vid}.pkl')
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def tracklet_agg(CB, obs):
    """aggregate per-detection cues over a tracklet's observations"""
    A = dict(n=0, la=0.0, asp=0.0, sol=0.0, hsv=np.zeros(24, np.float32), pr=np.zeros(22, np.float32), border=0.0, score=0.0)
    for fn, i in obs:
        fr = CB.get(fn) if CB else None
        if not fr or i >= len(fr['dets']):
            continue
        d = fr['dets'][i]; A['n'] += 1
        A['la'] += np.log(max(d['area'], 1e-5)); A['asp'] += np.log(max(d['aspect'], 1e-3)); A['sol'] += d['solidity']
        A['hsv'] += d['hsv']; A['pr'][d['prompt'] + 1] += 1; A['border'] += d['border']; A['score'] += d['score']
    return A


def merge_agg(dst, src):
    if dst is None:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in src.items()}
    for k, v in src.items():
        dst[k] = dst[k] + v
    return dst


def _mean(A, k):
    return A[k] / max(1, A['n'])


class HChain:
    """predict where a box seen at frame a would be at frame b (b after a) by chaining consecutive-keyframe homographies"""
    def __init__(self, CB, frames, max_steps=20):
        self.idx = {f: k for k, f in enumerate(frames)}; self.frames = frames; self.CB = CB; self.max_steps = max_steps
        self.W, self.H = 456.0, 256.0

    def predict(self, fa, fb, box):
        ia, ib = self.idx.get(fa), self.idx.get(fb)
        if ia is None or ib is None or ib < ia or ib - ia > self.max_steps:
            return None
        sx, sy = self.W / 854., self.H / 480.
        p = np.array([(box[0] + box[2]) / 2 * sx, (box[1] + box[3]) / 2 * sy, 1.0])
        for k in range(ia + 1, ib + 1):
            m = self.CB[self.frames[k]].get('motion')
            if m is None or m['H'] is None or m['inl'] < 15:
                return None
            p = m['H'] @ p; p = p / p[2]
        return p[0] / sx, p[1] / sy        # back to 854x480 coords


def pair_feats(app_all, iid, d, tr, tagg, visible, cov, offs, CB, hc, t, pool_c, n_look, qbuf=None):
    """feature vector for tracklet tr vs candidate identity iid (memory dict d)"""
    s = app_all[iid]
    others = [v for k, v in app_all.items() if k != iid]
    gap_o = s - (max(others) if others else 0.0)
    R = np.stack(d['res']) @ tr['q']
    top5 = float(np.mean(np.sort(R)[-5:]))
    nb = cov.get(iid, {})
    shared = [v for v in visible if nb.get(v, 0) >= pool_c]
    rel = len(shared) / len(visible) if visible else 0.0
    # left/right/above order agreement with shared neighbours (stored mean offset sign vs current sign)
    ag = []
    for v in shared:
        o = offs.get((iid, v))
        vb = tr['vis_boxes'].get(v)
        if o is None or vb is None:
            continue
        tb = tr['box']; dx = ((vb[0] + vb[2]) - (tb[0] + tb[2])) / 1708.; dy = ((vb[1] + vb[3]) - (tb[1] + tb[3])) / 960.
        ag.append(0.5 * (float(np.sign(dx) == np.sign(o[0])) + float(np.sign(dy) == np.sign(o[1]))))
    ord_agree = float(np.mean(ag)) if ag else 0.5
    ia = d.get('agg')
    if ia is not None and ia['n'] and tagg['n']:
        size_d = abs(_mean(tagg, 'la') - _mean(ia, 'la')); asp_d = abs(_mean(tagg, 'asp') - _mean(ia, 'asp'))
        sol_d = abs(_mean(tagg, 'sol') - _mean(ia, 'sol'))
        hsv_int = float(np.minimum(tagg['hsv'] / tagg['n'], ia['hsv'] / ia['n']).sum())
        pa, pb = tagg['pr'] / tagg['n'], ia['pr'] / ia['n']; prompt_sim = float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-8))
    else:
        size_d = asp_d = sol_d = 0.0; hsv_int = 0.5; prompt_sim = 0.5
    pred_d, pred_ok = 1.0, 0.0
    if hc is not None and d.get('last_fn') and d.get('lastbox') is not None:
        pp = hc.predict(d['last_fn'], tr['obs'][0][0], d['lastbox'])
        if pp is not None:
            fb = tr['obs_box0']
            pred_d = min(1.5, float(np.hypot((pp[0] - (fb[0] + fb[2]) / 2) / 854., (pp[1] - (fb[1] + fb[3]) / 2) / 480.))); pred_ok = 1.0
    thumb_cos, ghsv_int = 0.5, 0.5
    if CB and d.get('last_fn') in CB and tr['obs'][0][0] in CB:
        a, b = CB[d['last_fn']], CB[tr['obs'][0][0]]
        thumb_cos = float(np.asarray(a['thumb'], np.float32) @ np.asarray(b['thumb'], np.float32))
        ghsv_int = float(np.minimum(a['ghsv'], b['ghsv']).sum())
    log_gap = float(np.log1p(max(0.0, t - d['last']))) if d['last'] > -1e17 else 12.0
    kp_inl, kp_ok = (kp_place_match(CB, d['last_fn'], tr['obs'][0][0]) if (CB and d.get('last_fn')) else (0.0, 0.0))
    # hubness as features: reservoir diversity; score vs the identity's typical score to recent queries; reverse rank
    Rm = np.stack(d['res']); sub = Rm[:: max(1, len(Rm) // 16)]
    res_div = float(1.0 - (np.mean(sub @ sub.T) if len(sub) > 1 else 1.0))
    if qbuf is not None and len(qbuf) >= 5:
        hs = np.max(np.stack(qbuf) @ Rm.T, axis=1); app_hubz = float((s - hs.mean()) / (hs.std() + 1e-3))
        rev_rank = float(np.mean(hs >= s))
    else:
        app_hubz, rev_rank = 0.0, 0.5
    # distractor memory: hard negatives this identity was close to but did not take (DAM4SAM-style)
    if d.get('neg'):
        neg_sim = float(np.max(np.stack(d['neg']) @ tr['q'])); neg_gap = s - neg_sim
    else:
        neg_sim, neg_gap = 0.0, s
    cand_rank = float(sum(v > s for v in app_all.values()))
    # modern-Hopfield retrieval energy (smooth max over the reservoir) at two temperatures
    hop10 = float(np.log(np.mean(np.exp(10 * (R - R.max())))) / 10 + R.max())
    hop30 = float(np.log(np.mean(np.exp(30 * (R - R.max())))) / 30 + R.max())
    lse = lambda X, b: float(np.log(np.mean(np.exp(b * (X - X.max())))) / b + X.max())
    hop3, hop5, hop20 = lse(R, 3), lse(R, 5), lse(R, 20)
    Rs = np.sort(R); top3 = float(Rs[-3:].mean()); top10 = float(Rs[-10:].mean())
    neg_hop10 = lse(np.stack(d['neg']) @ tr['q'], 10) if d.get('neg') else 0.0
    hop_gap = hop10 - neg_hop10
    return [s, top5, gap_o, rel, ord_agree, len(shared), size_d, asp_d, sol_d, hsv_int, prompt_sim, pred_d, pred_ok,
            thumb_cos, ghsv_int, log_gap, float(np.log1p(len(d['res']))), n_look, float(np.log1p(len(tr['obs']))),
            _mean(tagg, 'border') if tagg['n'] else 0.0, _mean(tagg, 'score') if tagg['n'] else 1.0, kp_inl, kp_ok,
            res_div, app_hubz, rev_rank, neg_sim, neg_gap, cand_rank, hop10, hop30, hop3, hop5, hop20, top3, top10,
            neg_hop10, hop_gap]


class Combiner:
    """logistic (or any sklearn with predict_proba) over standardized FEATS; saved by ow_train.py"""
    def __init__(self, path):
        z = pickle.load(open(path, 'rb')); self.m, self.mu, self.sd, self.cols = z['model'], z['mu'], z['sd'], z['cols']

    def prob(self, rows):
        X = (np.asarray(rows, np.float64)[:, self.cols] - self.mu) / self.sd
        return self.m.predict_proba(X)[:, 1]
