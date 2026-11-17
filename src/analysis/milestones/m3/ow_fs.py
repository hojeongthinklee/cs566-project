"""Fellegi-Sunter style likelihood-ratio cue fusion with per-scene (per-video) online self-calibration.

Global prior: per-cue histograms of the cue value for same-object (M) and different-object (U) candidate pairs,
from DEV decision dumps (ow_comb rows). Score = sum_c log P(x_c|M)/P(x_c|U) + log prior odds.
Adaptation: free labels inside the current video update the histograms of ADAPT cues:
  positive = a long tracklet's first half (as pseudo-identity) vs its second half;
  negative = that tracklet vs a co-visible tracklet (cannot be the same object).
Updates are weighted by ALPHA; histograms are reset to the prior at every new video.
"""
import pickle, numpy as np
import ow_comb

ADAPT = ['app_max', 'app_top5', 'hsv_int', 'size_d', 'aspect_d', 'sol_d', 'thumb_cos', 'ghsv_int', 'kp_inl', 'hop10']
NB = 12


def fit_prior(X, y, w, feats):
    """per-cue quantile bins + smoothed class histograms"""
    P = {}
    for j, f in enumerate(feats):
        x = X[:, j]; edges = np.unique(np.quantile(x, np.linspace(0, 1, NB + 1)[1:-1]))
        b = np.searchsorted(edges, x)
        m = np.bincount(b[y == 1], weights=w[y == 1], minlength=len(edges) + 1) + 1.0
        u = np.bincount(b[y == 0], weights=w[y == 0], minlength=len(edges) + 1) + 1.0
        P[f] = dict(edges=edges, m=m, u=u)
    prior = float(np.log((w[y == 1].sum() + 1) / (w[y == 0].sum() + 1)))
    return dict(P=P, prior=prior, feats=list(feats))


class FS:
    def __init__(self, path, adapt=False, alpha=1.0, cues=None):
        z = pickle.load(open(path, 'rb')); self.base = z; self.adapt = adapt; self.alpha = alpha
        self.cues = cues or z['feats']; self.idx = {f: ow_comb.FEATS.index(f) for f in self.cues}
        self.reset()

    def reset(self):
        # prior rescaled to a pseudo-count budget per class so that in-scene evidence (alpha per pair) can move it
        N = float(__import__('os').environ.get('GM_FSN', 0))
        def sc(a): return a * (N / a.sum()) if N > 0 else a.copy()
        self.P = {f: dict(edges=v['edges'], m=sc(v['m']), u=sc(v['u'])) for f, v in self.base['P'].items()}

    def _bin(self, f, x):
        return int(np.searchsorted(self.P[f]['edges'], x))

    def score(self, row):
        s = self.base['prior']
        for f in self.cues:
            p = self.P[f]; b = self._bin(f, row[self.idx[f]])
            s += np.log(p['m'][b] / p['m'].sum()) - np.log(p['u'][b] / p['u'].sum())
        return float(s)

    def update(self, row, label):
        if not self.adapt:
            return
        for f in ADAPT:
            if f in self.P:
                b = self._bin(f, row[ow_comb.FEATS.index(f)])
                (self.P[f]['m'] if label else self.P[f]['u'])[b] += self.alpha


if __name__ == '__main__':      # build the prior from DEV dumps: ow_fs.py <dump_dir> <vids> <out.pkl>
    import sys, os
    X, y, w = [], [], []
    for v in sys.argv[2].split(','):
        p = os.path.join(sys.argv[1], v, 'cuedump.pkl')
        if os.path.exists(p):
            for r in pickle.load(open(p, 'rb')):
                for row, l in zip(r['rows'], r['lab']): X.append(row); y.append(l); w.append(r['n'])
    X, y, w = np.asarray(X), np.asarray(y), np.asarray(w, float)
    feats = [f for f in ow_comb.FEATS if ow_comb.FEATS.index(f) < X.shape[1]]
    pickle.dump(fit_prior(X[:, :len(feats)], y, w, feats), open(sys.argv[3], 'wb'))
    print('prior from', len(y), 'rows,', len(feats), 'cues ->', sys.argv[3])
