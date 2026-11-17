"""CANDIDATE: relational distance in OBJECT-SIZE UNITS (scale-free across zoom/distance), accumulated over views,
then a global spatial embedding (classical MDS on shortest-path distances) so objects never seen together still get a
distance through chains of relations. Place = consistency of the current view's size-normalised layout with each
stored view's layout over shared objects, under two viewpoint hypotheses (same side / mirrored = seen from behind)."""
import numpy as np, collections
from scipy.sparse.csgraph import shortest_path
def unit_dist(a, b):
    sa = np.sqrt(max(1.0, (a['box'][2] - a['box'][0]) * (a['box'][3] - a['box'][1])))
    sb = np.sqrt(max(1.0, (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1])))
    ca = np.array([(a['box'][0] + a['box'][2]) / 2, (a['box'][1] + a['box'][3]) / 2]); cb = np.array([(b['box'][0] + b['box'][2]) / 2, (b['box'][1] + b['box'][3]) / 2])
    return float(np.linalg.norm(ca - cb) / np.sqrt(sa * sb))
class Struct:
    def __init__(self): self.d = collections.defaultdict(list); self.layout = {}; self._emb = None; self._dirty = True
    def update(self, v):
        import sp_common as C
        self.__dict__.setdefault('vrec', {})[v['frame']] = C.view_rec(v)
        os_ = v['objs']
        for i, a in enumerate(os_):
            for b in os_[i + 1:]:
                self.d[(min(a['name'], b['name']), max(a['name'], b['name']))].append(unit_dist(a, b))
        if len(os_) >= 2:
            self.layout[v['frame']] = {o['name']: np.array([(o['box'][0] + o['box'][2]) / 2, (o['box'][1] + o['box'][3]) / 2]) for o in os_}
        self._dirty = True
    def _embed(self):
        if not self._dirty: return
        names = sorted({n for k in self.d for n in k}); idx = {n: i for i, n in enumerate(names)}
        M = np.full((len(names), len(names)), np.inf); np.fill_diagonal(M, 0)
        for (a, b), xs in self.d.items(): M[idx[a], idx[b]] = M[idx[b], idx[a]] = np.median(xs)
        self.SP = shortest_path(M, directed=False); self.idx = idx; self._dirty = False
    def affinity(self, a, b):
        self._embed()
        if a not in self.idx or b not in self.idx: return None
        x = self.SP[self.idx[a], self.idx[b]]
        return -float(x) if np.isfinite(x) else -1e3
    def place_scores(self, v):
        import sp_common as C
        q = C.view_rec(v)
        return {f: C.soft_overlap(q, r) for f, r in self.vrec.items()}
    def _known(self):
        return {n for k in self.d for n in k}
    def expected_missing(self, v):
        """How strongly the structure expects each known, not-visible object here, given the visible ones"""
        vis = [o['name'] for o in v['objs']]; out = {}
        for n in self._known():
            if n in vis: continue
            a = [self.affinity(n, x) for x in vis]; a = [x for x in a if x is not None]
            if a: out[n] = float(np.mean(a))
        return out
