"""Spatial graph with camera pose (optional module). Fair-game rule: inputs <= LMK / Whareformer, which consume
EPIC-Fields camera poses -> using the same poses is allowed. The ADT oracle-pose test showed pose is the missing piece.

Construction (online):
  - every keyframe with a pose: each object's box centre -> undistorted normalised ray -> world ray (camera centre +
    direction) using the EPIC-Fields pose (COLMAP world->camera quaternion + translation) and the OPENCV intrinsics
  - object position = least-squares intersection of its rays (>= 2 rays with angular spread >= PO_SPREAD deg)
  - moving objects (PO_EP=1, default): an object's rays are split into rest episodes at frames where its box moves
    relative to the camera-predicted position (residual of the ray against its running estimate > PO_MT m) -> one
    position per episode; the object's node position = the episode with the most rays (where it mostly rests)
  - edges keep seen-together counts; affinity = rank fusion of counts and -3D distance (PO_W = geometry weight; 0 = covis)
Pose source: EPIC-Fields json (data/EPIC-Data/EPIC-Fields/poses[_train]/<vid>.json). The video id comes from SP_VID,
set by the drivers (sp_align_all / sp_bench) before building each structure.
"""
import os, json, collections
import numpy as np, cv2
from scipy.stats import rankdata
from sp_relembed import Struct as RelEmbed

E = os.environ.get
W_ = float(E('PO_W', 1.0)); SPREAD = float(E('PO_SPREAD', 2.0)); EP = E('PO_EP', '1') == '1'; MT = float(E('PO_MT', 0.25)); POOL = E('PO_POOL') == '1'; PT = E('PO_PT', 'center');   # POOL: moved objects wait in a pool until they settle; PT: center | bottom
PRIOR = float(E("PO_PRIOR", 0)); RANSAC = float(E("PO_RANSAC", 0));   # >0: robust triangulation, drop rays farther than this (m) from the consensus point
PRIORK = int(E("PO_PRIORK", 0))   # >0: prior from the K well-placed objects nearest in the IMAGE (local 2D arrangement)   # >0: co-visibility depth prior for objects with spread < PRIOR deg
FIELDS = E('PO_FIELDS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/EPIC-Data/EPIC-Fields'))
SX, SY = 456. / 854., 256. / 480.


def quat_R(qw, qx, qy, qz):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def ls_point(C, D):
    """least-squares intersection of rays C + s D (D unit)"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, d in zip(C, D):
        P = np.eye(3) - np.outer(d, d); A += P; b += P @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]


def spread_deg(D):
    if len(D) < 2: return 0.0
    m = np.mean(D, 0); m /= np.linalg.norm(m) + 1e-12
    return float(np.degrees(np.max(np.arccos(np.clip(np.asarray(D) @ m, -1, 1)))))


class Struct(RelEmbed):
    def __init__(self):
        super().__init__()
        self.vid = E('SP_VID'); self.P = None; self.cnt = collections.Counter(); self.allnames = set()
        self.eps = collections.defaultdict(list)            # object -> list of episodes, each [list of (C, D)]
        self._cache = None; self._dirty = 0; self.n_rays = 0; self.n_nopose = 0; self.frames = collections.defaultdict(list)
        from sp_edgepose import PoseNoise; self.noise = PoseNoise()

    def _load(self):
        for sub in ('poses', 'poses_train'):
            p = os.path.join(FIELDS, sub, f'{self.vid}.json')
            if os.path.exists(p):
                d = json.load(open(p)); f = d['camera']['params']
                self.K = np.array([[f[0], 0, f[2]], [0, f[1], f[3]], [0, 0, 1]]); self.dist = np.array(f[4:8]); self.P = d['images']; return
        self.P = {}

    def update(self, v):
        super().update(v)
        if self.P is None: self._load()
        names = sorted({o['name'] for o in v['objs']}); self.allnames |= set(names)
        for i, a in enumerate(names):
            for b in names[i + 1:]: self.cnt[(a, b)] += 1
        if v['objs'] and 'wray' in v['objs'][0]:          # ADT: world rays precomputed from GT pose + fisheye calibration (adt_rays.py)
            self.noise.next_frame()
            for o in v['objs']:
                if 'wray' in o: self._add_ray(o['name'], o['wray'][0], self.noise.apply(o['wray'][1]), v['frame'])
            self._dirty += 1; return
        q = self.P.get(v['frame'] + '.jpg')
        if q is None or not v['objs']: self.n_nopose += 1; self._dirty += 1; return
        R = quat_R(*q[:4]); t = np.array(q[4:7]); C = -R.T @ t
        px = np.float32([[((o['box'][0] + o['box'][2]) / 2) * SX, (o['box'][3] if PT == 'bottom' else (o['box'][1] + o['box'][3]) / 2) * SY] for o in v['objs']]).reshape(-1, 1, 2)
        nrm = cv2.undistortPoints(px, self.K, self.dist).reshape(-1, 2)
        for o, (x, y) in zip(v['objs'], nrm):
            d = R.T @ np.array([x, y, 1.0]); d /= np.linalg.norm(d)
            self._add_ray(o['name'], C, d, v['frame'])
        self._dirty += 1

    def _add_ray(self, name, C, d, frame):
        E_ = self.eps[name]
        if not E_: E_.append([])
        cur = E_[-1]
        pool = self.__dict__.setdefault('pool', {})
        if POOL and name in pool:                    # candidate pool: buffer rays until a few agree on one spot (settled)
            buf = pool[name]; buf.append((C, d)); buf[:] = buf[-6:]
            if len(buf) >= 3 and spread_deg([r[1] for r in buf]) >= SPREAD:
                p = ls_point([r[0] for r in buf], [r[1] for r in buf])
                if max(np.linalg.norm((np.eye(3) - np.outer(dd, dd)) @ (p - cc)) for cc, dd in buf) < MT:
                    E_.append(list(buf)); del pool[name]                        # settled somewhere: rejoin with a new episode
            self.frames[frame].append((name, C, d)); return
        if EP and len(cur) >= 3 and spread_deg([r[1] for r in cur]) >= SPREAD:
            p = ls_point([r[0] for r in cur], [r[1] for r in cur])
            res = np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - C))          # distance of the estimate from the new ray
            if res > MT:
                if POOL: pool[name] = [(C, d)]; self.frames[frame].append((name, C, d)); return   # moved: into the pool, geometry frozen
                E_.append([]); cur = E_[-1]                                        # it moved: new rest episode
        cur.append((C, d)); self.n_rays += 1; self.frames[frame].append((name, C, d))

    def position(self, n):
        best = None
        for ep in sorted(self.eps.get(n, []), key=len, reverse=True):
            if len(ep) >= 2 and spread_deg([r[1] for r in ep]) >= SPREAD:
                best = self._robust(ep) if RANSAC else ls_point([r[0] for r in ep], [r[1] for r in ep]); break
        return best

    def _robust(self, ep):
        # identity errors (tracker merges / swaps) put foreign rays into an object: sample pairs of rays, keep the point
        # with the most rays within RANSAC m, refit on those inliers
        C = np.stack([r[0] for r in ep]); D = np.stack([r[1] for r in ep]); n = len(ep)
        def resid(p): return np.linalg.norm((p - C) - ((p - C) * D).sum(1, keepdims=True) * D, axis=1)
        rng = np.random.default_rng(n); best, bi = None, -1
        for _ in range(min(40, n * (n - 1) // 2)):
            i, j = rng.choice(n, 2, replace=False)
            if float(D[i] @ D[j]) > np.cos(np.radians(SPREAD)): continue
            p = ls_point(C[[i, j]], D[[i, j]]); k = int((resid(p) < RANSAC).sum())
            if k > bi: best, bi = p, k
        if best is None: return ls_point(C, D)
        m = resid(best) < RANSAC
        return ls_point(C[m], D[m]) if m.sum() >= 2 else best

    def _prior_fill(self, pos):
        # objects whose rays barely diverge get their depth from the well-placed objects seen in the same frames:
        # along each of the object's rays, the point at the depth of the co-visible placed objects' centroid
        good = {n for n, p in pos.items() if p is not None and self._spread(n) >= PRIOR}
        for n in [n for n in pos if n not in good]:
            pts = []
            for f, rows in self.frames.items():
                mine = [(C, d) for nm, C, d in rows if nm == n]
                cand = [(nm, dd) for nm, _, dd in rows if nm in good and nm != n]
                if mine and cand:
                    C, d = mine[0]
                    if PRIORK:      # nearest in the image = smallest angle between rays from the same camera
                        cand = sorted(cand, key=lambda x: -float(x[1] @ d))[:PRIORK]
                    m = np.mean([pos[nm] for nm, _ in cand], 0); pts.append(C + max(float(d @ (m - C)), 0.05) * d)
            if pts: pos[n] = np.median(np.stack(pts), 0)
        return pos

    def _spread(self, n):
        ep = max(self.eps.get(n, [[]]), key=len); return spread_deg([r[1] for r in ep]) if len(ep) >= 2 else 0.0

    def _geo(self):
        if self._cache is not None and self._dirty == 0: return self._cache
        names = sorted(self.allnames); pos = {n: self.position(n) for n in names}
        if PRIOR: pos = self._prior_fill(pos)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        cs = np.array([self.cnt.get(p, 0) for p in pairs], float)
        gs = np.array([-(np.linalg.norm(pos[a] - pos[b]) if pos[a] is not None and pos[b] is not None else 1e6) for a, b in pairs])
        rc = rankdata(cs) / max(1, len(cs)); rg = rankdata(gs) / max(1, len(gs))
        self._cache = ({p: (1 - W_) * x + W_ * y for p, x, y in zip(pairs, rc, rg)}, pos); self._dirty = 0
        return self._cache

    def affinity(self, a, b):
        if a not in self.allnames or b not in self.allnames: return None
        return self._geo()[0].get((min(a, b), max(a, b)))

    def positions(self):
        return self._geo()[1]

    def _known(self): return set(self.allnames)

    def expected_missing(self, v):
        # with pose: a known object whose 3D estimate projects INSIDE the current camera view (in front, within the
        # frame) but is not detected is genuinely missing (likely moved), not just out of view. Score = 1 inside the view
        # (+ closeness), 0 outside. Frames without pose: no scores.
        if 'cam' in v:                 # ADT: fisheye view cone (half-angle PO_FOV deg) around the optical axis
            C, ax = v['cam']; vis = {o['name'] for o in v['objs']}; out = {}; half = np.cos(np.radians(float(E('PO_FOV', 50))))
            for n, p in self.positions().items():
                if n in vis or p is None: continue
                d = p - C; r = np.linalg.norm(d)
                out[n] = (1.0 + 1.0 / (1.0 + r)) if r > 0.05 and float(d @ ax) / r >= half else 0.0
            return out
        if self.P is None: self._load()
        q = self.P.get(v['frame'] + '.jpg')
        if q is None: return {}
        R = quat_R(*q[:4]); t = np.array(q[4:7]); vis = {o['name'] for o in v['objs']}; out = {}
        for n, p in self.positions().items():
            if n in vis or p is None: continue
            pc = R @ p + t
            if pc[2] <= 0.05: out[n] = 0.0; continue
            u, w = self.K[0, 0] * pc[0] / pc[2] + self.K[0, 2], self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2]
            inside = 0 <= u < 456 and 0 <= w < 256
            out[n] = (1.0 + 1.0 / (1.0 + pc[2])) if inside else 0.0
        return out
