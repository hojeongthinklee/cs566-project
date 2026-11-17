"""GraphMem v2: graphmem_tracker + long-horizon memory + online nonsemantic chunks for identity search.

Same short-term tracklets and identity decision as graphmem_tracker.py (defaults reproduce it with GM_SEARCH=flat).
Additions:
- GM_PRIOR=n: the n earlier videos of the same participant are tracked first (outputs discarded); their identities
  stay in memory (they can never be right for the current video's GT, so they are pure distractors).
- Online chunks (2D, no GT): once per second, the current frame's detection features update the chunk structure
  (boundary when < THETA of them match the chunk's nodes seen in the last ACTIVE s; reopen an old chunk if >= 0.5
  of them match it). Each identity remembers the chunk it was last seen in.
- GM_SEARCH=chunk: a decision searches only identities whose chunk is the current chunk or one of the top-C chunks
  for the tracklet (by max cosine to chunk nodes); if nothing passes TAU, a global fallback at TAU + DG
  (DG < 0 disables the fallback).
Reports mPCL/IDF1 and comparisons per decision (features compared, incl. chunk nodes).
usage: graphmem2.py <out_dir> <vids,comma> [TAU DELTA K GAP_S LOCAL_S]
"""
import os, sys, json, pickle, zlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
import wf_common as w
from graphmem_tracker import iou, RES
def _R(d):     # cached stacked reservoir; invalidated by bumping d['_v'] whenever d['res'] changes
    v_ = d.get('_v', 0)
    if d.get('_Rv') != v_ or '_R' not in d: d['_R'] = np.stack(d['res']); d['_Rv'] = v_
    return d['_R']
def _bump(d): d['_v'] = d.get('_v', 0) + 1
def _C(d):     # cached centroid + radius of an identity's views: q.r <= q.m + rho for every stored view r (unit q) -> exact upper bound
    v_ = d.get('_v', 0)
    if d.get('_Cv') != v_ or '_C' not in d:
        R = _R(d); m = R.mean(0); d['_C'] = (m, float(np.sqrt(((R - m) ** 2).sum(1)).max())); d['_Cv'] = v_
    return d['_C']

SEARCH = os.environ.get('GM_SEARCH', 'flat')   # 'kind': hierarchical search -- graph parts (regions) whose object-kind sets match the frame first, flat search inside, rest only as fallback
KMARGIN = float(os.environ.get('GM_KMARGIN', -1))   # >=0: a stage-2 match is accepted only if it clears its bar by KMARGIN; otherwise the rest is searched too and the better wins
KTOP = int(os.environ.get('GM_KTOP', 3)); KJ = float(os.environ.get('GM_KJ', 0.0))   # regions kept in stage 1: top KTOP by kind overlap (and Jaccard >= KJ)
TOPC = int(os.environ.get('GM_TOPC', 3)); HCAP = int(os.environ.get('GM_HCAP', 0))   # >0: cap region-index and query features at HCAP each (cheap ranking)
DG = float(os.environ.get('GM_DG', 0.1))
THETA = float(os.environ.get('GM_THETA', 0.35))
ACTIVE = float(os.environ.get('GM_ACTIVE', 10))
MERGE = 0.5
DPREV = float(os.environ.get('GM_DPREV', 0))
CTX_S = float(os.environ.get('GM_CTX', 15))
# graph-relational candidate pool: co-visibility graph over identities; an identity whose strong
# neighbours were seen in the last POOL_W s while it was not is 'expected but missing' -> bar lowered by PB;
# identities unseen > POOL_W that are not expected get the bar raised by PN.
PB = float(os.environ.get('GM_PB', 0)); PN = float(os.environ.get('GM_PN', 0))
POOL_W = float(os.environ.get('GM_POOLW', 10)); POOL_M = int(os.environ.get('GM_POOLM', 5))
DROP = float(os.environ.get('GM_DROP', 0))   # detector-miss stress test: each detection dropped w.p. DROP (gets a one-off id)
DIAG = os.environ.get('GM_DIAG') == '1'
# relational score: bonus REL * fraction of currently visible identities that are the candidate's
# co-visibility neighbours (count >= POOL_C); lookalikes differ by what they are seen with
REL = float(os.environ.get('GM_REL', 0))
OBS_IN = os.environ.get('GM_OBS', w.OBS)   # observations to track (boxes + features); scoring always uses w.OBS GT
CLOV = float(os.environ.get('GM_CLOV', 1))   # <1: same-frame cannot-link only if box overlap (inter/min area) < CLOV
MINNEW = int(os.environ.get('GM_MINNEW', 0))   # tracklets shorter than this cannot create a memory identity
STC = float(os.environ.get('GM_STC', 0.3)); STO = float(os.environ.get('GM_STO', 0.3))   # short-term link gate (cos / IoU)
RELZ = float(os.environ.get('GM_RELZ', 0))   # weight of relative depth (m) in layout edges (0 = 2D offsets only)
RELG = float(os.environ.get('GM_RELG', 0)); RELG_SIG = float(os.environ.get('GM_RELG_SIG', 0.1))  # geometric (2D offset) relations
ABSM = float(os.environ.get('GM_ABSM', 0))   # matches with margin over threshold < ABSM do not absorb features
DECLOG = os.environ.get('GM_DECLOG'); VERIFY = os.environ.get('GM_VERIFY'); VB = float(os.environ.get('GM_VB', 0.1)); VPX = float(os.environ.get('GM_VPX', 0.5))   # E8b: log top-5 candidate pairs / same-place verification bonus from two-view geometry
FUSE = float(os.environ['GM_FUSE']) if os.environ.get('GM_FUSE') else None   # fragment fusion: a second identity that ALSO passes its bar by >= FUSE, never co-visible with the winner, is fused into it
LATE = float(os.environ.get('GM_LATE', 0))   # late re-ID: a YOUNG identity (born < LATE_S s ago) that now matches an OLDER identity (last seen before its birth, never co-visible) at score >= LATE is aliased to it (labels from then on)
LATE_S = float(os.environ.get('GM_LATE_S', 60)); LATE_N = int(os.environ.get('GM_LATE_N', 20))
MATR = float(os.environ.get('GM_MATR', 0))   # maturity re-rank: among PASSING candidates, prefer identities with more stored views (key += MATR*log(1+n)/log(1+RES)); the pass test is unchanged
FUSENB = float(os.environ.get('GM_FUSENB', -1))   # fusion also needs structural equivalence: Jaccard of the two identities' co-visibility neighbour sets (count >= POOL_C) >= FUSENB
FUSESIM = float(os.environ.get('GM_FUSESIM', -1))   # fusion also needs the two identities' reservoirs to look alike (max cross-cosine >= FUSESIM)
FUSEREL = os.environ.get('GM_FUSEREL') == '1'; FUSELAB = os.environ.get('GM_FUSEMODE') == 'label'   # label: union the output labels only, memory untouched   # offline variant: relabel past outputs of fused identities at the end (revises history)
PLACEF = os.environ.get('GM_PLACEF')   # whole-frame place features: dir with <vid>_frame.npz (names, cls, patch_mean); video only
PLMODE = os.environ.get('GM_PLMODE', 'cluster'); PLTH = float(os.environ.get('GM_PLTH', 0.7)); PLC = float(os.environ.get('GM_PLC', 0.5)); PLKEY = os.environ.get('GM_PLKEY', 'cls')
SUP = float(os.environ.get('GM_SUP', 0))   # support/containment edges (pose-free, directed): 'a on/in b' if a's box is >= 80 % inside b and b >= 3x larger
def inside(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    return ab >= 3 * aa and max(0, x2 - x1) * max(0, y2 - y1) >= 0.8 * aa
CHB = float(os.environ.get('GM_CHB', 0))   # place node bonus: candidate last seen in the current place (chunk of box-feature frames)
ANCH = float(os.environ.get('GM_ANCH', 0))   # anchor weighting of co-visibility context: weight of a visible identity = min(1, its lifespan / ANCH s)
SEMANY = os.environ.get('GM_SEMMODE') == 'any'   # prompt cannot-link only if the tracklet's prompt was NEVER seen on the identity (vs majority must match)
SEMP = float(os.environ.get('GM_SEMP', 0))   # >0: semantic cannot-link becomes a soft penalty (score -= SEMP on prompt mismatch) instead of a block
SEMCL = os.environ.get('GM_SEMCL') == '1'   # idea 7.9: semantic cannot-link (tracklet's majority SAM prompt must equal the identity's)
STLOC = float(os.environ.get('GM_STLOC', 0)); STLW = float(os.environ.get('GM_STLW', 1.0))   # short-term linking in 3D: forbid frame-to-frame links whose per-detection 3D positions are > STLOC m apart; add STLW * distance to the cost
PAIRDUMP = os.environ.get('GM_PAIRDUMP')   # learned scorer: dump short-term-link and decision candidates with features (on-policy data for a learned scorer)
LEARN = os.environ.get('GM_LEARN'); LEARN_M = pickle.load(open(LEARN, 'rb')) if LEARN else None   # learned scorer: {'link': (model, theta), 'dec': (model, theta)}
SIZE = float(os.environ.get('GM_SIZE', 0))   # >0: skip identity u if |median log box area (tracklet) - median over u's tracklets| > SIZE
CLAIM = os.environ.get('GM_CLAIM') == '1'   # co-visible undecided tracklets compete: T may not take u if a co-visible rival matches u better
TRKLOG = os.environ.get('GM_TRKLOG')   # diagnostics: dump observation -> tracklet key
LAGLOG = os.environ.get('GM_LAGLOG')   # B1 fixed-lag identity: log per observation (t_obs, provisional label from a side-effect-free peek, decision clock time)
ENROL = os.environ.get('GM_ENROL', '')   # 'oracle' = a tracklet enters memory only if it overlaps an annotated (active) object box (IoU >= 0.5) in some frame; 'hand' = only if its box touches a SAM hand box (GM_HANDS dir)
BIRTHS = float(os.environ.get('GM_BIRTHS', 0))   # min mean detection score for a tracklet to found a new identity
RELC = float(os.environ.get('GM_RELC', 0)); CT = float(os.environ.get('GM_CT', 0.1))   # contact edges (touching boxes)
def touch(a, b):
    g = CT * min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return not (a[2] + g < b[0] or b[2] + g < a[0] or a[3] + g < b[1] or b[3] + g < a[1])
CSLS = float(os.environ.get('GM_CSLS', 0))   # hubness (CSLS-style) correction weight
RELSOFT = float(os.environ.get('GM_RELSOFT', 0))
RELMODE = os.environ.get('GM_RELMODE', 'frac'); RELCTX = float(os.environ.get('GM_RELCTX', 0))
PROV = float(os.environ.get('GM_PROV', 0))   # extra bar for provisional (not yet re-identified) identities
MG = float(os.environ.get('GM_MG', 0)); MD = float(os.environ.get('GM_MD', 0.3))   # short-gap motion gate
CUES_DIR = os.environ.get('GM_CUES'); CUEDUMP = os.environ.get('GM_CUEDUMP') == '1'
COMB_MAP = json.load(open(os.environ['GM_COMB_MAP'])) if os.environ.get('GM_COMB_MAP') else None   # {vid: combiner} cross-fitting
COMB_PATH = os.environ.get('GM_COMB') or (next(iter(COMB_MAP.values())) if COMB_MAP else None); COMBTH = float(os.environ.get('GM_COMBTH', 0.5)); COMB_ADAPT = os.environ.get('GM_COMBADAPT') == '1'; COMB_LAZY2 = os.environ.get('GM_COMBLAZY2') == '1'; COMB_DEFER = float(os.environ.get('GM_COMBDEFER', 0)); COMB_DPREV = float(os.environ.get('GM_COMBDPREV', 0.2)); COMB_LR = float(os.environ.get('GM_COMBLR', 0.05)); COMB_N = int(os.environ.get('GM_COMBN', 8))
FS_PATH = os.environ.get('GM_FS'); FS_ADAPT = os.environ.get('GM_FSADAPT') == '1'
FSTH = float(os.environ.get('GM_FSTH', 0.0)); FSA = float(os.environ.get('GM_FSA', 1.0))
OWC = bool(CUEDUMP or COMB_PATH or FS_PATH)
if FS_PATH:
    import ow_fs
if OWC or os.environ.get('GM_GATE') == '1':
    import ow_comb
GATE = os.environ.get('GM_GATE') == '1'; GATE_A = float(os.environ.get('GM_GATE_A', 0.002)); GATE_B = float(os.environ.get('GM_GATE_B', 30))
CB_REF = [None]; GATE_BORDER = os.environ.get('GM_GATE_BORDER', '1') == '1'
MIX = os.environ.get('GM_MIX') == '1'; MIX_M = int(os.environ.get('GM_MIX_M', 8)); MIX_R = float(os.environ.get('GM_MIX_R', 0.25))
MIX_L = float(os.environ.get('GM_MIX_L', 2.0)); MIX_E = float(os.environ.get('GM_MIX_E', 0.05))
LOCW = float(os.environ.get('GM_LOC', 0)); LOCS = float(os.environ.get('GM_LOCS', 0.3)); LOCARM = os.environ.get('GM_LOCARM', 'aligned')   # E9 location continuity: bonus LOCW*exp(-d^2/2 LOCS^2) for reappearing near the identity's last known 3D position (per-detection 3D = WF/LMK input)
RAYGATE = float(os.environ.get('GM_RAYGATE', 0)); RAYD = float(os.environ.get('GM_RAYD', 0.5)); RAYRESET = os.environ.get('GM_RAYRESET') == '1'; RAYBONUS = float(os.environ.get('GM_RAYBONUS', 0))   # spatial-graph gate: penalty when a tracklet's rays miss a candidate's 3D estimate by > RAYD m
CORESET = os.environ.get('GM_CORESET') == '1'; FOCUS = float(os.environ.get('GM_FOCUS', 0)); PLACE = float(os.environ.get('GM_PLACE', 0)); MVB = float(os.environ.get('GM_MVB', 0)); MVT = float(os.environ.get('GM_MVT', 0.03)); PLACEFEAT = os.environ.get('GM_PLACEFEAT') == '1'; MVFEAT = os.environ.get('GM_MVFEAT') == '1'; PLACE_A = float(os.environ.get('GM_PLACEA', 0.3)); FOCUS_PEN = float(os.environ.get('GM_FOCUSPEN', -0.5)); FOCUS_C = int(os.environ.get('GM_FOCUSC', 3)); RECIP = int(os.environ.get('GM_RECIP', 0))
TOPK = int(os.environ.get('GM_TOPK', 1))   # identity score = mean of top-k reservoir similarities (1 = max)
# ambiguity (Lowe-style margin): if best and runner-up appearance scores differ by < MARGIN, the match is unsafe.
# AMB=new -> start a new identity; AMB=defer -> wait for more observations (re-check every 25, up to DEFER_X * K)
MARGIN = float(os.environ.get('GM_MARGIN', 0)); AMB = os.environ.get('GM_AMB', 'new')
DEFER_X = float(os.environ.get('GM_DEFERX', 3))
POOL_F = float(os.environ.get('GM_POOLF', 0.5)); POOL_C = int(os.environ.get('GM_POOLC', 3))


def track(vid, S, P, record):
    TAU, DELTA, K, GAP_S, LOCAL_S = P
    _exotic = (MIX or RAYGATE or RAYBONUS or LOCW or MVB or PLACE or FOCUS or RELC or SUP or RELG or VERIFY or CSLS or RECIP or CLAIM or PB or PN or DECLOG or COMB_PATH or FS_PATH or OWC or SIZE or (LEARN_M is not None and 'dec' in LEARN_M))
    FASTSKIP = os.environ.get('GM_FASTSKIP', '1') == '1' and not _exotic
    BMAX = max(REL, 0.0) + max(CHB, 0.0) + max(MATR, 0.0)
    BNB = os.environ.get('GM_BNB') == '1' and FASTSKIP and SEARCH == 'flat'   # exact pruning only in the plain flat search with bounded cues   # largest total bonus a candidate can receive in the fast path
    fps, _ = w.video_meta(vid)
    ann = json.load(open(os.path.join(OBS_IN, vid, 'object_annotations.json')))['video_annotations']
    apps = pickle.load(open(os.path.join(OBS_IN, vid, '2D_feat_masked_PCA_256d.pkl'), 'rb'))
    rng = np.random.default_rng(0)
    ids, ch = S['ids'], S['chunks']
    toff = S['toff']
    trk, out = {}, {}
    next_t = 0
    st = S['stats']

    def new_identity():
        iid = S['next_id']; S['next_id'] += 1
        ids[iid] = dict(res=[], last=-1e18, chunk=S['cur'], born=S.get('tnow'))
        return iid

    def fuse(c, b):      # identity c is a fragment of b: b takes c's views, prompts and co-visibility; c leaves the search
        dc, db = ids[c], ids[b]; st['fused'] = st.get('fused', 0) + 1
        if FUSELAB:
            al = S.setdefault('alias', {}); r_ = b
            while r_ in al: r_ = al[r_]
            if r_ != c: al[c] = r_
            return
        for v in dc['res']:
            if len(db['res']) < RES: db['res'].append(v)
            else: db['res'][int(np.random.default_rng(len(db['res']) + c).integers(RES))] = v
            _bump(db)
        db['last'] = max(db['last'], dc['last'])
        if dc.get('pr'): db.setdefault('pr', __import__('collections').Counter()).update(dc['pr'])
        if dc.get('pc'):
            ip_ = db.setdefault('pc', {})
            for k_, v_ in dc['pc'].items(): ip_[k_] = ip_.get(k_, 0) + v_
        cv = S.setdefault('cov', {}); sn = S.setdefault('seen', {})
        for n_, k_ in cv.pop(c, {}).items():
            if n_ == b: continue
            cv.setdefault(b, {})[n_] = cv.get(b, {}).get(n_, 0) + k_
            if c in cv.get(n_, {}): cv[n_][b] = cv[n_].get(b, 0) + cv[n_].pop(c)
        if c in sn: sn[b] = max(sn.get(b, -1e18), sn.pop(c))
        dc['res'] = []; _bump(dc); dc['oneoff'] = True; S.setdefault('alias', {})[c] = b
        for x_ in trk.values():
            if x_['ident'] == c: x_['ident'] = b

    def chunk_update(F, t):
        cur = ch[S['cur']] if S['cur'] is not None else None
        def frac(c, Fx):
            if not c['f']:
                return 0.0, {}
            Cm = np.stack(c['f']); Sm = Fx @ Cm.T; a, b = Sm.argmax(1), Sm.argmax(0)
            m = {i: int(a[i]) for i in range(len(a)) if b[a[i]] == i and Sm[i, a[i]] >= MERGE}
            return len(m) / len(Fx), m
        matched = None
        if cur is not None:
            act = [j for j, ls in enumerate(cur['last']) if t - ls <= ACTIVE]
            if act:
                sub = dict(f=[cur['f'][j] for j in act]); fr, m = frac(sub, F)
                if fr >= THETA:
                    matched = {i: act[j] for i, j in m.items()}
        if matched is None:
            best = None
            for cid, c in enumerate(ch):
                fr, m = frac(c, F); st['cmp'] += len(c['f']) * len(F)
                if fr >= 0.5 and (best is None or fr > best[0]):
                    best = (fr, cid, m)
            if best:
                S['cur'] = best[1]; matched = best[2]
            else:
                ch.append(dict(f=[], n=[], last=[])); S['cur'] = len(ch) - 1; matched = {}
        c = ch[S['cur']]
        for i in range(len(F)):
            if i in matched:
                j = matched[i]; n = c['n'][j]; v = c['f'][j] * n + F[i]
                c['f'][j] = v / (np.linalg.norm(v) + 1e-8); c['n'][j] = n + 1; c['last'][j] = t
            else:
                c['f'].append(F[i].copy()); c['n'].append(1); c['last'].append(t)

    def _la(x):
        return float(np.median([np.log(max(1.0, (b[2] - b[0]) * (b[3] - b[1]))) for b in (out[fn]['track_bboxes'][i] for fn, i in x['obs'])]))

    def decide(tid, t, force=True):
        tr = trk[tid]
        q = np.mean(tr['feats'], 0); q /= np.linalg.norm(q) + 1e-8
        if CLOV < 1:   # cannot-link only between NON-overlapping boxes (overlapping = possibly the same object, e.g. part/whole)
            def ovl(a, b):
                x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                return inter / (min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) + 1e-9)
            visible = {x['ident'] for k, x in trk.items() if k != tid and x['ident'] is not None and t - x['last'] <= GAP_S
                       and ovl(x['box'], tr['box']) < CLOV}
            for fn, i in tr['obs']:
                bi = out[fn]['track_bboxes'][i]
                visible |= {u for jj, u in enumerate(out[fn]['track_ids'])
                            if jj != i and u is not None and ovl(out[fn]['track_bboxes'][jj], bi) < CLOV}
        else:
            visible = {x['ident'] for k, x in trk.items() if k != tid and x['ident'] is not None and t - x['last'] <= GAP_S}
            for fn, i in tr['obs']:
                visible |= {u for jj, u in enumerate(out[fn]['track_ids']) if jj != i and u is not None}
        st['dec'] += 1
        la_T = _la(tr) if SIZE else None
        rivals = []
        if CLAIM:
            fr_ = {fn for fn, _ in tr['obs']}
            for k_, x_ in trk.items():
                if k_ != tid and not x_['decided'] and x_['obs'] and any(fn in fr_ for fn, _ in x_['obs']):
                    q_ = np.mean(x_['feats'], 0); rivals.append(q_ / (np.linalg.norm(q_) + 1e-8))
            rivals = np.stack(rivals) if rivals else None
        cov, seen = S.setdefault('cov', {}), S.setdefault('seen', {})

        def pool_adj(iid):
            if not (PB or PN) or t - seen.get(iid, -1e18) <= POOL_W:
                return 0.0
            nb = sorted(((c, n) for n, c in cov.get(iid, {}).items() if c >= POOL_C), reverse=True)[:POOL_M]
            if nb and sum(t - seen.get(n, -1e18) <= POOL_W for _, n in nb) >= POOL_F * len(nb):
                st['pool'] = st.get('pool', 0) + 1
                return -PB
            return PN

        t0_tr = int(tr['obs'][0][0].split('_')[1]) / fps + toff
        tb = tr['box']; tcx, tcy, tcz = (tb[0] + tb[2]) / 1708., (tb[1] + tb[3]) / 960., RELZ * (tr.get('z') or 0.0)
        ctxset = set(visible)
        place_r = None
        if (PLACE or PLACEFEAT) and S.get('vr'):
            seeds = sorted({S['vix'][int(int(fn_.split('_')[1]) / fps + toff)] for fn_, _ in tr['obs'] if int(int(fn_.split('_')[1]) / fps + toff) in S['vix']})
            if seeds:
                import scipy.sparse as sps
                n_ = len(S['vr']); E_ = np.array(S['ve']) if S['ve'] else np.zeros((0, 3))
                W_ = sps.coo_matrix((np.r_[E_[:, 2], E_[:, 2]], (np.r_[E_[:, 0], E_[:, 1]].astype(int), np.r_[E_[:, 1], E_[:, 0]].astype(int))), shape=(n_, n_)).tocsr()
                dg = np.asarray(W_.sum(1)).ravel(); dg[dg == 0] = 1.0; P_ = sps.diags(1.0 / dg) @ W_
                e_ = np.zeros(n_); e_[seeds] = 1.0 / len(seeds); r_ = e_.copy()
                for _ in range(20): r_ = PLACE_A * e_ + (1 - PLACE_A) * (P_.T @ r_)
                r_[seeds] = 0.0; mx = r_.max()
                place_r = r_ / mx if mx > 0 else None
        focus_d = None
        if FOCUS and visible:      # weighted hop distances from the lit region over the co-visibility graph
            import heapq
            cov_ = S.get('cov', {}); focus_d = {u: 0.0 for u in visible}; hq = [(0.0, u) for u in visible]
            while hq:
                d0, u = heapq.heappop(hq)
                if d0 > focus_d.get(u, 1e9) or d0 > 6: continue
                for x, c in cov_.get(u, {}).items():
                    if c < FOCUS_C: continue
                    nd = d0 + 1.0 / np.log1p(c)
                    if nd < focus_d.get(x, 1e9): focus_d[x] = nd; heapq.heappush(hq, (nd, x))
        if RELSOFT:     # undecided co-visible tracklets contribute their best-guess identity (if within RELSOFT of tau)
            for k, x in trk.items():
                if k == tid or x['ident'] is not None or t - x['last'] > GAP_S or not x['feats']:
                    continue
                qk = np.mean(x['feats'], 0); qk /= np.linalg.norm(qk) + 1e-8; bu, bsc = None, -1
                for u, d_ in ids.items():
                    if u in visible or not d_['res'] or d_.get('oneoff'): continue
                    sc_ = float(np.max(_R(d_) @ qk)); st['cmp'] += len(d_['res'])
                    if sc_ > bsc: bu, bsc = u, sc_
                if bu is not None and bsc >= TAU - RELSOFT:
                    ctxset.add(bu)
        if RELCTX > 0:     # context window: identities seen within RELCTX s (not only those visible right now)
            ctxset |= {u for u, ts_ in S.get('seen', {}).items() if t - ts_ <= RELCTX and u in ids}
        vis_ctr = {}
        tr_touch = set()     # identities whose current box touches the tracklet's last box
        if RELC:
            for k, x in trk.items():
                if k != tid and x['ident'] is not None and t - x['last'] <= GAP_S and touch(x['box'], tr['box']):
                    tr_touch.add(x['ident'])
        tr_in, tr_has = set(), set()      # visible identities the tracklet sits on/in, and ones sitting on/in it
        if SUP:
            for k, x in trk.items():
                if k != tid and x['ident'] is not None and t - x['last'] <= GAP_S:
                    if inside(tr['box'], x['box']): tr_in.add(x['ident'])
                    elif inside(x['box'], tr['box']): tr_has.add(x['ident'])
        if RELG:
            for k, x in trk.items():
                if k != tid and x['ident'] is not None and t - x['last'] <= GAP_S:
                    vis_ctr[x['ident']] = ((x['box'][0] + x['box'][2]) / 1708., (x['box'][1] + x['box'][3]) / 960., RELZ * (x.get('z') or 0.0))

        _dl = []
        tr_pr = None
        if SEMCL:
            import collections as _co
            _pc = _co.Counter(x_ for x_ in (PR.get(fn, [None] * 999)[i] for fn, i in tr['obs']) if x_ is not None); tr_pr = _pc.most_common(1)[0][0] if _pc else None   # no prompt -> no constraint
        _pass = []
        def search(cands, extra):
            best, bs = None, -1.0
            order, ubs = cands, None
            if BNB:      # branch-and-bound: visit identities by an exact upper bound on their appearance score, stop when none can win
                cl = [u for u in cands if u not in visible and ids[u]['res'] and not ids[u].get('oneoff')]
                if cl:
                    cm_ = [_C(ids[u]) for u in cl]; ub_ = np.stack([c_[0] for c_ in cm_]) @ q + np.array([c_[1] for c_ in cm_])
                    o_ = np.argsort(-ub_, kind='stable'); order = [cl[i] for i in o_]; ubs = ub_[o_]; st['cmp'] += len(cl)
                else:
                    order = []
                thr0 = min(TAU, TAU + DELTA) + extra
            for k_i, iid in enumerate(order):
                if ubs is not None and ubs[k_i] + BMAX - thr0 < max(bs, 0.0) - 1e-9:
                    st['bnb_stop'] = st.get('bnb_stop', 0) + 1; break      # every remaining identity is provably unable to win
                d = ids[iid]
                if iid in visible or not d['res'] or d.get('oneoff'):
                    continue
                st['cmp'] += len(d['res'])
                if MIX and d.get('mix'):      # identity-adaptive bar: distance relative to the nearest cluster's own spread
                    cs_ = np.array([float(c[0] @ q) for c in d['mix']]); b_ = int(np.argmax(cs_)); c = d['mix'][b_]
                    spread = c[2] / max(1, c[1] - 1) if c[1] > 1 else MIX_R
                    z = (1 - cs_[b_]) / (spread + MIX_E)
                    st['cmp'] += len(d['mix'])
                    if z <= MIX_L and cs_[b_] >= TAU - 0.2 and (MIX_L - z) > bs: best, bs = iid, MIX_L - z
                    continue
                _semx = bool(SEMCL and d.get('pr') and tr_pr is not None and ((d['pr'].get(tr_pr, 0) == 0) if SEMANY else (d['pr'].most_common(1)[0][0] != tr_pr)))
                if _semx and not SEMP: continue
                R_ = _R(d); sims = R_ @ q
                if FASTSKIP and float(sims.max()) + BMAX < min(TAU, TAU + DELTA) + extra - 1e-9:
                    continue      # exact early exit: even the largest possible bonus cannot lift this candidate over its bar
                if SIZE and d.get('la') and abs(la_T - float(np.median(d['la']))) > SIZE: continue
                if CLAIM and rivals is not None:
                    st['cmp'] += len(rivals) * len(R_)
                    if float(np.max(rivals @ R_.T)) > float(np.max(sims)): continue     # a co-visible rival claims u
                others_q = [qq for qq, ii in S.get('rq', []) if ii != iid]
                if RECIP and len(others_q) >= 5:     # reciprocal gate vs queries that went to OTHER identities
                    hs = np.max(np.stack(others_q) @ R_.T, axis=1); st['cmp'] += len(others_q) * len(R_)
                    if int(np.sum(hs > float(np.max(sims)))) >= RECIP: continue
                if CSLS and len(S.get('qbuf', [])) >= 5:     # hubness correction: typical similarity of c to recent queries
                    Qb = np.stack(S['qbuf']); hub = float(np.mean(np.max(Qb @ R_.T, axis=1))); st['cmp'] += len(Qb) * len(R_)
                else:
                    hub = None
                s = float(np.max(sims)) if TOPK <= 1 or len(sims) < TOPK else float(np.mean(np.partition(sims, -TOPK)[-TOPK:]))
                if hub is not None:
                    s -= CSLS * hub
                if _semx: s -= SEMP
                if (RAYGATE or RAYBONUS) and RAY:
                    rs = S.get('rays', {}).get(iid)
                    if rs is not None and rs['n'] >= 3 and rs['mc'] <= np.cos(np.radians(3.0)):
                        p_ = np.linalg.solve(rs['A'] + 1e-9 * np.eye(3), rs['b']); md = []
                        for fn_, i_ in tr['obs'][:8]:
                            r_ = ray_of(fn_, out[fn_]['track_bboxes'][i_])
                            if r_ is not None:
                                C_, d_ = r_; v_ = p_ - C_; md.append(np.linalg.norm(v_ - (v_ @ d_) * d_))
                        if md and np.median(md) > RAYD: s -= RAYGATE          # the tracklet is somewhere else in the room
                        elif md and np.median(md) < RAYD / 2: s += RAYBONUS    # its rays pass through the candidate's position
                if LOCW and L3 is not None and iid in S.get('loc', {}):
                    fn_, i_ = tr['obs'][0]; p_ = L3.get(fn_)
                    if p_ is not None and i_ < len(p_):
                        d_ = float(np.linalg.norm(np.asarray(p_[i_], float).ravel()[:3] - S['loc'][iid])); s += LOCW * np.exp(-d_ * d_ / (2 * LOCS * LOCS))
                if MVB and S.get('mv', {}).get(iid, 0.0) > MVT:
                    s += MVB          # an identity last seen being carried is expected to reappear elsewhere
                if PLACE and place_r is not None:
                    vv = S['uv'].get(iid, [])[-10:]
                    s += PLACE * ((max(place_r[j] for j in vv) if vv else 0.0) - 0.5)     # last seen in a place this view reaches
                if FOCUS and focus_d is not None:
                    dd = focus_d.get(iid)
                    s += FOCUS * (1.0 / (1.0 + dd) if dd is not None else FOCUS_PEN)     # near the lit region = plausible
                if REL and ctxset:
                    nb = cov.get(iid, {}); hit = sum(nb.get(v_, 0) >= POOL_C for v_ in ctxset)
                    if RELMODE == 'jac':      # Jaccard: don't reward identities that simply have many neighbours
                        nN = sum(c_ >= POOL_C for c_ in nb.values())
                        s += REL * hit / max(1, len(ctxset) + nN - hit)
                    elif ANCH:
                        wts = {v_: min(1.0, max(0.0, ids[v_]['last'] - ids[v_].get('first', ids[v_]['last'])) / ANCH) if v_ in ids else 0.0 for v_ in ctxset}
                        sw = sum(wts.values())
                        if sw > 0: s += REL * sum(w_ for v_, w_ in wts.items() if nb.get(v_, 0) >= POOL_C) / sw
                    else:
                        s += REL * hit / len(ctxset)
                if SUP and (tr_in or tr_has):
                    sp_ = S.get('sup', {}); a_ = sp_.get(iid, {})
                    hits = sum(a_.get(v_, 0) >= POOL_C for v_ in tr_in) + sum(sp_.get(v_, {}).get(iid, 0) >= POOL_C for v_ in tr_has)
                    s += SUP * hits / (len(tr_in) + len(tr_has))
                if CHB and PF is not None:
                    if PLMODE == 'cluster':
                        if d.get('pl') is not None and d['pl'] == S.get('pcur'): s += CHB
                    elif d.get('lf') is not None and S.get('pf_now') is not None:
                        s += CHB * (float(d['lf'] @ S['pf_now']) - PLC)
                elif CHB and d.get('chunk') is not None and d['chunk'] == S.get('cur'):
                    s += CHB
                if RELC and tr_touch:
                    cn = S.get('con', {}).get(iid, {})
                    s += RELC * (sum(cn.get(v_, 0) >= POOL_C for v_ in tr_touch) / len(tr_touch) - 0.5)
                if RELG and vis_ctr:
                    offs = S.get('off', {}); nb = cov.get(iid, {}); sims = []
                    for v_, (vx, vy, vz) in vis_ctr.items():
                        o_ = offs.get((iid, v_))
                        if o_ is not None and nb.get(v_, 0) >= POOL_C:
                            d2 = (vx - tcx - o_[0]) ** 2 + (vy - tcy - o_[1]) ** 2 + (vz - tcz - o_[2]) ** 2
                            sims.append(np.exp(-d2 / (2 * RELG_SIG ** 2)))
                    if sims:
                        s += RELG * (np.mean(sims) - 0.5)     # centred: consistent layout +, inconsistent -
                if VERIFY and VD is not None and d.get('last_fn'):
                    v_ = VD.get((tr['obs'][0][0], d['last_fn']))
                    if v_ is not None and v_ <= VPX: s += VB          # the candidate's stored spot reprojects onto this tracklet
                if DECLOG and d.get('last_fn'):
                    _dl.append((s, tr['obs'][0][0], [float(x) for x in out[tr['obs'][0][0]]['track_bboxes'][tr['obs'][0][1]]], d['last_fn'], [float(x) for x in d['lastbox']], iid))
                thr = (TAU if t - d['last'] <= LOCAL_S else TAU + DELTA) + extra
                thr += pool_adj(iid)
                if MG and d.get('lastbox') is not None and 0 <= t0_tr - d['last'] < MG:
                    lb = d['lastbox']; fb = tr['obs_box0']
                    dd = (((lb[0] + lb[2]) - (fb[0] + fb[2])) / 1708.) ** 2 + (((lb[1] + lb[3]) - (fb[1] + fb[3])) / 960.) ** 2
                    if dd ** 0.5 > MD:
                        continue                      # seen moments ago far away: implausible motion
                if PROV and not d.get('conf', True):
                    thr += PROV                   # provisional identity (seen by one tracklet only): stricter bar
                if d['last'] < -1e17:
                    thr += DPREV       # identity from an earlier session (prior / distractor memory)
                if s >= thr: _pass.append((s - thr, iid))
                key_ = s - thr + (MATR * np.log1p(len(d['res'])) / np.log1p(RES) if MATR else 0.0)
                if s >= thr and key_ > bs:
                    best, bs = iid, key_
            st['last_margin'] = bs
            return best
        if SEARCH == 'ctx' and ch:
            # recognise the current local configuration first: recent view context (features seen in the last CTX_S s)
            ctx = [f_ for tt, f_ in S['ctx'] if t - tt <= CTX_S]
            if HCAP and len(ctx) > HCAP: ctx = ctx[::int(np.ceil(len(ctx) / HCAP))]      # capped region query: a few recent views
            Q = np.stack(ctx) if ctx else q[None]
            sc = []
            for cid, c in enumerate(ch):
                if c['f']:
                    if HCAP:      # region index = a strided sample of its features, cached by length
                        if c.get('_pn') != len(c['f']):
                            fs_ = c['f'][::int(np.ceil(len(c['f']) / HCAP))] if len(c['f']) > HCAP else c['f']; c['_P'] = np.stack(fs_); c['_pn'] = len(c['f'])
                        Cm = c['_P']
                    else:
                        Cm = np.stack(c['f'])
                    st['cmp'] += len(Cm) * len(Q)
                    mx = (Q @ Cm.T).max(1); sc.append((float(mx[mx >= 0.2].sum()), cid))
            top = {cid for _, cid in sorted(sc, reverse=True)[:TOPC]} | {S['cur']}
            local = [iid for iid, d in ids.items() if d['chunk'] in top or t - d['last'] <= LOCAL_S]
            best = search(local, 0.0)
            if best is None and DG >= 0:
                _ls = set(local); best = search([iid for iid in ids if iid not in _ls], DG)
        elif SEARCH == 'kind' and S.get('rmem'):
            fn_q = tr['obs'][-1][0]; fk = {x_ for x_ in PR.get(fn_q, []) if x_ is not None}      # kinds in the current frame
            RK = S.get('rkind', {}); sc_ = []
            for cid, kc in RK.items():
                rk = {k_ for k_, n_ in kc.items() if n_ > 0}
                if rk: j_ = len(fk & rk) / len(fk | rk) if (fk | rk) else 0.0; sc_.append((j_, cid))
            sc_.sort(reverse=True); top = {cid for j_, cid in sc_[:KTOP] if j_ >= KJ} | {S['cur']}
            st['cmp'] += len(sc_)
            local = set().union(*[S['rmem'].get(c_, set()) for c_ in top]) | {u for u, ts_ in S.get('seen', {}).items() if t - ts_ <= LOCAL_S}
            best = search([u for u in local if u in ids], 0.0); m1_ = st.get('last_margin', -1.0)
            if best is None or (KMARGIN >= 0 and m1_ < KMARGIN):
                b2_ = search([u for u in ids if u not in local], 0.0); m2_ = st.get('last_margin', -1.0)       # fallback: the rest of the graph, no penalty
                if b2_ is not None and (best is None or m2_ > m1_): best = b2_
        elif SEARCH == 'chunk' and ch:
            sc = []
            for cid, c in enumerate(ch):
                if c['f']:
                    st['cmp'] += len(c['f']); sc.append((float(np.max(np.stack(c['f']) @ q)), cid))
            top = {cid for _, cid in sorted(sc, reverse=True)[:TOPC]} | {S['cur']}
            local = [iid for iid, d in ids.items() if d['chunk'] in top]
            best = search(local, 0.0)
            if best is None and DG >= 0:
                best = search([iid for iid, d in ids.items() if d['chunk'] not in top], DG)
        elif os.environ.get('GM_LAZY') == '1':
            # hierarchical, lazy: current-session identities first; earlier sessions only if nothing passes
            cur_s = [iid for iid, d in ids.items() if d['last'] > -1e17]
            best = search(cur_s, 0.0)
            if best is None:
                best = search([iid for iid, d in ids.items() if d['last'] <= -1e17], 0.0)
        else:
            best = search(list(ids), 0.0)
        if OWC:
            cands = [u for u, d_ in ids.items() if u not in visible and d_['res'] and not d_.get('oneoff')]
            if os.environ.get('GM_LAZY') == '1' and not COMB_LAZY2:
                cur_ = [u for u in cands if ids[u]['last'] > -1e17]; cands = cur_ if cur_ else cands
            app_all = {u: float(np.max(_R(ids[u]) @ q)) for u in cands}
            top = sorted(cands, key=lambda u: -app_all[u])[:COMB_N]
            tagg = ow_comb.tracklet_agg(CB, tr['obs']); tr['q'] = q
            tr['vis_boxes'] = {x['ident']: x['box'] for k, x in trk.items() if k != tid and x['ident'] is not None and t - x['last'] <= GAP_S}
            n_look = sum(v >= TAU - 0.1 for v in app_all.values())
            rows = [ow_comb.pair_feats(app_all, u, ids[u], tr, tagg, visible, S.get('cov', {}), S.get('off', {}), CB, HC, t, POOL_C, n_look,
                                       S.get('owq'))
                    for u in top]
            if PLACEFEAT:      # place relevance of the views where each candidate was seen (view-graph diffusion)
                def _pr(u):
                    vv = S.get('uv', {}).get(u, [])[-10:]
                    return float(max(place_r[j] for j in vv)) if (vv and place_r is not None) else -1.0
                rows = [r_ + [_pr(u)] for r_, u in zip(rows, top)]
                if MVFEAT: rows = [r_ + [S.get('mv', {}).get(u, -1.0)] for r_, u in zip(rows, top)]    # last-seen carried motion
            qb_ = S.setdefault('owq', []); qb_.append(q)
            if len(qb_) > 50: qb_.pop(0)
            if CUEDUMP and record and rows:
                gl_ = [G[fn][i] for fn, i in tr['obs']]; tg_ = max(set(gl_), key=gl_.count)
                lab = [int(bool(igt.get(u)) and max(set(igt[u]), key=igt[u].count) == tg_) for u in top]
                S.setdefault('cuedump', []).append(dict(rows=rows, lab=lab, n=len(gl_), tg=str(tg_)))
            def selfsup_rows():      # free labels from this scene: halves of the same tracklet (+), co-visible tracklet (-)
                out_rows = []
                if len(tr['obs']) < 4: return out_rows
                h = len(tr['obs']) // 2
                def pseudo(obs, feats_):
                    fn_l, i_l = obs[-1]
                    return dict(res=list(feats_), last=t - 1.0, last_fn=fn_l, lastbox=out[fn_l]['track_bboxes'][i_l],
                                agg=ow_comb.tracklet_agg(CB, obs))
                q2 = np.mean(tr['feats'][h:], 0); q2 /= np.linalg.norm(q2) + 1e-8
                fn0, i0 = tr['obs'][h]
                t2 = dict(q=q2, obs=tr['obs'][h:], obs_box0=out[fn0]['track_bboxes'][i0], box=tr['box'], vis_boxes={})
                a2 = ow_comb.tracklet_agg(CB, tr['obs'][h:])
                pos = pseudo(tr['obs'][:h], tr['feats'][:h]); sp = float(np.max(np.stack(pos['res']) @ q2))
                out_rows.append((ow_comb.pair_feats({-1: sp}, -1, pos, t2, a2, set(), {}, {}, CB, HC, t, POOL_C, 0) + ([-1.0] if PLACEFEAT else []) + ([-1.0] if MVFEAT else []), 1))
                for k, x in trk.items():
                    if k != tid and len(x['obs']) >= 2 and t - x['last'] <= GAP_S:
                        hx = max(1, len(x['obs']) // 2); neg = pseudo(x['obs'][:hx], x['feats'][:hx])
                        sn = float(np.max(np.stack(neg['res']) @ q2))
                        out_rows.append((ow_comb.pair_feats({-1: sn}, -1, neg, t2, a2, set(), {}, {}, CB, HC, t, POOL_C, 0) + ([-1.0] if PLACEFEAT else []) + ([-1.0] if MVFEAT else []), 0))
                        break
                return out_rows
            if FSM is not None:
                if FS_ADAPT:
                    for r_, l_ in selfsup_rows(): FSM.update(r_, l_)
                if rows:
                    sc = [FSM.score(r) for r in rows]; bi = int(np.argmax(sc)); best = top[bi] if sc[bi] >= FSTH else None
                else:
                    best = None
            if COMB is not None:
                if COMB_ADAPT:      # per-scene Platt recalibration of the GBM logit, SGD on free in-scene labels
                    cal = S.setdefault('_cal_' + vid, [1.0, 0.0])
                    for r_, l_ in selfsup_rows():
                        p0 = float(np.clip(COMB.prob([r_])[0], 1e-4, 1 - 1e-4)); z = np.log(p0 / (1 - p0))
                        pc = 1 / (1 + np.exp(-(cal[0] * z + cal[1]))); g = pc - l_
                        cal[0] -= COMB_LR * g * z; cal[1] -= COMB_LR * g
                if rows:
                    pr = COMB.prob(rows)
                    if COMB_ADAPT:
                        cal = S['_cal_' + vid]; pz = np.clip(pr, 1e-4, 1 - 1e-4); pr = 1 / (1 + np.exp(-(cal[0] * np.log(pz / (1 - pz)) + cal[1])))
                    if COMB_LAZY2:     # true two-level hierarchy: current session at θ; earlier sessions only if none passes, at θ + δ
                        cur_i = [k for k, u in enumerate(top) if ids[u]['last'] > -1e17]
                        old_i = [k for k, u in enumerate(top) if ids[u]['last'] <= -1e17]
                        best = None
                        if cur_i:
                            k = max(cur_i, key=lambda k: pr[k]); best = top[k] if pr[k] >= COMBTH else None
                        if best is None and old_i:
                            k = max(old_i, key=lambda k: pr[k]); best = top[k] if pr[k] >= COMBTH + COMB_DPREV else None
                    else:
                        bi = int(np.argmax(pr)); best = top[bi] if pr[bi] >= COMBTH else None
                        if COMB_DEFER and not force and len(tr['obs']) < 3 * K and len(pr) > 1:   # ambiguous: wait for more evidence
                            p_sorted = np.sort(pr)
                            if p_sorted[-1] >= COMBTH * 0.5 and p_sorted[-1] - p_sorted[-2] < COMB_DEFER:
                                tr['next_check'] = len(tr['obs']) + max(5, K // 2); st['deferred'] = st.get('deferred', 0) + 1
                                return
                else:
                    best = None
            tr['_tagg'] = tagg
            # hard-negative banks: identities this tracklet was close to (within 0.1 of tau) but not assigned to
            for u in top:
                if u != best and app_all[u] >= TAU - 0.1:
                    nb_ = ids[u].setdefault('neg', []); nb_.append(q)
                    if len(nb_) > 32: nb_.pop(0)
        if MARGIN and best is not None:
            sc2 = sorted((float(np.max(np.stack(d_['res']) @ q)) for u, d_ in ids.items()
                          if u not in visible and d_['res']), reverse=True)
            if len(sc2) > 1 and sc2[0] - sc2[1] < MARGIN:
                st['amb'] = st.get('amb', 0) + 1
                if AMB == 'defer' and not force and len(tr['obs']) < DEFER_X * K:
                    tr['next_check'] = len(tr['obs']) + 25
                    return
                if AMB == 'new':
                    best = None
        if DIAG and record:
            gl = [G[fn][i] for fn, i in tr['obs']]; tg = max(set(gl), key=gl.count)
            own = {u: max(set(c), key=c.count) for u, c in igt.items() if c}
            T = [u for u, g in own.items() if g == tg and u not in visible and ids[u]['res']]
            def sc(u): return float(np.max(_R(ids[u]) @ q))
            if not T:
                cat = 'correct-new' if best is None else 'false-merge'
            else:
                Tb = max(T, key=sc)
                cat = 'correct-reid' if best in T else ('missed-reid' if best is None else 'wrong-reid')
            Q = []      # all non-visible candidates: (appearance score, id)
            for u, d_ in ids.items():
                if u not in visible and d_['res']:
                    Q.append((sc(u), u))
            Q.sort(reverse=True)
            def relv(u):
                nb = S.get('cov', {}).get(u, {})
                return sum(nb.get(v_, 0) >= POOL_C for v_ in visible) / len(visible) if visible else None
            bb = best if best is not None else (Q[0][1] if Q else None)
            feat = dict(s1=Q[0][0] if Q else None, s2=Q[1][0] if len(Q) > 1 else None,
                        nabove=sum(x >= TAU for x, _ in Q), nvis=len(visible),
                        bgap=(t - ids[bb]['last']) if bb is not None else None,
                        brel=relv(bb) if bb is not None else None,
                        bn=sum(len(r_) for r_ in [ids[bb]['res']]) if bb is not None else None)
            bl = (own.get(best) if best is not None else None)
            rec = dict(feat, cat=cat, n=len(gl), tg=str(tg), f0=tr['obs'][0][0], f1=tr['obs'][-1][0],
                       T_last_fn=ids[Tb].get('last_fn') if T else None,
                       B_last_fn=ids[best].get('last_fn') if best is not None else None, bl=None if bl is None else str(bl), gap=t - max([ids[u]['last'] for u in T], default=t) if T else None,
                       sT=sc(Tb) if T else None, sB=sc(best) if best is not None else None,
                       pool=(pool_adj(Tb) < 0) if (T and (PB or PN)) else None)
            S.setdefault('diag', []).append(rec)
        if os.environ.get('GM_ORACLE_JUNK') == '1':   # DIAGNOSTIC ONLY (uses labels): junk tracklets never enter memory
            gl_ = [G[fn][i] for fn, i in tr['obs']]
            if str(max(set(gl_), key=gl_.count)).startswith('junk'):
                best = None; tr['obs_junk'] = True
        if PAIRDUMP or (LEARN_M is not None and 'dec' in LEARN_M):
            cand_ = [u for u, d_ in ids.items() if u not in visible and d_['res'] and not d_.get('oneoff')]
            t0_ = int(tr['obs'][0][0].split('_')[1]) / fps + toff
            rows_ = dec_feats(tr, cand_, q, t0_) if cand_ else []
            if PAIRDUMP: PD_dec.append((tid, [(u, f) for u, f in rows_]))
            if LEARN_M is not None and 'dec' in LEARN_M:
                m_, th_ = LEARN_M['dec']; best = None
                if rows_:
                    P_ = m_.predict_proba([f for _, f in rows_])[:, 1]; j_ = int(np.argmax(P_))
                    if P_[j_] >= th_: best = rows_[j_][0]
        if DECLOG:
            fo_ = tr['obs'][0]
            S.setdefault('declog', []).append(dict(vid=vid, kind='skip', q_fn=fo_[0], q_box=[float(x) for x in out[fo_[0]]['track_bboxes'][fo_[1]]],
                vis=sorted(u for u in visible if u is not None), semx=sorted(u for u, d_ in ids.items() if SEMCL and d_.get('pr') and tr_pr is not None and d_['pr'].most_common(1)[0][0] != tr_pr),
                oneoff=sorted(u for u, d_ in ids.items() if d_.get('oneoff'))))
        if DECLOG and _dl:
            _dl.sort(key=lambda x: -x[0]); S.setdefault('declog', []).extend([dict(vid=vid, s=x[0], q_fn=x[1], q_box=x[2], c_fn=x[3], c_box=x[4], cand=x[5], rank=r_) for r_, x in enumerate(_dl[:5])])
        if ENROL and not active(tr):
            best = None; tr['obs_junk'] = True; tr['inactive'] = True     # never enrolled: one-off, output as no identity
        if BIRTHS and best is None:
            msc = float(np.mean([SC[fn][i] for fn, i in tr['obs']]))
            if msc < BIRTHS: tr['obs_junk'] = True       # low-confidence tracklet: may match, may not found an identity
        if best is None and (len(tr['obs']) < MINNEW or tr.get('obs_junk')):     # too short to found a memory identity: one-off, not stored
            iid = new_identity(); ids[iid]['oneoff'] = True
            if tr.get('inactive'): ids[iid]['inactive'] = True
        else:
            iid = best if best is not None else new_identity()
        if FUSE is not None and best is not None:
            for m_, c_ in sorted(_pass, reverse=True):
                if c_ == best or m_ < FUSE or ids[c_].get('oneoff') or cov.get(best, {}).get(c_, 0) > 0 or cov.get(c_, {}).get(best, 0) > 0: continue
                if FUSENB > -1:
                    na_ = {x_ for x_, k_ in cov.get(c_, {}).items() if k_ >= POOL_C and x_ != best}; nb_ = {x_ for x_, k_ in cov.get(best, {}).items() if k_ >= POOL_C and x_ != c_}
                    if not na_ or not nb_ or len(na_ & nb_) / len(na_ | nb_) < FUSENB: continue
                if FUSESIM > -1 and (not ids[c_]['res'] or not ids[best]['res'] or float(np.max(np.stack(ids[c_]['res']) @ np.stack(ids[best]['res']).T)) < FUSESIM): continue
                fuse(c_, best)
        if DIAG and record:
            igt.setdefault(iid, []).extend(gl)
        if PROV:
            if best is None: ids[iid]['conf'] = False     # new identity starts provisional
            else: ids[iid]['conf'] = True                 # re-identified by a second tracklet: confirmed
        if OWC:
            ids[iid]['agg'] = ow_comb.merge_agg(ids[iid].get('agg'), tr.get('_tagg', ow_comb.tracklet_agg(CB, tr['obs'])))
            if CUEDUMP and record:
                igt.setdefault(iid, []).extend([G[fn][i] for fn, i in tr['obs']])
        if RECIP:
            rq = S.setdefault('rq', []); rq.append((q, iid))
            if len(rq) > 50: rq.pop(0)
        tr['ident'] = iid; tr['decided'] = True
        if LAGLOG: TD[tid] = CLK[0]
        if CSLS:
            qb = S.setdefault('qbuf', []); qb.append(q)
            if len(qb) > 50: qb.pop(0)
        if ABSM and best is not None and st.get('last_margin', 1.0) < ABSM:
            tr['noabsorb'] = True        # weak match: label it, but don't let it contaminate the identity's reservoir
        lab_ = iid
        if (FUSELAB or LATE) and S.get('alias'):
            while lab_ in S['alias']: lab_ = S['alias'][lab_]
        for fn, i in tr['obs']:
            out[fn]['track_ids'][i] = lab_
        absorb(tid, iid, t)

    def absorb(tid, iid, t):
        if LATE and iid in ids and ids[iid].get('born') is not None and t - ids[iid]['born'] <= LATE_S and len(ids[iid]['res']) >= LATE_N and iid not in S.get('alias', {}):
            Ru = np.stack(ids[iid]['res']); q_ = Ru.mean(0); q_ /= np.linalg.norm(q_) + 1e-8; cv_ = S.get('cov', {}); best_, bs_ = None, LATE
            for v_, d_ in ids.items():
                if v_ == iid or not d_['res'] or d_.get('oneoff') or d_['last'] >= ids[iid]['born'] or cv_.get(iid, {}).get(v_, 0) > 0 or cv_.get(v_, {}).get(iid, 0) > 0: continue
                sc_ = float(np.max(np.stack(d_['res']) @ q_))
                if sc_ > bs_: best_, bs_ = v_, sc_
            if best_ is not None:
                r_ = best_
                while r_ in S.get('alias', {}): r_ = S['alias'][r_]
                if r_ != iid: S.setdefault('alias', {})[iid] = r_; st['late'] = st.get('late', 0) + 1
        if SIZE and trk[tid]['obs']: ids[iid].setdefault('la', []).append(_la(trk[tid]))
        if (PAIRDUMP or LEARN) and trk[tid].get('pc'):
            ip_ = ids[iid].setdefault('pc', {})
            for k_, v_ in trk[tid]['pc'].items(): ip_[k_] = ip_.get(k_, 0) + v_
            trk[tid]['pc'] = {}
        if SEMCL:
            import collections as _co
            ids[iid].setdefault('pr', _co.Counter()).update(x_ for x_ in (PR.get(fn, [None] * 999)[i] for fn, i in trk[tid]['obs']) if x_ is not None)
        if LOCW and L3 is not None and trk[tid]['obs']:
            fn_, i_ = trk[tid]['obs'][-1]; p_ = L3.get(fn_)
            if p_ is not None and i_ < len(p_): S.setdefault('loc', {})[iid] = np.asarray(p_[i_], float).ravel()[:3]
        if (RAYGATE or RAYBONUS) and RAY:            # identity's 3D estimate: running ray sums (A, b) + spread
            rs = S.setdefault('rays', {}).setdefault(iid, dict(A=np.zeros((3, 3)), b=np.zeros(3), n=0, m=np.zeros(3), mc=1.0))
            if RAYRESET and rs['n'] >= 3 and rs['mc'] <= np.cos(np.radians(3.0)):
                # moved-aware: if this accepted tracklet sits far from the identity's estimate, the object was moved ->
                # restart its 3D estimate at the new place so later tracklets are gated against where it is now
                p_ = np.linalg.solve(rs['A'] + 1e-9 * np.eye(3), rs['b']); md = []
                for fn_, i_ in trk[tid]['obs'][:8]:
                    r_ = ray_of(fn_, out[fn_]['track_bboxes'][i_])
                    if r_ is not None: C_, d_ = r_; v_ = p_ - C_; md.append(np.linalg.norm(v_ - (v_ @ d_) * d_))
                if md and np.median(md) > RAYD: rs.update(A=np.zeros((3, 3)), b=np.zeros(3), n=0, m=np.zeros(3), mc=1.0)
            for fn_, i_ in trk[tid]['obs']:
                r_ = ray_of(fn_, out[fn_]['track_bboxes'][i_])
                if r_ is None: continue
                C_, d_ = r_; Pm = np.eye(3) - np.outer(d_, d_); rs['A'] += Pm; rs['b'] += Pm @ C_; rs['n'] += 1; rs['m'] += d_
                rs['mc'] = min(rs['mc'], float(d_ @ (rs['m'] / np.linalg.norm(rs['m']))))
        if (MVB or MVFEAT) and HC is not None and len(trk[tid]['obs']) >= 2:   # was the identity moving (carried) when last seen?
            (fa_, ia_), (fb_, ib_) = trk[tid]['obs'][-2], trk[tid]['obs'][-1]
            ba_, bb_ = out[fa_]['track_bboxes'][ia_], out[fb_]['track_bboxes'][ib_]
            pa_ = HC.predict(fa_, fb_, ba_)
            if pa_ is not None:
                S.setdefault('mv', {})[iid] = float(np.hypot(pa_[0] - (bb_[0] + bb_[2]) / 2, pa_[1] - (bb_[1] + bb_[3]) / 2)) / 854.
        if PLACE or PLACEFEAT:      # the identity was seen in the views of this tracklet
            for fn_, _ in trk[tid]['obs']:
                j_ = S.get('vix', {}).get(int(int(fn_.split('_')[1]) / fps + toff))
                L_ = S.setdefault('uv', {}).setdefault(iid, [])
                if j_ is not None and j_ not in L_[-20:]: L_.append(j_)
        d = ids[iid]
        if trk[tid].get('noabsorb'):
            trk[tid]['feats_new'] = []; d['last'] = max(d['last'], t); return
        fnew = trk[tid]['feats_new']
        if GATE and CB_REF[0]:          # gated writes: only clean views (not cut off, not blurred, not tiny) enter memory
            CBg = CB_REF[0]; ob = trk[tid]['obs'][-len(fnew):] if fnew else []; keep_ = []
            for v, (fn_, i_) in zip(fnew, ob):
                fr_ = CBg.get(fn_); dd = fr_['dets'][i_] if fr_ and i_ < len(fr_['dets']) else None
                ok_ = dd is None or ((not GATE_BORDER or not dd['border']) and dd['area'] >= GATE_A and fr_['blur'] >= GATE_B)
                if ok_: keep_.append(v)
            fnew = keep_ if (keep_ or d['res']) else fnew[:1]     # never leave a new identity empty
        if MIX:      # per-identity mixture of appearance clusters with spread
            mx = d.setdefault('mix', [])
            for v in fnew:
                if mx:
                    cs_ = [float(c[0] @ v) for c in mx]; b_ = int(np.argmax(cs_))
                if mx and (1 - cs_[b_] <= MIX_R or len(mx) >= MIX_M):
                    c = mx[b_]; n_ = c[1] + 1; cen = c[0] * c[1] + v; cen /= np.linalg.norm(cen) + 1e-8
                    mx[b_] = [cen, n_, c[2] + (1 - cs_[b_])]
                else:
                    mx.append([v / (np.linalg.norm(v) + 1e-8), 1, 0.0])
        for v in fnew:
            if len(d['res']) < RES:
                d['res'].append(v); _bump(d)
            elif CORESET:      # keep a diverse coreset: evict the most redundant stored view (if the new one adds diversity)
                Rm = np.stack(d['res']); G = Rm @ Rm.T; np.fill_diagonal(G, -1); red = G.max(1); k_ = int(np.argmax(red))
                if float(np.max(Rm @ v)) < red[k_]: d['res'][k_] = v; _bump(d)
            else:
                d['res'][rng.integers(RES)] = v; _bump(d)
        trk[tid]['feats_new'] = []
        d['pl'] = S.get('pcur'); d['lf'] = S.get('pf_now')
        d.setdefault('first', t); d['last'] = max(d['last'], t); d['chunk'] = S['cur']; d['lastbox'] = trk[tid]['box']
        if SEARCH == 'kind':      # region index: members + kind histogram per region, kept incrementally (O(1) per absorb)
            kd_ = d['pr'].most_common(1)[0][0] if d.get('pr') else None; key_ = (d['chunk'], kd_)
            if d.get('_rk') != key_:
                RM, RK = S.setdefault('rmem', {}), S.setdefault('rkind', {})
                if d.get('_rk') is not None:
                    oc, ok_ = d['_rk']; RM.get(oc, set()).discard(iid)
                    if ok_ is not None: RK[oc][ok_] -= 1
                RM.setdefault(d['chunk'], set()).add(iid)
                if kd_ is not None: RK.setdefault(d['chunk'], {}); RK[d['chunk']][kd_] = RK[d['chunk']].get(kd_, 0) + 1
                d['_rk'] = key_
        if trk[tid]['obs']: d['last_fn'] = trk[tid]['obs'][-1][0]

    SC = {e['frame_name']: [a.get('score', 1.0) for a in e.get('annotations', [])] for e in ann}
    import ow_comb
    RAY = {}
    PRC = {e_['frame_name']: [a_.get('prompt') for a_ in e_.get('annotations', [])] for e_ in ann} if (PAIRDUMP or LEARN) else {}
    TOBS = {}; PD_link = []; PD_dec = []
    def link_feats(fn, i, k, c, o, t):
        x = trk[k]; pc = x.get('pca', {}); pi = PRC.get(fn, [None] * 999)[i]
        top = max(pc, key=pc.get) if pc else None
        b, lb = boxes[i], x['box']; ar = ((b[2] - b[0]) * (b[3] - b[1]) + 1) / ((lb[2] - lb[0]) * (lb[3] - lb[1]) + 1)
        return [c, o, t - x['last'], SC.get(fn, [1.0] * 999)[i], float(pi is not None and pi == top), float(pi is None or top is None), float(np.log(ar)), len(x['obs'])]
    def dec_feats(tr, cand, q, t0):
        rows = []
        tp = __import__('collections').Counter(x_ for x_ in (PRC.get(fn_, [None] * 999)[i_] for fn_, i_ in tr['obs']) if x_ is not None)
        tpm = tp.most_common(1)[0][0] if tp else None
        sims = {}
        for u in cand:
            R_ = _R(ids[u]) @ q; sims[u] = (float(R_.max()), float(np.mean(np.sort(R_)[-5:])), len(R_))
        order = sorted(cand, key=lambda u: -sims[u][0])[:10]
        best_s = sims[order[0]][0] if order else 0.0; second = sims[order[1]][0] if len(order) > 1 else 0.0
        for r_, u in enumerate(order):
            ip = ids[u].get('pc', {}); ipm = max(ip, key=ip.get) if ip else None
            other = second if r_ == 0 else best_s
            rows.append((u, [sims[u][0], sims[u][1], sims[u][0] - other, float(r_), float(np.log1p(sims[u][2])), t0 - ids[u]['last'],
                             float(tpm is not None and tpm == ipm), float(tpm is None or ipm is None), len(tr['obs']), float(len(cand))]))
        return rows
    CLK = [0.0]; LL = []; TD = {}; TK = {}
    L3S = pickle.load(open(os.path.join(w.OBS, vid, '3D_feat_aligned.pkl'), 'rb')) if STLOC else None
    PR = {e_['frame_name']: [a_.get('prompt') for a_ in e_.get('annotations', [])] for e_ in ann} if SEMCL else {}
    VD = None
    if VERIFY:
        _vp = os.path.join(VERIFY, f'{vid}.pkl'); VD = pickle.load(open(_vp, 'rb')) if os.path.exists(_vp) else {}
    if ENROL:
        def _ov(a, b):
            x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); inter = max(0, x2 - x1) * max(0, y2 - y1)
            return inter, inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
        if ENROL == 'oracle':
            _GTB = {e['frame_name']: [a_['bounding_box'] for a_ in e.get('annotations', [])] for e in
                    json.load(open(os.path.join(w.OBS, vid, 'object_annotations.json')))['video_annotations']}
            def active(tr):
                return any(_ov(out[fn]['track_bboxes'][i], g_)[1] >= 0.5 for fn, i in tr['obs'] for g_ in _GTB.get(fn, []))
        else:
            _HB = {fn_: [h_['box'] for h_ in r_.get('hands', []) if h_['s'] >= float(os.environ.get('GM_HAND_S', 0.5))]
                   for fn_, r_ in json.load(open(os.path.join(os.environ['GM_HANDS'], f'{vid}.json'))).items()}
            HPAD = float(os.environ.get('GM_HAND_PAD', 0.0))
            def active(tr):     # touched by a hand in some frame (boxes overlap after padding the hand box)
                for fn, i in tr['obs']:
                    b_ = out[fn]['track_bboxes'][i]
                    for h_ in _HB.get(fn, []):
                        pw, ph = HPAD * (h_[2] - h_[0]), HPAD * (h_[3] - h_[1])
                        if _ov(b_, [h_[0] - pw, h_[1] - ph, h_[2] + pw, h_[3] + ph])[0] > 0 and _ov(b_, h_)[1] < 0.8: return True
                return False
    L3 = pickle.load(open(os.path.join(w.OBS, vid, f'3D_feat_{LOCARM}.pkl'), 'rb')) if LOCW else None
    PF = None
    if PLACEF and os.path.exists(os.path.join(PLACEF, f'{vid}_frame.npz')):
        _z = np.load(os.path.join(PLACEF, f'{vid}_frame.npz')); _f = _z[PLKEY].astype(np.float32); _f /= np.linalg.norm(_f, axis=1, keepdims=True) + 1e-8
        PF = {str(n_): _f[i_] for i_, n_ in enumerate(_z['names'])}
    S['places'] = []; S['pcur'] = None; S['pf_now'] = None
    RAYSRC = os.environ.get('GM_RAYSRC')        # streaming poses instead: dir(s) with <vid>_ttt3r.npz (causal TTT3R, c2w, focal), ':'-separated
    if (RAYGATE or RAYBONUS) and RAYSRC:
        _np_ = [os.path.join(d_, f'{vid}_ttt3r.npz') for d_ in RAYSRC.split(':') if os.path.exists(os.path.join(d_, f'{vid}_ttt3r.npz'))]
        if _np_:
            _Z = np.load(_np_[0]); RAY = dict(src='ttt3r', fid=_Z['frame_ids'].astype(np.int64), P=_Z['poses_c2w'].astype(np.float64),
                                              f=_Z['focal'].astype(np.float64), hw=_Z['hw'])
    elif RAYGATE or RAYBONUS:                     # camera poses (EPIC-Fields, fair game: LMK / Whareformer use them) for box rays
        import sp_posegraph as _PG, cv2 as _cv2
        _src = [os.path.join(_PG.FIELDS, d_, f'{vid}.json') for d_ in ('poses', 'poses_train') if os.path.exists(os.path.join(_PG.FIELDS, d_, f'{vid}.json'))]
        if _src:
            _J = json.load(open(_src[0])); _f = _J['camera']['params']
            RAY = dict(P=_J['images'], K=np.array([[_f[0], 0, _f[2]], [0, _f[1], _f[3]], [0, 0, 1.0]]), dist=np.array(_f[4:8]), PG=_PG, cv2=_cv2)
    def ray_of(fn, box):
        if not RAY: return None
        if RAY.get('src') == 'ttt3r':         # nearest streaming frame (5 fps vs 2 fps keyframes); pinhole, centred principal point
            fno = int(fn.split('_')[1]); j = int(np.searchsorted(RAY['fid'], fno)); j = min(max(j, 0), len(RAY['fid']) - 1)
            if j > 0 and abs(RAY['fid'][j - 1] - fno) < abs(RAY['fid'][j] - fno): j -= 1
            H_, W_ = float(RAY['hw'][0]), float(RAY['hw'][1]); fx = RAY['f'][j]
            u_ = (box[0] + box[2]) / 2 * W_ / 854.; v_ = (box[1] + box[3]) / 2 * H_ / 480.
            Pc = RAY['P'][j]; d_ = Pc[:3, :3] @ np.array([(u_ - W_ / 2) / fx, (v_ - H_ / 2) / fx, 1.0]); return Pc[:3, 3].copy(), d_ / np.linalg.norm(d_)
        q = RAY['P'].get(fn + '.jpg')
        if q is None: return None
        R_ = RAY['PG'].quat_R(*q[:4]); C_ = -R_.T @ np.array(q[4:7])
        x_, y_ = RAY['cv2'].undistortPoints(np.float32([[[(box[0] + box[2]) / 2 * RAY['PG'].SX, (box[1] + box[3]) / 2 * RAY['PG'].SY]]]), RAY['K'], RAY['dist']).reshape(2)
        d_ = R_.T @ np.array([x_, y_, 1.0]); return C_, d_ / np.linalg.norm(d_)
    CB = ow_comb.load_cb(CUES_DIR, vid) if ((OWC or GATE or MVB or MVFEAT) and CUES_DIR) else None
    CB_REF[0] = CB
    HC = ow_comb.HChain(CB, sorted(CB)) if CB else None
    COMB = (ow_comb.Combiner(COMB_MAP[vid]) if (COMB_MAP and vid in COMB_MAP) else
            S.setdefault('_comb', ow_comb.Combiner(COMB_PATH))) if COMB_PATH else None
    FSM = ow_fs.FS(FS_PATH, adapt=FS_ADAPT, alpha=FSA) if FS_PATH else None      # fresh per video: per-scene calibration
    G = {e['frame_name']: [a['id' if w.DS == 'hd_epic' else 'name'] for a in e.get('annotations', [])] for e in ann}; igt = {}
    drng = np.random.default_rng(zlib.crc32(vid.encode()) + 7)
    last_sec = None; fn_prev = None; t_prev = None
    for e in sorted(ann, key=lambda e: int(e['frame_name'].split('_')[1])):
        fn = e['frame_name']; A = e.get('annotations', [])
        if fn not in apps or not A:
            continue
        t = int(fn.split('_')[1]) / fps + toff
        F = np.asarray(apps[fn], np.float32); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
        if PF is not None and fn in PF:
            g_ = PF[fn]; S['pf_now'] = g_
            if PLMODE == 'cluster':
                pl_ = S['places']; sims_ = [float(p_[0] @ g_) for p_ in pl_]
                j_ = int(np.argmax(sims_)) if sims_ else -1
                if j_ >= 0 and sims_[j_] >= PLTH:
                    p_ = pl_[j_]; v_ = p_[0] * p_[1] + g_; pl_[j_] = [v_ / (np.linalg.norm(v_) + 1e-8), p_[1] + 1]; S['pcur'] = j_
                else:
                    pl_.append([g_.copy(), 1]); S['pcur'] = len(pl_) - 1
        keep = drng.random(len(A)) >= DROP if DROP else np.ones(len(A), bool)
        if int(t) != last_sec and keep.any():
            if (PB or PN or REL or RELG or RELC or SUP or DIAG or OWC or FOCUS or PLACE or PLACEFEAT) and fn_prev is not None:     # co-visibility edges from decided identities in one frame/s
                pres = [u for u in out[fn_prev]['track_ids'] if u is not None]
                cov, seen = S.setdefault('cov', {}), S.setdefault('seen', {})
                if RELG or OWC:
                    offs = S.setdefault('off', {})
                    ctr = {u: ((b[0] + b[2]) / 1708., (b[1] + b[3]) / 960., RELZ * (z_ or 0.0)) for u, b, z_ in zip(out[fn_prev]['track_ids'], out[fn_prev]['track_bboxes'], out[fn_prev].get('track_z', [None] * len(out[fn_prev]['track_ids']))) if u is not None}
                if SUP:
                    sup = S.setdefault('sup', {}); bxs_ = [(u, b) for u, b in zip(out[fn_prev]['track_ids'], out[fn_prev]['track_bboxes']) if u is not None]
                    for u, a in bxs_:
                        for v_, b in bxs_:
                            if v_ != u and inside(a, b):
                                su_ = sup.setdefault(u, {}); su_[v_] = su_.get(v_, 0) + 1
                if RELC:
                    con = S.setdefault('con', {}); bxs = [(u, b) for u, b in zip(out[fn_prev]['track_ids'], out[fn_prev]['track_bboxes']) if u is not None]
                    for ia, (u, a) in enumerate(bxs):
                        for v_, b in bxs[ia + 1:]:
                            if v_ != u and touch(a, b):
                                cu_ = con.setdefault(u, {}); cu_[v_] = cu_.get(v_, 0) + 1
                                cv_ = con.setdefault(v_, {}); cv_[u] = cv_.get(u, 0) + 1
                for u in pres:
                    seen[u] = t_prev; cu = cov.setdefault(u, {})
                    for v_ in pres:
                        if v_ != u:
                            cu[v_] = cu.get(v_, 0) + 1
                            if RELG or OWC:      # running mean of v's centre relative to u's (normalised image coords)
                                dl = [ctr[v_][c] - ctr[u][c] for c in range(3)]
                                o_ = offs.get((u, v_)); n_ = cu[v_]
                                offs[(u, v_)] = tuple(dl) if o_ is None else tuple(o_[c] + (dl[c] - o_[c]) / n_ for c in range(3))
            if PLACE or PLACEFEAT:      # identity-free view graph (one view per second): temporal + top-5 appearance object-set overlap
                import sp_common as SPC
                vr, ve, vix, uv = S.setdefault('vr', []), S.setdefault('ve', []), S.setdefault('vix', {}), S.setdefault('uv', {})
                rec = dict(F=F[keep]); i_ = len(vr)
                if vr: ve.append((i_ - 1, i_, 0.5))
                sims_ = sorted(((SPC.soft_overlap(rec, vr[j]), j) for j in range(max(0, i_ - 300), i_)), reverse=True)[:5]
                ve += [(j, i_, w_) for w_, j in sims_ if w_ > 0]
                vr.append(rec); vix[int(t)] = i_
            last_sec = int(t); chunk_update(F[keep], t)
            S['ctx'] = [(tt, f_) for tt, f_ in S.get('ctx', []) if t - tt <= CTX_S] + [(t, f_) for f_ in F[keep]]
        CLK[0] = t; S['tnow'] = t
        boxes = [a['bounding_box'] for a in A]
        for k in [k for k, x in trk.items() if t - x['last'] > GAP_S]:
            if not trk[k]['decided']:
                decide(k, trk[k]['last'])
            del trk[k]
        act = list(trk)
        C = np.full((len(A), len(act)), 9.0); _lq = []
        for i in np.flatnonzero(keep):
            for j, k in enumerate(act):
                c = float(F[i] @ trk[k]['feat']); o = iou(boxes[i], trk[k]['box'])
                if STLOC and L3S is not None and trk[k].get('p3') is not None and fn in L3S:
                    dd = float(np.linalg.norm(np.asarray(L3S[fn][i], float).ravel()[:3] - trk[k]['p3']))
                    if not np.isfinite(dd) or dd > STLOC: continue
                    if c >= STC or o >= STO or dd <= STLOC / 2: C[i, j] = (1 - c) + (1 - o) + STLW * dd
                    continue
                if PAIRDUMP and (c >= STC - 0.2 or o > 0):
                    PD_link.append((fn, i, k, link_feats(fn, i, k, c, o, t)))
                if LEARN_M is not None and 'link' in LEARN_M:
                    if c >= STC - 0.2 or o > 0: _lq.append((i, j, link_feats(fn, i, k, c, o, t)))
                    continue
                if c >= STC or o >= STO:
                    C[i, j] = (1 - c) + (1 - o)
        if _lq:      # learned linker, batched per frame (same values as per-pair calls)
            m_, th_ = LEARN_M['link']; P_ = m_.predict_proba([f_ for _, _, f_ in _lq])[:, 1]
            for (i_, j_, _), p_ in zip(_lq, P_):
                if p_ >= th_: C[i_, j_] = -np.log(max(float(p_), 1e-6))
        rows, cols = linear_sum_assignment(C) if len(act) else ([], [])
        match = {r: act[c_] for r, c_ in zip(rows, cols) if C[r, c_] < 9.0}
        out[fn] = dict(track_ids=[None] * len(A), track_obj_names=[a['name'] for a in A], track_bboxes=boxes,
                       track_z=[a.get('z') for a in A],
                       track_3d_locs=[np.zeros(3, np.float32)] * len(A))
        for i in np.flatnonzero(keep):
            if i in match:
                k = match[i]
            else:
                k = next_t; next_t += 1
                trk[k] = dict(feat=F[i], box=boxes[i], last=t, obs=[], feats=[], feats_new=[], ident=None, decided=False,
                              obs_box0=boxes[i])
            x = trk[k]
            x['feat'] = F[i]; x['box'] = boxes[i]; x['last'] = t; x['obs'].append((fn, i)); x['z'] = A[i].get('z')
            if STLOC and L3S is not None and fn in L3S: x['p3'] = np.asarray(L3S[fn][i], float).ravel()[:3]
            x['feats'].append(F[i]); x['feats_new'].append(F[i])
            if TRKLOG: TK[(fn, int(i))] = k
            if PAIRDUMP or LEARN:
                TOBS.setdefault(k, []).append((fn, i)); pi_ = PRC.get(fn, [None] * 999)[i]
                if pi_ is not None:
                    x.setdefault('pc', {}); x['pc'][pi_] = x['pc'].get(pi_, 0) + 1; x.setdefault('pca', {}); x['pca'][pi_] = x['pca'].get(pi_, 0) + 1
            if LAGLOG:
                if x['decided']: LL.append((fn, i, t, ('id', x['ident']), None))
                else:
                    q_ = np.mean(x['feats'], 0); q_ = q_ / (np.linalg.norm(q_) + 1e-8)
                    vis_ = {y['ident'] for kk, y in trk.items() if kk != k and y['ident'] is not None and t - y['last'] <= GAP_S}
                    bu_, bs_ = None, -1.0
                    for u_, d_ in ids.items():
                        if u_ in vis_ or not d_['res'] or d_.get('oneoff'): continue
                        sc_ = float(np.max(np.stack(d_['res']) @ q_))
                        if sc_ >= TAU and sc_ > bs_: bu_, bs_ = u_, sc_
                    LL.append((fn, i, t, ('id', bu_) if bu_ is not None else ('trk', k), k))
            if x['decided']:
                lab_ = x['ident']
                if (FUSELAB or LATE) and S.get('alias'):
                    while lab_ in S['alias']: lab_ = S['alias'][lab_]
                out[fn]['track_ids'][i] = lab_
                if len(x['feats_new']) >= 10:
                    absorb(k, x['ident'], t)
            elif len(x['obs']) >= max(K, x.get('next_check', 0)):
                decide(k, t, force=False)
        fn_prev, t_prev = fn, t
        if DROP:
            out[fn]['keep'] = keep
    for k in list(trk):
        if not trk[k]['decided']:
            decide(k, trk[k]['last'])
        else:
            absorb(k, trk[k]['ident'], trk[k]['last'])
    if DROP:     # dropped detections leave the output (scored on kept detections only; one-off ids would blow up IDF1's O(n^2) matrix)
        for fn in list(out):
            r = out[fn]; kp = r.pop('keep')
            for key in ('track_ids', 'track_obj_names', 'track_bboxes', 'track_3d_locs'):
                r[key] = [x for x, k_ in zip(r[key], kp) if k_]
            if not r['track_ids']:
                del out[fn]
    for fn, r in out.items():
        seen = {}
        for i, u in enumerate(r['track_ids']):
            if u is None:
                r['track_ids'][i] = new_identity()
            elif u in seen:
                b, a = r['track_bboxes'][i], r['track_bboxes'][seen[u]]
                x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
                o = max(0, x2 - x1) * max(0, y2 - y1) / (min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) + 1e-9)
                if CLOV < 1 and os.environ.get('GM_CLOVOUT') == 'big':   # strict: ONE box per identity per frame; the larger box keeps it
                    if (b[2] - b[0]) * (b[3] - b[1]) > (a[2] - a[0]) * (a[3] - a[1]):
                        r['track_ids'][seen[u]] = new_identity(); seen[u] = i; continue
                    r['track_ids'][i] = new_identity()
                elif not (CLOV < 1 and os.environ.get('GM_NOSCORE') == '1' and os.environ.get('GM_CLOVOUT', '1') == '1' and o >= CLOV):   # part/whole duplicates may share
                    r['track_ids'][i] = new_identity()
            seen.setdefault(r['track_ids'][i], i)
    if FUSEREL and S.get('alias'):
        al = S['alias']
        def root(u):
            while u in al: u = al[u]
            return u
        for r_ in out.values(): r_['track_ids'] = [root(u) if u is not None else None for u in r_['track_ids']]
    if PAIRDUMP:
        os.makedirs(PAIRDUMP, exist_ok=True)
        pickle.dump(dict(link=PD_link, dec=PD_dec, tobs=TOBS, out={fn_: list(r_['track_ids']) for fn_, r_ in out.items()}), open(os.path.join(PAIRDUMP, f'{vid}.pkl'), 'wb'))
    if TRKLOG:
        os.makedirs(TRKLOG, exist_ok=True); pickle.dump(TK, open(os.path.join(TRKLOG, f'{vid}.pkl'), 'wb'))
    if LAGLOG:
        os.makedirs(LAGLOG, exist_ok=True)
        pickle.dump(dict(rows=LL, tdec=TD, fps=fps, toff=toff), open(os.path.join(LAGLOG, f'{vid}.pkl'), 'wb'))
    if DECLOG:
        os.makedirs(DECLOG, exist_ok=True); pickle.dump(S.pop('declog', []), open(os.path.join(DECLOG, f'{vid}.pkl'), 'wb'))
    S['toff'] = (t if out else toff) + 1e5     # next video starts long after (prior identities count as 'older')
    return out


def prior_of(vid, pool, n):
    p = vid.split('_')[0]
    same = sorted(u for u in pool if u.split('_')[0] == p and u < vid)[-n:] if n else []
    nd = int(os.environ.get('GM_DISTRACT', 0))
    if nd:      # distractor memory: videos from OTHER participants (other kitchens): guaranteed non-matches
        others = sorted(u for u in pool if u.split('_')[0] != p)
        import zlib; rng = np.random.default_rng(zlib.crc32(vid.encode()))
        same = list(rng.choice(others, min(nd, len(others)), replace=False)) + same
    return same


MEMCACHE = os.environ.get('GM_MEMCACHE') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gm_memcache')
DRES = 64


def video_memory(vid, P):
    """identities + chunks a video leaves behind when tracked alone (cached; reservoirs capped at DRES, float16)"""
    os.makedirs(MEMCACHE, exist_ok=True)
    tag = '' if OBS_IN == w.OBS else '_' + (os.environ.get('GM_MEMTAG') or format(zlib.crc32(OBS_IN.encode()), 'x'))   # observation source in the key (GM_MEMTAG: path-independent)
    tag += '_owc' if OWC else ''
    cp = os.path.join(MEMCACHE, f'{vid}_{P[0]}_{P[2]}_{P[3]}{tag}.pkl')
    if os.path.exists(cp):
        return pickle.load(open(cp, 'rb'))
    S = dict(ids={}, chunks=[], cur=None, next_id=0, toff=0.0, stats=dict(dec=0, cmp=0))
    track(vid, S, P, record=False)
    rng = np.random.default_rng(0)
    ids = []
    for d in S['ids'].values():
        if not d['res']:
            continue
        R = np.stack(d['res']); R = R[rng.choice(len(R), min(DRES, len(R)), replace=False)]
        ids.append((R.astype(np.float16), d['chunk'], d.get('agg')))
    chunks = [np.stack(c['f']).astype(np.float16) for c in S['chunks']] if S['chunks'] else []
    mem = dict(ids=ids, chunks=chunks)
    pickle.dump(mem, open(cp, 'wb'))
    return mem


def load_memory(S, mem):
    """insert a cached video memory as OLD memory (seen long ago), with its own chunks"""
    off = len(S['chunks'])
    for F in mem['chunks']:
        F = F.astype(np.float32)
        S['chunks'].append(dict(f=list(F), n=[1] * len(F), last=[-1e18] * len(F)))
    for it in mem['ids']:
        R, c = it[0], it[1]
        iid = S['next_id']; S['next_id'] += 1
        S['ids'][iid] = dict(res=list(R.astype(np.float32)), last=-1e18, chunk=(off + c) if c is not None else None)
        if len(it) > 2 and it[2] is not None: S['ids'][iid]['agg'] = it[2]


def run(args):
    vid, out_dir, P, pool, n_prior = args
    S = dict(ids={}, chunks=[], cur=None, next_id=0, toff=0.0, stats=dict(dec=0, cmp=0))
    for pv in prior_of(vid, pool, n_prior):
        load_memory(S, video_memory(pv, P))
    S['stats'] = dict(dec=0, cmp=0)
    out = track(vid, S, P, record=True)
    os.makedirs(os.path.join(out_dir, vid), exist_ok=True)
    pickle.dump(out, open(os.path.join(out_dir, vid, 'tracking_outputs.pkl'), 'wb'))
    if DIAG:
        json.dump(S.get('diag', []), open(os.path.join(out_dir, vid, 'diag.json'), 'w'))
    if CUEDUMP:
        pickle.dump(S.get('cuedump', []), open(os.path.join(out_dir, vid, 'cuedump.pkl'), 'wb'))
    return vid, S['stats'], len(S['ids']), len(S['chunks'])


if __name__ == '__main__':
    out_dir, vids = sys.argv[1], sys.argv[2].split(',')
    TAU, DELTA, K, GAP_S, LOCAL_S = [float(x) for x in (sys.argv[3:8] + ['0.4', '0', '20', '0.25', '30'][len(sys.argv[3:8]):])]
    n_prior = int(os.environ.get('GM_PRIOR', 0))
    pool_file = os.environ.get('GM_POOLF_VIDS', 'graph_train_vids.txt')
    pool = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), pool_file)).read().strip().split(',') \
        if os.environ.get('GM_POOL', 'train') == 'train' else vids
    pool = sorted(set(pool) | set(vids))
    P = (TAU, DELTA, int(K), GAP_S, LOCAL_S)
    stats = []
    with Pool(int(os.environ.get('GM_WORKERS', 7))) as p:
        for v, s, nid, nch in p.imap_unordered(run, [(v, out_dir, P, pool, n_prior) for v in vids]):
            stats.append((s['dec'], s['cmp'], nid, nch))
    if os.environ.get('GM_NOSCORE') == '1':
        if stats:   # cost summary for scaling studies: decisions, comparisons, identities in memory at the end
            D_, C_ = sum(x[0] for x in stats), sum(x[1] for x in stats)
            print(f'COST decisions {D_} comparisons {C_} cmp/decision {C_ / max(1, D_):.0f} identities {sum(x[2] for x in stats)} chunks {sum(x[3] for x in stats)}', flush=True)
        sys.exit(0)
    per = w.rescore_run(out_dir, vids, clean_locs=True)
    A = w.aggregate(per)
    if DIAG:
        R = [r for v in vids for r in json.load(open(os.path.join(out_dir, v, 'diag.json')))]
        tot = sum(r['n'] for r in R)
        for c in ('correct-new', 'correct-reid', 'missed-reid', 'wrong-reid', 'false-merge'):
            Rc = [r for r in R if r['cat'] == c]
            m = lambda k: np.median([r[k] for r in Rc if r[k] is not None]) if any(r[k] is not None for r in Rc) else float('nan')
            print(f'DIAG {c:13s} decisions {len(Rc):5d}  det-weighted {100*sum(r["n"] for r in Rc)/tot:5.1f}%  '
                  f'med sT {m("sT"):.2f} sB {m("sB"):.2f} gap {m("gap"):.0f}s')
        Rm = [r for r in R if r['cat'] == 'missed-reid']
        for lo, hi in ((0.3, 0.4), (0.2, 0.3), (0, 0.2)):
            print(f'DIAG missed-reid with sT in [{lo},{hi}): det-weighted {100*sum(r["n"] for r in Rm if lo <= r["sT"] < hi)/tot:.1f}%')
    dec = sum(x[0] for x in stats); cmp_ = sum(x[1] for x in stats)
    print(f'drop={DROP} PB={PB} PN={PN} W={POOL_W} GraphMem2 search={SEARCH} prior={n_prior} topC={TOPC} DG={DG} tau={TAU} K={int(K)} gap={GAP_S}: '
          f'mPCL {A[0]:.1f} IDF1 {A[1]:.1f}  cmp/decision {cmp_/max(1,dec):.0f}  identities/memory {np.mean([x[2] for x in stats]):.0f}'
          f'  chunks {np.mean([x[3] for x in stats]):.0f}')
