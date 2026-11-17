"""identity-free helpers for place retrieval: objects are matched by appearance only (no names)"""
import numpy as np
from scipy.optimize import linear_sum_assignment
def centre(o): return np.array([(o['box'][0] + o['box'][2]) / 2, (o['box'][1] + o['box'][3]) / 2])
def size(o): return float(np.sqrt(max(1.0, (o['box'][2] - o['box'][0]) * (o['box'][3] - o['box'][1]))))
def view_rec(v):
    return dict(F=np.stack([o['feat'] for o in v['objs']]) if v['objs'] else np.zeros((0, 256), np.float32),
                C=np.stack([centre(o) for o in v['objs']]) if v['objs'] else np.zeros((0, 2)),
                Z=np.array([size(o) for o in v['objs']]))
def match(q, r, cmin=0.6):
    """Hungarian on appearance cosine; returns matched index pairs and their cosines"""
    if len(q['F']) == 0 or len(r['F']) == 0: return [], []
    S = q['F'] @ r['F'].T; i, j = linear_sum_assignment(-S); keep = S[i, j] >= cmin
    return list(zip(i[keep], j[keep])), list(S[i, j][keep])
def soft_overlap(q, r):
    pr, cs = match(q, r); return (sum(cs) / max(1, len(q['F']) + len(r['F']) - len(pr))) if pr else 0.0
