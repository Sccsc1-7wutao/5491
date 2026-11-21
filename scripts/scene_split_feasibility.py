"""Data-only feasibility check for scene-attribute-based client partitions (no training).

Uses the sequences of the current split (lasher_depthtrack_visevent_seed42_v2.json) so counts are comparable.
Run: D:\\Conda\\envs\\sdstrack\\python.exe D:\\cvresearch\\scripts\\scene_split_feasibility.py
"""
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

SPLIT = r"D:\cvresearch\ViPT\configs\fed\client_splits\lasher_depthtrack_visevent_seed42_v2.json"
DATA = r"D:\cvresearch\data"
ATTR = r"D:\cvresearch\ViPT\output\eval_results\batch3_supp\lasher_attrs"

d = json.load(open(SPLIT, encoding="utf-8"))
by_ds = defaultdict(list)
for c in d["clients"]:
    for s in c["sequence_ids"]:
        by_ds[c["dataset"]].append((s, c["client_id"]))


def nframes(root, seq):
    p = os.path.join(root, seq)
    best = 0
    for sub in os.listdir(p):
        q = os.path.join(p, sub)
        if os.path.isdir(q):
            best = max(best, len(os.listdir(q)))
    return best


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


print("== current Dirichlet split: client sizes (sequences / frames)")
for ds in ("lasher", "depthtrack", "visevent"):
    cs = [c for c in d["clients"] if c["dataset"] == ds]
    sz = [c["num_samples"] for c in cs]
    print(f"{ds}: seqs {[c['num_sequences'] for c in cs]} frames {sz}  max/min={max(sz)/min(sz):.0f}  gini={gini(sz):.2f}")

# ---------------- LasHeR ----------------
print("\n== LasHeR (881 sequences in the split)")
order = [x.strip() for x in open(os.path.join(ATTR, "Attributes_order.txt"), encoding="utf-8").read().split(",") if x.strip()]
root = os.path.join(DATA, "lasher", "trainingset")
rows = []
for s, cid in by_ds["lasher"]:
    v = np.fromstring(open(os.path.join(ATTR, "AttriSeqsTxt", s + ".txt"), encoding="utf-8").read(), sep=",")
    rows.append((s, v, nframes(root, s)))
V = np.stack([r[1] for r in rows]); F = np.array([r[2] for r in rows])
print("total frames", int(F.sum()))
print("attribute: sequences / frames (share of frames)")
for j, a in enumerate(order):
    m = V[:, j] == 1
    print(f"  {a:4s} {int(m.sum()):4d} seq  {int(F[m].sum()):7d} fr ({F[m].sum()/F.sum():.0%})")
ix = {a: i for i, a in enumerate(order)}
li, hi = V[:, ix["LI"]] == 1, V[:, ix["HI"]] == 1
print("LI&HI overlap:", int((li & hi).sum()), "seq; neither LI nor HI:", int((~li & ~hi).sum()), "seq", int(F[~li & ~hi].sum()), "fr")
occ = {a: V[:, ix[a]] == 1 for a in ("NO", "PO", "TO", "HO")}
print("occlusion groups (exclusive by worst level): ")
worst = np.where(occ["HO"], "HO", np.where(occ["TO"], "TO", np.where(occ["PO"], "PO", "NO")))
for g in ("NO", "PO", "TO", "HO"):
    m = worst == g; print(f"  {g}: {int(m.sum())} seq {int(F[m].sum())} fr")
# candidate: 3 groups = night(LI) / bright(HI, not LI) / normal
g_night = li; g_bright = hi & ~li; g_norm = ~li & ~hi
print("candidate illumination split  night(LI) / bright(HI,!LI) / normal:",
      [(int(m.sum()), int(F[m].sum())) for m in (g_night, g_bright, g_norm)])

# ---------------- DepthTrack ----------------
print("\n== DepthTrack (146 sequences in the split)")
root = os.path.join(DATA, "depthtrack", "train")
dt = [(s, nframes(root, s)) for s, _ in by_ds["depthtrack"]]
for tag in ("indoor", "wild"):
    sel = [(s, n) for s, n in dt if s.endswith("_" + tag)]
    print(f"  {tag}: {len(sel)} seq {sum(n for _, n in sel)} fr")
other = [(s, n) for s, n in dt if not (s.endswith("_indoor") or s.endswith("_wild"))]
print("  neither suffix:", len(other))
seqfile = os.path.join(root, dt[0][0], "sequence")
print("  per-frame tag files present in a train sequence:", [f for f in os.listdir(os.path.join(root, dt[0][0])) if f not in ("color", "depth", "groundtruth.txt", "sequence")])

# ---------------- VisEvent ----------------
print("\n== VisEvent (train sequences in the split; no attribute files for train)")
root = os.path.join(DATA, "visevent", "train_subset")
ve = [(s, nframes(root, s)) for s, _ in by_ds["visevent"]]
print("  total seq", len(ve), "frames", sum(n for _, n in ve))
cat, scene = Counter(), Counter()
catf = Counter()
for s, n in ve:
    m = re.match(r"^\d+_(.+?)_(indoor|outdoor)(\d*)$", s)
    if m:
        cat[m.group(1)] += 1; catf[m.group(1)] += n; scene[m.group(2)] += n
    else:
        scene["unparsed"] += n
print("  indoor/outdoor by name (frames):", dict(scene))
print("  target categories parsed:", len(cat), " top by frames:", [(k, cat[k], catf[k]) for k, _ in catf.most_common(8)])
