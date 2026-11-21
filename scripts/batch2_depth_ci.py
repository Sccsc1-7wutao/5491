"""Batch-2 DepthTrack paired bootstrap: batch2 run vs batch1 references.

Reuses scripts/batch1_depth_ci.py (vot-toolkit 0.5.3 thresholds / TPR curves,
seed-42 draws of 20,000 x 50 sequences). Regression gate: batch1
modality - fedavg must reproduce +2.633 [+0.726, +4.653].
Run: D:\Conda\envs\vot\python.exe D:\cvresearch\scripts\batch2_depth_ci.py <run_name>
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import batch1_depth_ci as b1

run = sys.argv[1]
ws = b1.VIPT / f"Depthtrack_workspace_{run}"
names = sorted(p.name for p in b1.SEQUENCES.iterdir() if p.is_dir())
assert len(names) == 50

def load_root(root):
    data, scores = {}, []
    for n in names:
        seq = b1.load_sequence(str(b1.SEQUENCES / n)); f = root / n
        tr = b1.read_trajectory(str(next(f.glob("*_001.txt"))))
        cf = b1.load_confidence(next(f.glob("*_001_confidence.value")))
        assert len(tr) == len(seq) == len(cf), n
        data[n] = (seq, tr, cf); scores += cf
    return data, b1.determine_thresholds(scores, 100)

sets = {}
for v in ("fedavg", "modality"):
    d, th = b1.load_variant(v, names); sets[v] = (d, th, json.loads(b1.result_json(v).read_text())["results"]["rgbd-unsupervised"]["results"][0][0])
res_json = next((ws / "analysis").rglob("results.json"))
d, th = load_root(ws / "results" / run / "rgbd-unsupervised")
sets[run] = (d, th, json.loads(res_json.read_text())["results"]["rgbd-unsupervised"]["results"][0][0])

curves, per = {}, {}
for v, (d, th, off) in sets.items():
    c = b1.curve(d, names, th); err = np.max(np.abs(np.asarray(c["summary"]) - np.asarray(off)))
    print(f"acceptance {v}: recomputed {np.round(c['summary'],6)} official {np.round(off,6)} err {err:.2e}")
    assert err <= 1e-6
    sr = [b1.curve(d, [n], th) for n in names]
    per[v] = (np.asarray([x["pr"] for x in sr]), np.asarray([x["re"] for x in sr]))
    curves[v] = c

draws = np.random.default_rng(42).integers(0, 50, size=(20_000, 50))
def boot(a, b):
    (ap, ar), (bp, br) = per[a], per[b]
    fa = np.max(2*ap[draws].mean(1)*ar[draws].mean(1)/(ap[draws].mean(1)+ar[draws].mean(1)), axis=1)
    fb = np.max(2*bp[draws].mean(1)*br[draws].mean(1)/(bp[draws].mean(1)+br[draws].mean(1)), axis=1)
    diff = fa - fb; pt = curves[a]["summary"][2] - curves[b]["summary"][2]
    lo, hi = np.quantile(diff, [.025, .975])
    return pt*100, lo*100, hi*100, float(np.mean(diff > 0))

reg = boot("modality", "fedavg"); print("REGRESSION modality-fedavg %+.3f [%+.3f, %+.3f] P>0=%.4f" % reg)
assert abs(reg[0]-2.633) < 5e-3 and abs(reg[1]-0.726) < 5e-3 and abs(reg[2]-4.653) < 5e-3
for ref in ("modality", "fedavg"):
    print("%s - %s: %+.3f [%+.3f, %+.3f] P>0=%.4f" % ((run, ref) + boot(run, ref)))
