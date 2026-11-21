"""DepthTrack convergence summary: bucket (modality) vs FedAvg at r20/r60/r100/r120, with paired bootstrap differences.

Reuses the validated machinery of scripts/batch1_depth_ci.py (each run is acceptance-checked against its official vot analysis).
Run: D:\\Conda\\envs\\vot\\python.exe D:\\cvresearch\\scripts\\depth_conv.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import batch1_depth_ci as b1  # noqa: E402

names = sorted(p.name for p in b1.SEQUENCES.iterdir() if p.is_dir())
assert len(names) == 50
RUNS = {"mo20": "cvmo20d", "mo60": "cvmo60d", "mo100": "cvmo100d", "fa20": "cvfa20d", "fa60": "cvfa60d", "fa100": "cvfa100d"}


def load_run(run):
    ws = b1.VIPT / f"Depthtrack_workspace_{run}"
    res = next((ws / "analysis").rglob("results.json"))
    root = ws / "results" / run / "rgbd-unsupervised"
    data, scores = {}, []
    for n in names:
        seq = b1.load_sequence(str(b1.SEQUENCES / n)); f = root / n
        tr = b1.read_trajectory(str(next(f.glob("*_001.txt"))))
        cf = b1.load_confidence(next(f.glob("*_001_confidence.value")))
        assert len(tr) == len(seq) == len(cf), n
        data[n] = (seq, tr, cf); scores += cf
    th = b1.determine_thresholds(scores, 100)
    off = json.loads(res.read_text())["results"]["rgbd-unsupervised"]["results"][0][0]
    return data, th, off


sets = {k: load_run(v) for k, v in RUNS.items()}
for v, key in (("fedavg", "fa120"), ("modality", "mo120")):
    d, th = b1.load_variant(v, names)
    sets[key] = (d, th, json.loads(b1.result_json(v).read_text())["results"]["rgbd-unsupervised"]["results"][0][0])

curves, per = {}, {}
for k, (d, th, off) in sets.items():
    c = b1.curve(d, names, th)
    err = np.max(np.abs(np.asarray(c["summary"]) - np.asarray(off)))
    assert err <= 1e-6, (k, err)
    sr = [b1.curve(d, [n], th) for n in names]
    per[k] = (np.asarray([x["pr"] for x in sr]), np.asarray([x["re"] for x in sr]))
    curves[k] = c

draws = np.random.default_rng(42).integers(0, 50, size=(20_000, 50))


def fscore(k):
    p, r = per[k]
    return np.max(2 * p[draws].mean(1) * r[draws].mean(1) / (p[draws].mean(1) + r[draws].mean(1)), axis=1)


def boot(a, b):  # b - a
    diff = fscore(b) - fscore(a)
    pt = curves[b]["summary"][2] - curves[a]["summary"][2]
    lo, hi = np.quantile(diff, [.025, .975])
    return pt * 100, lo * 100, hi * 100


print("F (acceptance passed for all):")
for r in (20, 60, 100, 120):
    print(f"  r{r:<4d} bucket {curves['mo%d' % r]['summary'][2] * 100:6.2f}   FedAvg {curves['fa%d' % r]['summary'][2] * 100:6.2f}")
print("bucket - FedAvg per round:")
for r in (20, 60, 100, 120):
    print("  r%-4d %+.2f [%+.2f, %+.2f]" % ((r,) + boot("fa%d" % r, "mo%d" % r)))
print("r120 - r100:")
for m, lab in (("mo", "bucket"), ("fa", "FedAvg")):
    print("  %-7s %+.2f [%+.2f, %+.2f]" % ((lab,) + boot(m + "100", m + "120")))
print("r120 - r60:")
for m, lab in (("mo", "bucket"), ("fa", "FedAvg")):
    print("  %-7s %+.2f [%+.2f, %+.2f]" % ((lab,) + boot(m + "60", m + "120")))
