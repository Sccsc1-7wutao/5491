"""DepthTrack paired bootstrap between arbitrary finished runs (reuses the validated scripts/batch1_depth_ci.py machinery).

Run: D:\\Conda\\envs\\vot\\python.exe D:\\cvresearch\\scripts\\depth_pair.py
Each run needs Depthtrack_workspace_<run>\\results\\<run>\\rgbd-unsupervised and an analysis\\*\\results.json (official numbers).
Every run is acceptance-checked: the recomputed F-curve summary must equal the official vot analysis (err <= 1e-6).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import batch1_depth_ci as b1  # noqa: E402

names = sorted(p.name for p in b1.SEQUENCES.iterdir() if p.is_dir())
assert len(names) == 50

RUNS = {  # label -> workspace run name
    "main_norgb": "b8m_depth",
    "indep_norgb": "b9id",
    "fna_norgb": "batch5_fna_norgb_depth",
    "fedavg_norgb": "b8f_depth",
}


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


sets = {}
for label, run in RUNS.items():
    sets[label] = load_run(run)
for v in ("fedavg", "modality"):
    d, th = b1.load_variant(v, names)
    sets[v] = (d, th, json.loads(b1.result_json(v).read_text())["results"]["rgbd-unsupervised"]["results"][0][0])

curves, per = {}, {}
for v, (d, th, off) in sets.items():
    c = b1.curve(d, names, th)
    err = np.max(np.abs(np.asarray(c["summary"]) - np.asarray(off)))
    assert err <= 1e-6, (v, err)
    sr = [b1.curve(d, [n], th) for n in names]
    per[v] = (np.asarray([x["pr"] for x in sr]), np.asarray([x["re"] for x in sr]))
    curves[v] = c
    print(f"{v:12s} F = {c['summary'][2]*100:.2f}  (acceptance err {err:.1e})")

draws = np.random.default_rng(42).integers(0, 50, size=(20_000, 50))


def fscore(v):
    p, r = per[v]
    return np.max(2 * p[draws].mean(1) * r[draws].mean(1) / (p[draws].mean(1) + r[draws].mean(1)), axis=1)


def boot(a, b):  # b - a
    diff = fscore(b) - fscore(a)
    pt = curves[b]["summary"][2] - curves[a]["summary"][2]
    lo, hi = np.quantile(diff, [.025, .975])
    return pt * 100, lo * 100, hi * 100, float(np.mean(diff > 0))


pairs = [
    ("【公平检验·深度】主方法去纯RGB − 独立去纯RGB", "indep_norgb", "main_norgb"),
    ("主方法去纯RGB − 主方法含纯RGB(分桶)", "modality", "main_norgb"),
    ("主方法去纯RGB − FedAvg含纯RGB", "fedavg", "main_norgb"),
    ("fna去纯RGB − 主方法去纯RGB", "main_norgb", "fna_norgb"),
    ("fna去纯RGB − 独立去纯RGB", "indep_norgb", "fna_norgb"),
    ("FedAvg去纯RGB − FedAvg含纯RGB(去纯RGB的效应)", "fedavg", "fedavg_norgb"),
    ("主方法去纯RGB − FedAvg去纯RGB(分桶效应,去纯RGB下)", "fedavg_norgb", "main_norgb"),
    ("独立去纯RGB − FedAvg去纯RGB", "fedavg_norgb", "indep_norgb"),
]
for label, a, b in pairs:
    print("%-48s %+.3f [%+.3f, %+.3f] P>0=%.3f" % ((label,) + boot(a, b)))
# interaction: (main_norgb - modality) - (fedavg_norgb - fedavg), same bootstrap draws
inter = (fscore("main_norgb") - fscore("modality")) - (fscore("fedavg_norgb") - fscore("fedavg"))
pt = ((curves["main_norgb"]["summary"][2] - curves["modality"]["summary"][2]) - (curves["fedavg_norgb"]["summary"][2] - curves["fedavg"]["summary"][2])) * 100
lo, hi = np.quantile(inter, [.025, .975]) * 100
print("%-48s %+.3f [%+.3f, %+.3f] P>0=%.3f" % ("【交互项】主方法去纯RGB效应 − FedAvg去纯RGB效应", pt, lo, hi, float(np.mean(inter > 0))))
