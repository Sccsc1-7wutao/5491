"""Batch-1 re-analysis (2026-09-29): all pairwise paired tests, Holm correction, VisEvent attributes.

Reads existing raw results only (no inference). Scoring is delegated to the
project's validated scorers (scripts/visevent_eval.py, LasHeR npz curves).
Run from D:\\cvresearch\\scripts with D:\\Conda\\envs\\sdstrack\\python.exe.
Report: ViPT/output/eval_results/batch1_analysis_0929.md
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, r"D:\cvresearch\scripts")
from visevent_eval import sequence_scores  # noqa: E402

VIPT = Path(r"D:\cvresearch\ViPT")
TK = Path(r"D:\cvresearch\tools\VisEvent_SOT_Benchmark")
OUT = Path(r"D:\cvresearch\ViPT\output\eval_results")
V = ["fedavg", "fedprox", "modality_a05", "modality", "ours"]


def paired(d):
    n = d.size
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1)
    c = stats.t.ppf(0.975, n - 1)
    return m, se, t, p, (m - c * se, m + c * se), int((d > 0).sum()), int((d < 0).sum())


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def family(name, scores, pairs):
    rows = []
    for a, b in pairs:
        d = (scores[b] - scores[a]) * 100
        rows.append((f"{b} - {a}",) + paired(d))
    adj = holm([r[4] for r in rows])
    print(f"\n=== {name}  (n={len(next(iter(scores.values())))}, Holm over {len(rows)} tests)")
    print(f"{'comparison':<28}{'diff':>7}{'SE':>6}{'t':>7}{'p':>10}{'p_holm':>10}  CI95            W/L")
    for r, pa in zip(rows, adj):
        lab, m, se, t, p, ci, w, l = r
        print(f"{lab:<28}{m:+7.2f}{se:6.2f}{t:+7.2f}{p:10.2g}{pa:10.2g}  [{ci[0]:+.2f},{ci[1]:+.2f}]  {w}/{l}")


# ---------------------------------------------------------------- LasHeR
las = {}
names0 = None
for v in V:
    d = np.load(VIPT / "output" / "eval_results" / f"batch1_{v}_curves.npz")
    names = d["names"].astype(str)
    if names0 is None:
        names0 = names
    assert np.array_equal(names, names0)
    las[v] = {"AUC": d["curves_ovl"].mean(axis=1), "PR": d["curves_err"][:, 20]}
print("LasHeR means:", {v: (round(las[v]["AUC"].mean() * 100, 2), round(las[v]["PR"].mean() * 100, 2)) for v in V})

# ---------------------------------------------------------------- VisEvent
seqs = [x.strip() for x in (TK / "sequence_evaluation_config" / "VisEvent_testing_subset.txt").read_text().splitlines() if x.strip()]
ve = {}
for v in V + ["deep_rgbe"]:
    rd = VIPT / "RGBE_workspace" / "results" / "VisEvent" / (f"batch1_{v}_event" if v != "deep_rgbe" else "deep_rgbe")
    sc = sequence_scores(TK / "annos", rd, seqs)
    ve[v] = {"PR": np.array([sc[s][0] for s in seqs]), "SR": np.array([sc[s][1] for s in seqs])}
print("VisEvent means:", {v: (round(ve[v]["SR"].mean() * 100, 2), round(ve[v]["PR"].mean() * 100, 2)) for v in ve})

allpairs = list(combinations(V, 2))
primary = [("fedavg", "modality"), ("modality", "ours"), ("fedavg", "fedprox"), ("fedavg", "modality_a05")]
for metric in ["SR", "PR"]:
    family(f"VisEvent {metric} all pairs", {v: ve[v][metric] for v in V}, allpairs)
    family(f"VisEvent {metric} primary", {v: ve[v][metric] for v in V}, primary)
for metric in ["AUC", "PR"]:
    family(f"LasHeR {metric} all pairs", {v: las[v][metric] for v in V}, allpairs)
    family(f"LasHeR {metric} primary", {v: las[v][metric] for v in V}, primary)

# ---------------------------------------------------------------- VisEvent attributes
# order from Evaluate_VisEvent_SOT_benchmark.m att_fig_name
att_names = ['CM', 'ROT', 'DEF', 'FOC', 'LI', 'OV', 'POC', 'VC', 'SV', 'BC', 'MB', 'ARC', 'FM', 'NMO', 'IV', 'OE', 'BOM']
att = np.array([np.loadtxt(TK / "annos" / "att" / f"{s}.txt", ndmin=1) for s in seqs])
outdoor = np.array(["outdoor" in s.lower() for s in seqs])
print("\natt matrix", att.shape, "outdoor-in-name:", outdoor.sum())

cmp = [("fedavg", "modality"), ("modality", "ours"), ("fedavg", "modality_a05"), ("fedavg", "fedprox")]
print("\n=== VisEvent SR diff by attribute (pp; t over sequences in subset; EXPLORATORY)")
hdr = f"{'attr':<6}{'n':>4}" + "".join(f"{b+'-'+a:>26}" for a, b in cmp)
print(hdr)
rows = [("ALL", np.ones(len(seqs), bool))] + [(n, att[:, i] > 0) for i, n in enumerate(att_names)] + [("name:outdoor", outdoor), ("name:other", ~outdoor)]
for lab, msk in rows:
    line = f"{lab:<6}{msk.sum():>4}"
    for a, b in cmp:
        d = (ve[b]["SR"][msk] - ve[a]["SR"][msk]) * 100
        if msk.sum() >= 3:
            m, se, t, p, ci, w, l = paired(d)
            line += f"{m:+10.2f} (t={t:+5.2f}, {w:>3}/{l:<3})"
        else:
            line += f"{'n<3':>26}"
    print(line)

# interaction: is the modality-FedAvg gain different with/without an attribute? (Welch, exploratory)
g = (ve["modality"]["SR"] - ve["fedavg"]["SR"]) * 100
print("\nmodality-FedAvg SR gain, with vs without attribute (Welch t, exploratory):")
for a in ["NMO", "LI", "FM", "MB", "IV", "OE"]:
    m = att[:, att_names.index(a)] > 0
    r = stats.ttest_ind(g[m], g[~m], equal_var=False)
    print(f"  {a:<4} with {g[m].mean():+.2f} (n={m.sum()}) vs without {g[~m].mean():+.2f}; t={r.statistic:+.2f} p={r.pvalue:.3f}")

base = ve["fedavg"]["SR"] * 100
q = np.quantile(base, [0, .25, .5, .75, 1])
print("\nmodality-FedAvg SR gain by FedAvg-SR quartile:")
for lo, hi in zip(q[:-1], q[1:]):
    m = (base >= lo) & (base <= hi)
    print(f"  FedAvg SR in [{lo:5.1f},{hi:5.1f}] n={m.sum():3d} mean gain {g[m].mean():+.2f}")
print("share of total gain from top-20 sequences:", np.sort(g)[::-1][:20].sum() / g.sum())
print("official deep_rgbe (centralized, author weights) SR/PR:", ve["deep_rgbe"]["SR"].mean() * 100, ve["deep_rgbe"]["PR"].mean() * 100)

np.savez(OUT / "batch1_perseq_0929.npz", seqs=np.array(seqs), las_names=names0,
         **{f"ve_{v}_{k}": ve[v][k] for v in ve for k in ("SR", "PR")},
         **{f"las_{v}_{k}": las[v][k] for v in V for k in ("AUC", "PR")}, att=att)
