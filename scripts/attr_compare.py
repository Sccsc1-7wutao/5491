"""Attribute-stratified paired comparisons for the 10-04 result set (LasHeR thermal + VisEvent event).

For every (dataset, pair) the per-sequence score difference is split by each official attribute; Holm correction is applied across the
attributes of that (dataset, pair).  Descriptive only: single training seed, sequence-level pairing, intervals ignore training randomness.
Run: D:\\Conda\\envs\\sdstrack\\python.exe D:\\cvresearch\\scripts\\attr_compare.py
Outputs: ViPT\\output\\eval_results\\final_supp\\attr_compare.{csv,md}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, r"D:\cvresearch\scripts")
from visevent_eval import sequence_scores  # noqa: E402

VIPT = Path(r"D:\cvresearch\ViPT")
EV = VIPT / "output" / "eval_results"
OUT = EV / "final_supp"
BRES = EV / "Bres"
LAS_ATTR = EV / "batch3_supp" / "lasher_attrs"
TK = Path(r"D:\cvresearch\tools\VisEvent_SOT_Benchmark")
VE_RES = VIPT / "RGBE_workspace" / "results" / "VisEvent"

LAS = {  # label -> npz
    "fedavg": BRES / "ref" / "batch1_fedavg_curves.npz", "bucket": BRES / "ref" / "batch1_modality_curves.npz",
    "indep": BRES / "ref" / "batch2_indep_thermal_curves.npz", "main_norgb": BRES / "B_main_norgb_thermal_curves.npz",
    "indep_norgb": BRES / "B_indep_thermal_norgb_curves.npz", "fedavg_norgb": BRES / "B_fedavg_norgb_thermal_curves.npz",
    "fna_norgb": BRES / "B_norgb_thermal_curves.npz",
}
VE = {  # label -> result directory
    "fedavg": "batch1_fedavg_event", "bucket": "batch1_modality_event", "indep": "batch2_indep_event_event",
    "main_norgb": "b8m_event", "indep_norgb": "batch7_indep_event_norgb", "fedavg_norgb": "b8f_event",
    "fna_norgb": "batch5_fna_norgb_event",
}
PAIRS = [  # (name, a, b): difference a - b, positive = a better
    ("独立 − 分桶（含纯RGB）", "indep", "bucket"),
    ("分桶 − FedAvg（含纯RGB）", "bucket", "fedavg"),
    ("【公平检验】分桶去纯RGB − 独立去纯RGB", "main_norgb", "indep_norgb"),
    ("fna去纯RGB − 分桶去纯RGB", "fna_norgb", "main_norgb"),
    ("分桶去纯RGB − 分桶含纯RGB（去纯RGB的效应）", "main_norgb", "bucket"),
    ("FedAvg去纯RGB − FedAvg含纯RGB（去纯RGB的效应）", "fedavg_norgb", "fedavg"),
    ("分桶去纯RGB − FedAvg去纯RGB", "main_norgb", "fedavg_norgb"),
]


def holm(p):
    m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def load_las():
    scores, names0 = {}, None
    for k, f in LAS.items():
        if not f.is_file():
            continue
        d = np.load(f, allow_pickle=False); names = d["names"].astype(str)
        if names0 is None:
            names0 = names
        assert np.array_equal(names0, names), k
        scores[k] = d["curves_ovl"].mean(axis=1) * 100
    attrs = [x.strip() for x in (LAS_ATTR / "Attributes_order.txt").read_text(encoding="utf-8").split(",") if x.strip()]
    vec = np.stack([np.fromstring((LAS_ATTR / "AttriSeqsTxt" / f"{n}.txt").read_text(encoding="utf-8"), sep=",") for n in names0])
    return scores, attrs, vec


def load_ve():
    seqs = [x.strip() for x in (TK / "sequence_evaluation_config" / "VisEvent_testing_subset.txt").read_text().splitlines() if x.strip()]
    scores = {}
    for k, d in VE.items():
        if (VE_RES / d).is_dir():
            s = sequence_scores(TK / "annos", VE_RES / d, seqs); scores[k] = np.array([s[x][1] for x in seqs]) * 100
    names = ['CM', 'ROT', 'DEF', 'FOC', 'LI', 'OV', 'POC', 'VC', 'SV', 'BC', 'MB', 'ARC', 'FM', 'NMO', 'IV', 'OE', 'BOM']
    vec = np.array([np.loadtxt(TK / "annos" / "att" / f"{s}.txt", ndmin=1) for s in seqs])
    return scores, names, vec


def analyse(dsname, scores, attrs, vec, rows):
    for pname, a, b in PAIRS:
        if a not in scores or b not in scores:
            continue
        d_all = scores[a] - scores[b]
        entries = [("ALL", np.arange(d_all.size))] + [(attrs[j], np.flatnonzero(vec[:, j] == 1)) for j in range(len(attrs))]
        pv = []
        for name, idx in entries:
            d = d_all[idx]
            if idx.size < 8:
                pv.append(np.nan); continue
            pv.append(float(stats.ttest_1samp(d, 0).pvalue))
        valid = [i for i, p in enumerate(pv) if not np.isnan(p) and entries[i][0] != "ALL"]
        adj = {i: v for i, v in zip(valid, holm([pv[i] for i in valid]))}
        for i, (name, idx) in enumerate(entries):
            d = d_all[idx]; n = int(idx.size)
            if n < 8:
                rows.append((dsname, pname, name, n, "", "", "", "", "", "")); continue
            se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
            rows.append((dsname, pname, name, n, round(float(d.mean()), 3), round(float(d.mean() - tc * se), 3),
                         round(float(d.mean() + tc * se), 3), round(pv[i], 5), round(adj[i], 5) if i in adj else "", f"{int((d > 0).sum())}/{int((d < 0).sum())}"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    s, a, v = load_las(); analyse("LasHeR(AUC)", s, a, v, rows)
    s2, a2, v2 = load_ve(); analyse("VisEvent(SR)", s2, a2, v2, rows)
    hdr = ["dataset", "pair(a-b)", "attribute", "n", "mean_diff", "ci95_low", "ci95_high", "p_uncorrected", "p_holm", "wins/losses"]
    with open(OUT / "attr_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    lines = ["# 属性分层配对比较（描述性；单 seed；Holm 校正在每个对比的各属性之间）", ""]
    cur = None
    for r in rows:
        key = (r[0], r[1])
        if key != cur:
            cur = key; lines += ["", f"## {r[0]}：{r[1]}", "", "| 属性 | n | 均差 | 95% CI | p | Holm p | 胜/负 |", "|---|---:|---:|---|---:|---:|---|"]
        if r[4] == "":
            lines.append(f"| {r[2]} | {r[3]} | — | — | — | — | — |"); continue
        flag = " **" if (r[8] != "" and r[8] < 0.05) else ""
        lines.append(f"| {r[2]}{flag} | {r[3]} | {r[4]:+.2f} | [{r[5]:+.2f}, {r[6]:+.2f}] | {r[7]:.4f} | {r[8] if r[8] != '' else '—'} | {r[9]} |")
    (OUT / "attr_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # console summary: Holm-significant attributes only
    print("Holm<0.05 attributes (dataset | pair | attr | n | diff [CI] | holm p):")
    for r in rows:
        if r[8] != "" and r[8] < 0.05:
            print(f"  {r[0]} | {r[1]} | {r[2]} | n={r[3]} | {r[4]:+.2f} [{r[5]:+.2f},{r[6]:+.2f}] | {r[8]}")
    print("rows:", len(rows))


if __name__ == "__main__":
    main()
