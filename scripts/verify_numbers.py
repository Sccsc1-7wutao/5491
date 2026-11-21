"""Independent re-computation of every number in handoff_strategy_fable_v3_cvpr.md that was not recomputed this session.

Event: VisEvent SR from the raw result directories (scripts/visevent_eval.sequence_scores).
Depth: official VOT F from each workspace's analysis results.json.
Thermal: B-evaluated runs are pulled with thermal_pull_score.pull_and_score and scored with the validated lasher_eval.
Personalized baselines (Local / FedPer / Ditto) = simple mean over the 3 sensor clients (the project's fixed protocol).
Output: ViPT\\output\\eval_results\\final_supp\\verify_numbers.md ; every row = recomputed vs claimed, flag if |diff| > 0.06.
Run: D:\\Conda\\envs\\sdstrack\\python.exe D:\\cvresearch\\scripts\\verify_numbers.py [--no-pull]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\cvresearch\scripts")
import thermal_pull_score as tp  # noqa: E402
from visevent_eval import sequence_scores  # noqa: E402

VIPT = Path(r"D:\cvresearch\ViPT")
OUT = VIPT / "output" / "eval_results" / "final_supp"
TK = Path(r"D:\cvresearch\tools\VisEvent_SOT_Benchmark")
VE = VIPT / "RGBE_workspace" / "results" / "VisEvent"
rows = []  # (section, name, recomputed, claimed)


def add(sec, name, val, claim):
    rows.append((sec, name, val, claim))


# ---------------- event ----------------
seqs = [x.strip() for x in (TK / "sequence_evaluation_config" / "VisEvent_testing_subset.txt").read_text().splitlines() if x.strip()]


def ev(dirname):
    s = sequence_scores(TK / "annos", VE / dirname, seqs)
    return float(np.mean([s[x][1] for x in seqs])) * 100


def ev_mean(prefix, ids):
    return float(np.mean([ev(f"{prefix}{i}") for i in ids]))


EVENT = [("FedAvg", "batch1_fedavg_event", 53.04), ("分桶", "batch1_modality_event", 56.07), ("独立", "batch2_indep_event_event", 55.52),
         ("α=0.25", "batch2_modality_a025_event", 53.91), ("α=0.5", "batch1_modality_a05_event", 54.60), ("α=0.75", "batch2_modality_a075_event", 55.06),
         ("FedProx μ=0.01(batch1)", "batch1_fedprox_event", None), ("FedProx μ=0.1(batch2)", "batch2_fedprox_mu01_event", 52.52),
         ("warm-up(ours)", "batch1_ours_event", 56.47), ("fa", "batch3_bucket_fa_event", 53.99), ("fna", "batch3_bucket_fna_event", 53.9),
         ("fna去纯RGB", "batch5_fna_norgb_event", 57.07), ("独立去纯RGB", "batch7_indep_event_norgb", 56.89), ("主方法去纯RGB", "b8m_event", 55.60),
         ("FedAvg去纯RGB", "b8f_event", 52.00), ("E2分桶", "batch6_e2_bucket_ds_event", 55.88),
         ("置零 分桶(含RGB)", "noaux_modality_event", 54.69), ("置零 FedAvg(含RGB)", "noaux_fedavg_event", 54.32),
         ("置零 独立(含RGB)", "noaux_batch2_indep_event", None), ("置零 独立去纯RGB", "noaux_batch7_indep_event_norgb", 54.80),
         ("置零 主方法去纯RGB", "noaux_b8m_event", 54.34), ("置零 fna去纯RGB", "noaux_batch5_fna_norgb_event", 54.50)]
for name, d, claim in EVENT:
    try:
        add("事件 SR", name, ev(d), claim)
    except Exception as e:
        add("事件 SR", name + " [ERROR %s]" % type(e).__name__, float("nan"), claim)
add("事件 SR", "Local(3 客户端均值)", ev_mean("batch2_local_", ["rgbe_000", "rgbe_002", "rgbe_003"]), 52.02)
add("事件 SR", "FedPer(3 客户端均值)", ev_mean("batch2_fedper_", ["rgbe_000", "rgbe_002", "rgbe_003"]), 54.21)
add("事件 SR", "Ditto(3 客户端均值)", ev_mean("batch4_ditto_", ["rgbe_000", "rgbe_002", "rgbe_003"]), 53.77)

# ---------------- depth ----------------


def dF(tag):
    ws = VIPT / f"Depthtrack_workspace_{tag}" / "analysis"
    cands = sorted(ws.glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(tag)
    j = json.loads(cands[0].read_text())
    return float(j["results"]["rgbd-unsupervised"]["results"][0][0][2]) * 100


DEPTH = [("FedAvg", "batch1_fedavg", 53.05), ("分桶", "batch1_modality", 55.68), ("独立", "batch2_indep_depth", 54.02),
         ("α=0.25", "batch2_modality_a025", 52.12), ("α=0.5", "batch1_modality_a05", 54.80), ("α=0.75", "batch2_modality_a075", 55.20),
         ("FedProx μ=0.01(batch1)", "batch1_fedprox", None), ("FedProx μ=0.1(batch2)", "batch2_fedprox_mu01", 54.59), ("warm-up(ours)", "batch1_ours", None),
         ("fa", "batch3_bucket_fa", 54.7), ("fna", "batch3_bucket_fna", 55.55), ("fna去纯RGB", "batch5_fna_norgb_depth", 56.49),
         ("主方法去纯RGB", "b8m_depth", 55.24), ("FedAvg去纯RGB", "b8f_depth", 55.78), ("独立去纯RGB", "b9id", 54.78), ("E2分桶", "batch6_e2_bucket_ds_depth", 56.16)]
for name, tag, claim in DEPTH:
    try:
        add("深度 F", name, dF(tag), claim)
    except Exception as e:
        add("深度 F", name + " [ERROR %s]" % type(e).__name__, float("nan"), claim)
for lab, pre, claim in (("Local", "batch2_local_rgbd_", 48.4), ("FedPer", "batch2_fedper_rgbd_", 51.58), ("Ditto", "batch4_ditto_rgbd_", None)):
    try:
        add("深度 F", f"{lab}(3 客户端均值)", float(np.mean([dF(pre + i) for i in ("001", "002", "003")])), claim)
    except Exception as e:
        add("深度 F", f"{lab}(3 客户端均值) [ERROR {type(e).__name__}]", float("nan"), claim)

# ---------------- thermal (B) ----------------
THERMAL = [("FedProx μ=0.01(batch1)", "B_batch1_fedprox", 46.63), ("FedProx μ=0.1(batch2)", "B_batch2_fedprox_mu01", None), ("warm-up(ours)", "B_batch1_ours", 46.65),
           ("α=0.25", "B_batch2_modality_a025", 46.68), ("α=0.5", "B_batch1_modality_a05", 47.22), ("α=0.75", "B_batch2_modality_a075", 47.40),
           ("fa", "B_fa", 44.55), ("fna", "B_newfna", 44.52), ("E2分桶", "B_e2_bucket_ds", 45.24), ("E2独立", "B_e2_indep_ds", 46.79),
           ("置零 FedAvg", "B_noaux_fedavg", None), ("置零 分桶", "B_noaux_modality", None)]
GROUPS = [("Local", "B_local_rgbt_", 47.58), ("FedPer", "B_fedper_rgbt_", 47.52), ("Ditto", "B_ditto_rgbt_", 45.32)]
if "--no-pull" not in sys.argv:
    for _, run, _ in THERMAL:
        try:
            tp.pull_and_score(run)
        except Exception as e:
            print("[pull ERROR]", run, e)
    for _, pre, _ in GROUPS:
        for i in ("001", "002", "003"):
            try:
                tp.pull_and_score(pre + i)
            except Exception as e:
                print("[pull ERROR]", pre + i, e)


def th(run):
    p = Path(tp.BRES) / f"{run}_curves.npz"
    if not p.exists():
        p = Path(tp.REF) / f"{run}_curves.npz"
    if not p.exists():
        raise FileNotFoundError(run)
    return float(np.load(p, allow_pickle=False)["curves_ovl"].mean(axis=1).mean()) * 100


for name, run, claim in THERMAL:
    try:
        add("热红外 AUC (B)", name, th(run), claim)
    except Exception as e:
        add("热红外 AUC (B)", name + " [missing]", float("nan"), claim)
for name, pre, claim in GROUPS:
    try:
        add("热红外 AUC (B)", f"{name}(3 客户端均值, B)", float(np.mean([th(pre + i) for i in ("001", "002", "003")])), claim)
    except Exception:
        add("热红外 AUC (B)", f"{name}(3 客户端均值, B) [missing]", float("nan"), claim)
for lab, run, claim in (("FedAvg", "batch1_fedavg", 47.07), ("分桶", "batch1_modality", 47.54), ("独立", "batch2_indep_thermal", 49.02),
                        ("主方法去纯RGB", "B_main_norgb_thermal", 48.23), ("FedAvg去纯RGB", "B_fedavg_norgb_thermal", 47.84), ("独立去纯RGB", "B_indep_thermal_norgb", 48.27),
                        ("fna去纯RGB", "B_norgb_thermal", 48.58), ("置零 主方法去纯RGB", "B_noaux_main_norgb", 44.52), ("置零 fna去纯RGB", "B_noaux_fna_norgb", 44.71)):
    try:
        add("热红外 AUC (B)", lab, th(run), claim)
    except Exception:
        add("热红外 AUC (B)", lab + " [missing]", float("nan"), claim)

# ---------------- report ----------------
OUT.mkdir(parents=True, exist_ok=True)
lines = ["# 材料包数字核对（重新从原始结果计算）", "", "| 类别 | 项目 | 重算 | 材料包里的数 | 判定 |", "|---|---|---:|---:|---|"]
bad = 0
for sec, name, val, claim in rows:
    if np.isnan(val):
        flag = "无法计算"; bad += 1
    elif claim is None:
        flag = "材料包未给数（本次补上）"
    else:
        flag = "一致" if abs(val - claim) <= 0.06 else f"**不一致（差 {val - claim:+.2f}）**"
        bad += flag.startswith("**")
    lines.append(f"| {sec} | {name} | {'—' if np.isnan(val) else f'{val:.2f}'} | {'—' if claim is None else claim} | {flag} |")
(OUT / "verify_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(l for l in lines if "一致（" in l or "无法" in l or "未给数" in l or "不一致" in l))
print("rows:", len(rows), " problems:", bad)
