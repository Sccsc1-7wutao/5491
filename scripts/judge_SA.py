"""S-A verdict against the FROZEN criteria (scripts/SA_success_criteria_FROZEN_1004.md). Nothing is tuned here.

Run: D:\\Conda\\envs\\sdstrack\\python.exe D:\\cvresearch\\scripts\\judge_SA.py
Writes ViPT\\output\\eval_results\\SA_verdict.md and prints it. Missing runs show as 'pending'.

Criteria (S-A, round 120 only): thermal AUC >= 48.0 (local caliber), event SR >= 56.07, depth F >= 55.0.
Step 1 (continue to more seeds?): S-A seed 42 passes all three.
Step 2 (final): the MEAN over seeds 42/0/123 passes all three; every seed's numbers are listed, none dropped.
Thermal must be read with the paired CIs (S-A vs bucket, S-A vs independent): cross-GPU eval noise is ~1 pt.
Thermal numbers here are all LOCAL-caliber (4080S); never mix with B-caliber numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\cvresearch\scripts")
from paired_compare import bootstrap_ci  # noqa: E402
from scipy import stats  # noqa: E402

VIPT = Path(r"D:\cvresearch\ViPT")
EV = VIPT / "output" / "eval_results"
VE = VIPT / "RGBE_workspace" / "results" / "VisEvent"
TK = Path(r"D:\cvresearch\tools\VisEvent_SOT_Benchmark")
LINES = {"thermal": 48.0, "event": 56.07, "depth": 55.0}

# method -> seed -> (thermal npz tag, event result dir, depth workspace tag)
RUNS = {
    "fedavg": {42: ("batch1_fedavg", "batch1_fedavg_event", "batch1_fedavg"),
               0: ("s0fa_thermal", "s0fa_event", "s0fa_depth"),
               123: ("s123fa_thermal", "s123fa_event", "s123fa_depth")},
    "bucket": {42: ("batch1_modality", "batch1_modality_event", "batch1_modality"),
               0: ("s0bk_thermal", "s0bk_event", "s0bk_depth"),
               123: ("s123bk_thermal", "s123bk_event", "s123bk_depth")},
    "indep": {42: ("batch2_indep_thermal", "batch2_indep_event_event", "batch2_indep_depth"),
              0: ("s0id_thermal", "s0id_event", "s0id_depth"),
              123: ("s123id_thermal", "s123id_event", "s123id_depth")},
    # fna = same three bucketed groups as S-A but pure-RGB clients DO participate (seed 42 only; isolates "exclude pure RGB")
    "fna": {42: ("batch3_bucket_fna", "batch3_bucket_fna_event", "batch3_bucket_fna"),
            0: ("s0fn_thermal", "s0fn_event", "s0fn_depth"),
            123: ("s123fn_thermal", "s123fn_event", "s123fn_depth")},
    "S-A": {42: ("sa42_thermal", "sa42_event", "sa42_depth"),
            0: ("s0sa_thermal", "s0sa_event", "s0sa_depth"),
            123: ("s123sa_thermal", "s123sa_event", "s123sa_depth")},
}
SEEDS = (42, 0, 123)


def thermal_vec(tag):
    p = EV / f"{tag}_curves.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)["curves_ovl"].mean(axis=1) * 100


_EVSEQ = None


def event_vec(tag):
    global _EVSEQ
    d = VE / tag
    if not d.is_dir():
        return None
    from visevent_eval import sequence_scores
    if _EVSEQ is None:
        _EVSEQ = [x.strip() for x in (TK / "sequence_evaluation_config" / "VisEvent_testing_subset.txt").read_text().splitlines() if x.strip()]
    try:
        s = sequence_scores(TK / "annos", d, _EVSEQ)
        return np.array([s[x][1] for x in _EVSEQ]) * 100
    except Exception:
        return None


def depth_F(tag):
    ws = VIPT / f"Depthtrack_workspace_{tag}" / "analysis"
    c = sorted(ws.glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not c:
        return None
    return float(json.loads(c[0].read_text())["results"]["rgbd-unsupervised"]["results"][0][0][2]) * 100


def pair(a, b, seed=42):
    """paired difference a-b over sequences: mean, t-CI, bootstrap CI, wins/losses"""
    if a is None or b is None or a.size != b.size:
        return None
    d = a - b
    se = d.std(ddof=1) / np.sqrt(d.size)
    tc = stats.t.ppf(0.975, d.size - 1)
    bl, bh = bootstrap_ci(d / 100.0, seed)
    return d.mean(), d.mean() - tc * se, d.mean() + tc * se, bl * 100, bh * 100, int((d > 0).sum()), int((d < 0).sum())


def fmtp(p):
    return "pending" if p is None else "%+.2f t[%+.2f,%+.2f] boot[%+.2f,%+.2f] 胜/负 %d/%d" % p


def main():
    vec = {}   # (method, seed, mod) -> per-seq vector or scalar
    for m, by in RUNS.items():
        for s, (t, e, d) in by.items():
            vec[(m, s, "thermal")] = thermal_vec(t)
            vec[(m, s, "event")] = event_vec(e)
            vec[(m, s, "depth")] = depth_F(d)

    def score(m, s, mod):
        v = vec.get((m, s, mod))
        if v is None:
            return None
        return float(v.mean()) if isinstance(v, np.ndarray) else float(v)

    out = ["# S-A 判定（冻结标准，第 120 轮，热红外为本机口径）", "",
           "线：热红外 ≥ 48.0 / 事件 ≥ 56.07 / 深度 ≥ 55.0（深度线待学长确认）", ""]
    out += ["## 逐 seed 全部数字", "", "| 方法 | seed | 热红外 AUC | 事件 SR | 深度 F |", "|---|---|---:|---:|---:|"]
    for m in RUNS:
        for s in SEEDS:
            cells = [score(m, s, k) for k in ("thermal", "event", "depth")]
            out.append("| %s | %d | %s |" % (m, s, " | ".join("pending" if c is None else "%.2f" % c for c in cells)))

    def verdict(vals):
        res = {}
        for k in LINES:
            if vals[k] is None:
                res[k] = None
            else:
                res[k] = vals[k] >= LINES[k]
        return res

    def mark(r):
        return "pending" if r is None else ("过线" if r else "**未过线**")

    out += ["", "## 第 1 步：S-A seed 42 是否三条线都过", ""]
    v42 = verdict({k: score("S-A", 42, k) for k in LINES})
    out.append("热红外 %s；事件 %s；深度 %s" % (mark(v42["thermal"]), mark(v42["event"]), mark(v42["depth"])))
    if all(x is not None for x in v42.values()):
        out.append("**结论：%s**" % ("三条线都过，继续补 seed" if all(v42.values()) else "有未过线，按冻结规则不继续补 seed"))

    out += ["", "## 第 2 步：三个 seed 的均值", ""]
    mean = {}
    for k in LINES:
        xs = [score("S-A", s, k) for s in SEEDS]
        mean[k] = None if any(x is None for x in xs) else float(np.mean(xs))
    vm = verdict(mean)
    out.append("S-A 三 seed 均值：" + "；".join("%s %s（%s）" % (k, "pending" if mean[k] is None else "%.2f" % mean[k], mark(vm[k])) for k in LINES))
    if all(x is not None for x in vm.values()):
        out.append("**结论：%s**" % ("均值三条线都过" if all(vm.values()) else "均值有未过线"))

    out += ["", "## 配对比较（逐序列，S-A 减对照；热红外必看，跨 GPU 评估噪声约 1 点）", ""]
    for mod in ("thermal", "event"):
        for ref in ("bucket", "indep", "fedavg", "fna"):
            for s in SEEDS:
                a, b = vec.get(("S-A", s, mod)), vec.get((ref, s, mod))
                out.append("- %s seed %d S-A − %s：%s" % (mod, s, ref, fmtp(pair(a, b))))
    # ---- multi-seed MEAN view (direction 2026-10-06: every judgement is on the multi-seed mean; per-seed numbers stay listed above) ----
    out += ["", "## 多 seed 均值（所有判断以此为准；n = 已有 seed 数，n<3 为暂时值）", ""]
    out += ["| 方法 | n（热红外/事件/深度） | 热红外 AUC | 事件 SR | 深度 F |", "|---|---:|---:|---:|---:|"]
    meanv = {}
    for m in RUNS:
        cells, ns = [], []
        for k in ("thermal", "event", "depth"):
            xs = [score(m, s, k) for s in SEEDS]
            xs = [x for x in xs if x is not None]
            ns.append(len(xs)); cells.append(None if not xs else float(np.mean(xs)))
            meanv[(m, k)] = (cells[-1], len(xs))
        out.append("| %s | %s | %s |" % (m, "/".join(str(n) for n in ns), " | ".join("pending" if c is None else "%.2f" % c for c in cells)))
    out += ["", "S-A 均值对判定线：" + "；".join("%s %s（线 %.2f）" % (k, "pending" if meanv[("S-A", k)][0] is None else "%.2f（n=%d，%s）" % (meanv[("S-A", k)][0], meanv[("S-A", k)][1], "过" if meanv[("S-A", k)][0] >= LINES[k] else "未过"), LINES[k]) for k in LINES)]
    out += ["", "### 均值差（S-A 减对照；只用 S-A 与对照都有的 seed；逐序列区间为多 seed 逐序列差取平均后的区间，只覆盖测试序列抽样）", ""]
    for mod in ("thermal", "event", "depth"):
        for ref in ("bucket", "indep", "fedavg", "fna"):
            diffs, vecs = [], []
            for s in SEEDS:
                a, b = score("S-A", s, mod), score(ref, s, mod)
                if a is None or b is None:
                    continue
                diffs.append(a - b)
                va, vb = vec.get(("S-A", s, mod)), vec.get((ref, s, mod))
                if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray) and va.size == vb.size:
                    vecs.append(va - vb)
            if not diffs:
                out.append("- %s S-A − %s：pending" % (mod, ref)); continue
            ci = ""
            if vecs and len(vecs) == len(diffs):
                d = np.mean(np.stack(vecs), axis=0); se = d.std(ddof=1) / np.sqrt(d.size); tc = stats.t.ppf(0.975, d.size - 1)
                ci = " t[%+.2f,%+.2f]" % (d.mean() - tc * se, d.mean() + tc * se)
            out.append("- %s S-A − %s（n=%d）：均值差 %+.2f%s；各 seed：%s" % (mod, ref, len(diffs), float(np.mean(diffs)), ci, ", ".join("%+.2f" % x for x in diffs)))
    text = "\n".join(out)
    (EV / "SA_verdict.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8"); print(text)


if __name__ == "__main__":
    main()
