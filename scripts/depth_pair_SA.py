"""DepthTrack paired bootstrap for the S-A comparison (reuses scripts/batch1_depth_ci.py machinery, same as depth_pair.py).
Every run is acceptance-checked: recomputed F summary must equal the official vot analysis (err <= 1e-6).
Per seed, for every available run: F value, and paired bootstrap of S-A minus {bucket, indep, fna, fedavg}, plus bucket minus fedavg.
Bootstrap = 20,000 resamples of the 50 sequences (seed 42 draws); it covers test-sequence sampling ONLY (not training seed / partition).
Run: D:\Conda\envs\vot\python.exe D:\cvresearch\scripts\depth_pair_SA.py   -> eval_results/depth_pairs_SA.md
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import batch1_depth_ci as b1

names = sorted(p.name for p in b1.SEQUENCES.iterdir() if p.is_dir())
assert len(names) == 50
RUNS = {
    42: {"fedavg": "batch1_fedavg", "bucket": "batch1_modality", "indep": "batch2_indep_depth", "fna": "batch3_bucket_fna", "S-A": "sa42_depth"},
    0: {"fedavg": "s0fa_depth", "bucket": "s0bk_depth", "indep": "s0id_depth", "fna": "s0fn_depth", "S-A": "s0sa_depth"},
    123: {"fedavg": "s123fa_depth", "bucket": "s123bk_depth", "indep": "s123id_depth", "fna": "s123fn_depth", "S-A": "s123sa_depth"},
}


def load_run(run):
    ws = b1.VIPT / f"Depthtrack_workspace_{run}"
    cands = sorted((ws / "analysis").rglob("results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    root = ws / "results" / run / "rgbd-unsupervised"
    if not root.is_dir():
        return None
    data, scores = {}, []
    for n in names:
        seq = b1.load_sequence(str(b1.SEQUENCES / n)); f = root / n
        tr = b1.read_trajectory(str(next(f.glob("*_001.txt"))))
        cf = b1.load_confidence(next(f.glob("*_001_confidence.value")))
        assert len(tr) == len(seq) == len(cf), n
        data[n] = (seq, tr, cf); scores += cf
    th = b1.determine_thresholds(scores, 100)
    off = json.loads(cands[0].read_text())["results"]["rgbd-unsupervised"]["results"][0][0]
    c = b1.curve(data, names, th)
    err = float(np.max(np.abs(np.asarray(c["summary"]) - np.asarray(off))))
    assert err <= 1e-6, (run, err)
    sr = [b1.curve(data, [n], th) for n in names]
    return {"c": c, "pr": np.asarray([x["pr"] for x in sr]), "re": np.asarray([x["re"] for x in sr]), "err": err}


draws = np.random.default_rng(42).integers(0, 50, size=(20_000, 50))


def fscore(r):
    p, q = r["pr"], r["re"]
    return np.max(2 * p[draws].mean(1) * q[draws].mean(1) / (p[draws].mean(1) + q[draws].mean(1)), axis=1)


def boot(a, b):  # b - a
    diff = fscore(b) - fscore(a)
    pt = b["c"]["summary"][2] - a["c"]["summary"][2]
    lo, hi = np.quantile(diff, [.025, .975])
    return pt * 100, lo * 100, hi * 100, float(np.mean(diff > 0))


lines = ["# DepthTrack 配对比较（S-A 相关；官方 F，对 50 条序列自助重采样，仅覆盖测试序列抽样）", ""]
for seed, runs in RUNS.items():
    loaded = {}
    for k, tag in runs.items():
        try:
            r = load_run(tag)
        except Exception as e:  # missing / failed run -> pending
            r = None
        if r is not None:
            loaded[k] = r
    lines.append(f"## seed {seed}")
    lines.append("| 方法 | F | 验收误差 |"); lines.append("|---|---:|---:|")
    for k in runs:
        if k in loaded:
            lines.append(f"| {k} | {loaded[k]['c']['summary'][2]*100:.2f} | {loaded[k]['err']:.1e} |")
        else:
            lines.append(f"| {k} | pending | |")
    lines.append("")
    lines.append("| 对比（差=前者−后者） | 差 | 95% 区间 | P(差>0) |"); lines.append("|---|---:|---|---:|")
    for x, y in (("S-A", "bucket"), ("S-A", "indep"), ("S-A", "fna"), ("S-A", "fedavg"), ("bucket", "fedavg"), ("fna", "bucket")):
        if x in loaded and y in loaded:
            pt, lo, hi, pp = boot(loaded[y], loaded[x])
            lines.append(f"| {x} − {y} | {pt:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {pp:.3f} |")
        else:
            lines.append(f"| {x} − {y} | pending | | |")
    lines.append("")
out = Path(r"D:\cvresearch\ViPT\output\eval_results\depth_pairs_SA.md")
out.write_text("\n".join(lines), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8"); print("\n".join(lines))
