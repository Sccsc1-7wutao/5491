"""Batch-1 DepthTrack paired-bootstrap and descriptive attribute analysis.

Inputs (read only): batch1 DepthTrack VOT result folders, the shared
Depthtrack_workspace sequences, and the official VOT results.json files.
Outputs: ViPT/output/eval_results/batch1_supp/depth_ci.{json,md}.
Run only with: D:\Conda\envs\vot\python.exe D:\cvresearch\scripts\batch1_depth_ci.py

This script deliberately calls vot-toolkit 0.5.3's determine_thresholds and
compute_tpr_curves; it does not implement IoU or thresholding itself.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from vot.analysis.tpr import compute_tpr_curves, determine_thresholds
from vot.dataset import load_sequence
from vot.region import RegionType, calculate_overlaps
from vot.region.io import read_trajectory

VIPT = Path(r"D:\cvresearch\ViPT")
SEQUENCES = VIPT / "Depthtrack_workspace" / "sequences"
OUT = VIPT / "output" / "eval_results" / "batch1_supp"
VARIANTS = ("fedavg", "modality", "modality_a05", "ours", "fedprox")
PAIRS = (
    ("modality", "fedavg"), ("ours", "modality"),
    ("fedprox", "fedavg"), ("modality_a05", "fedavg"),
    ("ours", "fedavg"), ("modality_a05", "modality"),
)


def result_json(variant: str) -> Path:
    kind = "fixed" if variant in {"fedavg", "modality", "modality_a05"} else "final"
    return VIPT / f"Depthtrack_workspace_batch1_{variant}" / "analysis" / f"{kind}_batch1_{variant}" / "results.json"


def load_confidence(path: Path) -> list[float]:
    # An empty first line is VOT's confidence property for the initialization.
    return [float(x) if x.strip() else 0.0 for x in path.read_text().splitlines()]


def load_variant(variant: str, names: list[str]):
    root = VIPT / f"Depthtrack_workspace_batch1_{variant}" / "results" / f"batch1_{variant}" / "rgbd-unsupervised"
    data, all_scores = {}, []
    for name in names:
        seq = load_sequence(str(SEQUENCES / name))
        folder = root / name
        trajectory = read_trajectory(str(next(folder.glob("*_001.txt"))))
        confidence = load_confidence(next(folder.glob("*_001_confidence.value")))
        if len(trajectory) != len(seq) or len(confidence) != len(seq):
            raise RuntimeError(f"{variant}/{name}: prediction, confidence, GT lengths differ")
        data[name] = (seq, trajectory, confidence)
        all_scores.extend(confidence)
    return data, determine_thresholds(all_scores, 100)


def curve(data, names, thresholds):
    prs, res = [], []
    for name in names:
        seq, trajectory, confidence = data[name]
        pr, re = compute_tpr_curves(trajectory, confidence, seq, thresholds, ignore_unknown=True, bounded=True)
        prs.append(pr); res.append(re)
    pr = np.mean(np.asarray(prs), axis=0)
    re = np.mean(np.asarray(res), axis=0)
    f = 2 * pr * re / (pr + re)
    best = int(np.argmax(f))
    return {"pr": pr, "re": re, "f": f, "best": best,
            "summary": [float(pr[best]), float(re[best]), float(f[best])]}


def tag_summary(data, names, thresholds, best: int):
    """Use VOT's overlap routine, restricted to tag=1 frames at a fixed threshold."""
    tag_files = sorted({p.name for n in names for p in (SEQUENCES / n).glob("*.tag")})
    rows = []
    for tag in tag_files:
        per_pr, per_re, n_frames = [], [], 0
        n_seq = 0
        per_sequence = {}
        for name in names:
            path = SEQUENCES / name / tag
            if not path.exists():
                continue
            flags = np.loadtxt(path, ndmin=1).astype(int)
            seq, trajectory, confidence = data[name]
            if len(flags) != len(seq):
                raise RuntimeError(f"{name}/{tag}: tag and GT lengths differ")
            chosen = flags == 1
            if not np.any(chosen):
                continue
            overlaps = np.asarray(calculate_overlaps(trajectory, seq.groundtruth(), seq.size))
            visible = np.asarray([r.type is not RegionType.SPECIAL for r in seq.groundtruth()])
            chosen_visible = chosen & visible
            if not np.any(chosen_visible):
                continue
            selected = chosen_visible & (np.asarray(confidence) >= thresholds[best])
            n_seq += 1; n_frames += int(chosen_visible.sum())
            if not np.any(selected):
                spr, sre = 1.0, 0.0
            else:
                spr = float(overlaps[selected].mean())
                sre = float(overlaps[selected].sum() / chosen_visible.sum())
            per_pr.append(spr); per_re.append(sre)
            per_sequence[name] = float(2 * spr * sre / (spr + sre)) if spr + sre else 0.0
        pr, re = float(np.mean(per_pr)), float(np.mean(per_re))
        rows.append({"attribute": tag.removesuffix(".tag"), "sequences": n_seq, "frames": n_frames,
                     "pr": pr, "re": re, "f": float(2 * pr * re / (pr + re)), "per_sequence_f": per_sequence})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in SEQUENCES.iterdir() if p.is_dir())
    if len(names) != 50:
        raise RuntimeError(f"Expected 50 DepthTrack sequences, got {len(names)}")
    all_data, curves, official = {}, {}, {}
    for variant in VARIANTS:
        all_data[variant], thresholds = load_variant(variant, names)
        curves[variant] = curve(all_data[variant], names, thresholds)
        official[variant] = json.loads(result_json(variant).read_text())["results"]["rgbd-unsupervised"]["results"][0][0]
        error = np.max(np.abs(np.asarray(curves[variant]["summary"]) - np.asarray(official[variant])))
        if error > 1e-6:
            raise RuntimeError(f"ACCEPTANCE FAILED {variant}: max abs error={error:.12g}; official={official[variant]}, recomputed={curves[variant]['summary']}")
        curves[variant]["thresholds"] = thresholds

    # The whole analysis is gated behind the five official-score acceptance checks above.
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(names), size=(20_000, len(names)))
    comparisons = []
    per_sequence, per_sequence_curves = {}, {}
    for variant in VARIANTS:
        c = curves[variant]
        seq_results = [curve(all_data[variant], [name], c["thresholds"]) for name in names]
        vals = [float(x["f"][c["best"]]) for x in seq_results]
        per_sequence[variant] = vals
        per_sequence_curves[variant] = (np.asarray([x["pr"] for x in seq_results]), np.asarray([x["re"] for x in seq_results]))
    for left, right in PAIRS:
        # At every resample each variant is recomputed from its own fixed threshold grid.
        # Curves were obtained above with VOT's function; averaging them is exactly
        # PrecisionRecallCurve.aggregate's sequence aggregation.
        lp, lr = per_sequence_curves[left]; rp, rr = per_sequence_curves[right]
        lpr, lre = lp[draws].mean(axis=1), lr[draws].mean(axis=1)
        rpr, rre = rp[draws].mean(axis=1), rr[draws].mean(axis=1)
        f_l = np.max(2 * lpr * lre / (lpr + lre), axis=1)
        f_r = np.max(2 * rpr * rre / (rpr + rre), axis=1)
        diff = f_l - f_r
        raw = np.asarray(curves[left]["summary"]) - np.asarray(curves[right]["summary"])
        comparisons.append({"comparison": f"{left} - {right}", "point_delta_f": float(raw[2]),
                            "bootstrap_ci95": [float(x) for x in np.quantile(diff, [.025, .975])],
                            "p_delta_f_gt_0": float(np.mean(diff > 0)),
                            "wins_losses_ties_at_global_optimum": [int(np.sum(np.asarray(per_sequence[left]) > np.asarray(per_sequence[right]))), int(np.sum(np.asarray(per_sequence[left]) < np.asarray(per_sequence[right]))), int(np.sum(np.asarray(per_sequence[left]) == np.asarray(per_sequence[right])))]})
    payload = {"method": {"toolkit": "vot-toolkit 0.5.3", "resolution": 100, "ignore_unknown": True, "bounded": True,
                          "bootstrap_seed": 42, "bootstrap_replicates": 20000,
                          "threshold_choice": "Each variant's full-dataset threshold grid is fixed before resampling; thresholds are not recomputed inside bootstrap draws."},
               "sequence_names": names,
               "acceptance": {v: {"official": official[v], "recomputed": curves[v]["summary"], "max_abs_error": float(np.max(np.abs(np.asarray(curves[v]["summary"]) - np.asarray(official[v]))))} for v in VARIANTS},
               "comparisons": comparisons,
               "per_sequence_f_at_variant_global_optimum": {v: dict(zip(names, per_sequence[v])) for v in VARIANTS},
               "attributes": {v: {"indoor_outdoor": {"indoor": curve(all_data[v], [n for n in names if n.endswith("_indoor")], curves[v]["thresholds"])["summary"], "wild": curve(all_data[v], [n for n in names if n.endswith("_wild")], curves[v]["thresholds"])["summary"]}, "tags": tag_summary(all_data[v], names, curves[v]["thresholds"], curves[v]["best"])} for v in VARIANTS}}
    (OUT / "depth_ci.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# DepthTrack F 配对 bootstrap 置信区间", "", "所有数值均使用 vot-toolkit 0.5.3 的 `determine_thresholds` 与 `compute_tpr_curves`。五个变体的官方复算验收均通过（最大绝对误差 ≤1e-6）。", "", "阈值网格在完整 50 序列上对每个变体各自确定，并在 bootstrap 内固定，不会随重采样重算。", "", "| 比较 | ΔF (百分点) | 95% 分位区间 (百分点) | P(ΔF>0) | 胜/负/平 |", "|---|---:|---:|---:|---:|"]
    for r in comparisons:
        lo, hi = (100*x for x in r["bootstrap_ci95"]); w,l,t = r["wins_losses_ties_at_global_optimum"]
        lines.append(f"| {r['comparison']} | {100*r['point_delta_f']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['p_delta_f_gt_0']:.4f} | {w}/{l}/{t} |")
    lines += ["", "说明：这是单 seed、基于测试序列的描述性重采样区间，不代表训练 seed 方差，也未作多重比较校正。逐标签、室内/室外结果见 `depth_ci.json`。"]
    (OUT / "depth_ci.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
