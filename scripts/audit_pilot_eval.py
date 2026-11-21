"""2026-09-20: audit task-1 predictions before using the unchanged official scorer.

Run with --variant modality/fedavg after each evaluation; --compare after both.
Assertions prevent partial, mixed-checkpoint or misaligned results entering reports.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\cvresearch")
REPO = ROOT / "ViPT"
OUT = REPO / "output/eval_results"
TOOLKIT = ROOT / "tools/LasHeR_Toolkit"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(variant):
    label = "pilot_" + variant
    names = (TOOLKIT / "sequence_evaluation_config/all_dataset_lashertest.txt").read_text().split()
    assert len(names) == len(set(names)) == 245
    results = REPO / "RGBT_workspace/results/LasHeR" / label
    files = list(results.glob("*.txt"))
    assert {p.stem for p in files} == set(names)
    frames = 0
    for name in names:
        pred = np.loadtxt(results / (name + ".txt"), ndmin=2)
        gt = np.loadtxt(TOOLKIT / "annos" / (name + ".txt"), delimiter=",", ndmin=2)
        assert pred.shape == gt.shape and pred.shape[1] == 4, (name, pred.shape, gt.shape)
        assert np.isfinite(pred).all(), name
        frames += len(pred)
    log = (REPO / "output" / ("eval_" + label + ".log")).read_text(encoding="utf-8-sig")
    assert not re.search(r"(?m)^-1\s|Traceback|RuntimeError|FILE NOT FOUND", log)
    paths = re.findall(r"\[params\] checkpoint -> ([^\r\n]+)", log)
    expected = REPO / "output/checkpoints/train/vipt" / label / "ViPTrack_ep0020.pth.tar"
    assert len(paths) == 245, len(paths)
    assert all(Path(p.strip()).resolve() == expected.resolve() for p in paths)
    assert len(re.findall(r" , fps:", log)) == 245
    elapsed = re.findall(r"Totally cost ([\d.]+) seconds!", log)
    assert len(elapsed) == 1, "Missing normal completion marker"
    source = REPO / "output/fed/pilot20" / variant / "checkpoints/ViPTrack_thermal_ep0020.pth.tar"
    sha = digest(expected)
    assert sha == digest(source), "Checkpoint differs from thermal export"
    cfg = (REPO / "experiments/vipt" / (label + ".yaml")).read_bytes()
    assert cfg.isascii() and cfg == (REPO / "experiments/vipt/repro_rgbt.yaml").read_bytes()
    record = dict(variant=variant, sequences=245, frames=frames, checkpoint=str(expected),
                  sha256=sha, seconds=float(elapsed[0]), skipped=0, empty=0,
                  checkpoint_loads=len(paths), finite_predictions=True)
    command = [sys.executable, str(ROOT / "scripts/lasher_eval.py"), "--results", str(results),
               "--save-npz", str(OUT / (label + "_curves.npz"))]
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    assert "scored            : 245 / 245" in proc.stdout
    assert "MISSING result" not in proc.stdout and "FAILED to score" not in proc.stdout
    (OUT / (label + "_scores.log")).write_text(proc.stdout + proc.stderr, encoding="utf-8")
    (OUT / (label + "_audit.json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(proc.stdout)


def compare():
    a = np.load(OUT / "pilot_fedavg_curves.npz", allow_pickle=False)
    b = np.load(OUT / "pilot_modality_curves.npz", allow_pickle=False)
    assert list(a["names"]) == list(b["names"]) and len(a["names"]) == 245
    comparison = {}
    for tag, key, idx in (("AUC", "curves_ovl", None), ("PR", "curves_err", 20)):
        xa = a[key].mean(axis=1) if idx is None else a[key][:, idx]
        xb = b[key].mean(axis=1) if idx is None else b[key][:, idx]
        assert np.isfinite(xa).all() and np.isfinite(xb).all()
        d = xb - xa
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        comparison[tag] = dict(fedavg=float(xa.mean()), modality=float(xb.mean()),
                               delta=float(d.mean()), se=se, t=float(d.mean()/se) if se else None,
                               wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()),
                               historical_floor_pp=1.0 if tag == "AUC" else 1.4)
    (OUT / "pilot_paired_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["modality", "fedavg"])
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.variant:
        audit(args.variant)
    if args.compare:
        compare()
