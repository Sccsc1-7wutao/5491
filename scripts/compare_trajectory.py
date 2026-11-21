"""Paired LasHeR trajectory comparison for FedAvg versus modality-aware FL."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\cvresearch\ViPT\output\eval_results")


def curves_path(variant: str, round_idx: int) -> Path:
    if round_idx == 20:
        return ROOT / f"pilot_{variant}_curves.npz"
    if round_idx == 120:
        return ROOT / f"long_{variant}_curves.npz"
    return ROOT / f"traj_{variant}_r{round_idx}_curves.npz"


def metric(blob, name: str):
    return blob["curves_ovl"].mean(axis=1) if name == "AUC" else blob["curves_err"][:, 20]


def main():
    for round_idx in (20, 40, 80, 120):
        fedavg = np.load(curves_path("fedavg", round_idx), allow_pickle=True)
        modality = np.load(curves_path("modality", round_idx), allow_pickle=True)
        if list(fedavg["names"]) != list(modality["names"]):
            raise ValueError(f"r{round_idx}: sequence order mismatch")
        print(f"r{round_idx}  sequences={len(fedavg['names'])}")
        for name in ("AUC", "PR"):
            x, y = metric(fedavg, name), metric(modality, name)
            delta = y - x
            se = delta.std(ddof=1) / math.sqrt(len(delta))
            threshold = 1.96 * se
            t_value = delta.mean() / se
            verdict = "discernible" if abs(t_value) >= 2 else "indistinguishable"
            print(
                f"  {name}: fedavg={100*x.mean():.2f} modality={100*y.mean():.2f} "
                f"diff={100*delta.mean():+.2f} pp threshold={100*threshold:.2f} pp "
                f"t={t_value:+.2f} {verdict}"
            )


if __name__ == "__main__":
    main()
