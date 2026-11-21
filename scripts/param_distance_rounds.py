#!/usr/bin/env python
"""Round-wise aux_specific parameter distances for batch-1 (CPU only)."""
from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(r"D:\cvresearch")
REPO = ROOT / "ViPT"
FED = REPO / "output" / "fed" / "batch1"
OUT = REPO / "output" / "eval_results"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "figure_assets"))

from lib.config.vipt.config import cfg, update_config_from_file
from lib.federated.aggregator import GlobalState
from lib.federated.param_groups import build_param_groups
from lib.models.vipt import build_viptrack
from figstyle import MODALITY_COLOR, new_fig

ROUNDS = (20, 40, 60, 80, 100, 120)
WEIGHTS = {"thermal": 238827, "depth": 203709, "event": 32072}
EXPECTED_R120 = {"f_to_thermal": 10.70, "f_to_depth": 13.28,
                 "f_to_event": 16.39, "f_to_weighted": 7.96}


def paths(variant: str, rnd: int) -> Path:
    # The five downloaded intermediates are in a variant subdirectory; r120
    # was already present at batch1 root when this analysis was requested.
    if rnd == 120:
        name = "fed_state_fedavg_r0120.pt" if variant == "fedavg" else "fed_state_modality_r0120.pt"
        return FED / name
    return FED / variant / f"fed_state_r{rnd:04d}.pt"


def load_state(path: Path) -> GlobalState:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if int(blob["round"]) not in ROUNDS:
        raise ValueError(f"unexpected round in {path}: {blob['round']}")
    return GlobalState.from_flat_dict(blob["state"])


def vector(mapping, keys):
    return torch.cat([mapping[k].detach().reshape(-1).float() for k in keys])


def initial_vector(keys):
    # Matches fed_state_to_ckpt.py: change to ViPT so the experiment's relative
    # pretrained path resolves, load config, seed torch=42, and build on CPU.
    old_cwd = os.getcwd()
    try:
        os.chdir(REPO)
        update_config_from_file(str(REPO / "experiments" / "vipt" / "repro_rgbt.yaml"))
        torch.manual_seed(42)
        reference = build_viptrack(cfg, training=True).cpu().state_dict()
    finally:
        os.chdir(old_cwd)
    return vector(reference, keys)


def metrics(fedavg: GlobalState, modality: GlobalState, keys, initial):
    f = vector(fedavg.view_for("thermal"), keys)  # identical for all views in FedAvg
    t = vector(modality.view_for("thermal"), keys)
    d = vector(modality.view_for("depth"), keys)
    e = vector(modality.view_for("event"), keys)
    total = sum(WEIGHTS.values())
    pooled = sum(WEIGHTS[n] / total * x for n, x in
                 (("thermal", t), ("depth", d), ("event", e)))
    norm = lambda x: float(torch.linalg.vector_norm(x).item())
    return {
        "f_to_thermal": norm(f - t), "f_to_depth": norm(f - d),
        "f_to_event": norm(f - e), "f_to_weighted": norm(f - pooled),
        "thermal_to_depth": norm(t - d), "thermal_to_event": norm(t - e),
        "depth_to_event": norm(d - e),
        "fedavg_from_initial": norm(f - initial),
        "thermal_from_initial": norm(t - initial),
        "depth_from_initial": norm(d - initial), "event_from_initial": norm(e - initial),
    }


def make_figure(rows):
    rounds = [r["round"] for r in rows]
    fig, axes = new_fig("double", 5.8, nrows=1, ncols=3, sharex=True)
    for ax in axes:
        ax.set_xticks(rounds)
        ax.set_xlabel("Round")
        ax.set_ylabel("L2 distance")
    specs = [
        (axes[0], [("f_to_thermal", "FedAvg to RGB-T bucket", MODALITY_COLOR["T"]),
                   ("f_to_depth", "FedAvg to RGB-D bucket", MODALITY_COLOR["D"]),
                   ("f_to_event", "FedAvg to RGB-E bucket", MODALITY_COLOR["E"]),
                   ("f_to_weighted", "FedAvg to weighted mean", "#7F7F7F")],
         "FedAvg relative to buckets"),
        (axes[1], [("thermal_to_depth", "RGB-T to RGB-D", "#A45A84"),
                   ("thermal_to_event", "RGB-T to RGB-E", "#7F7F7F"),
                   ("depth_to_event", "RGB-D to RGB-E", "#8E7A2A")],
         "Between modality buckets"),
        (axes[2], [("fedavg_from_initial", "FedAvg", "#7F7F7F"),
                   ("thermal_from_initial", "RGB-T bucket", MODALITY_COLOR["T"]),
                   ("depth_from_initial", "RGB-D bucket", MODALITY_COLOR["D"]),
                   ("event_from_initial", "RGB-E bucket", MODALITY_COLOR["E"])],
         "Displacement from initialization"),
    ]
    for ax, series, title in specs:
        for key, label, color in series:
            ax.plot(rounds, [r[key] for r in rows], marker="o", color=color, label=label)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=6.4)
    fig.savefig(OUT / "param_distance_rounds.pdf")
    fig.savefig(OUT / "param_distance_rounds.png", dpi=600)
    plt.close(fig)


def main():
    required = [paths(v, r) for v in ("fedavg", "modality") for r in ROUNDS]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing fed states:\n" + "\n".join(missing))
    # Use the saved model keys themselves for the grouping definition.
    sample = torch.load(paths("fedavg", 120), map_location="cpu", weights_only=False)
    flat = sample["state"]
    state_keys = [k for k in flat if not k.startswith("@")]
    groups = build_param_groups(state_keys)
    aux_keys = [k for k, group in groups.items() if group == "aux_specific"]
    if len(aux_keys) != 26:
        raise AssertionError(f"expected 26 aux_specific tensors, got {len(aux_keys)}")
    initial = initial_vector(aux_keys)

    # Mandatory regression gate before calculating/reporting the trajectory.
    gate = metrics(load_state(paths("fedavg", 120)), load_state(paths("modality", 120)), aux_keys, initial)
    bad = {k: (gate[k], expected) for k, expected in EXPECTED_R120.items()
           if round(gate[k], 2) != expected}
    if bad:
        raise AssertionError(f"r120 regression did not reproduce: {bad}")

    rows = []
    for rnd in ROUNDS:
        row = {"round": rnd}
        row.update(metrics(load_state(paths("fedavg", rnd)), load_state(paths("modality", rnd)), aux_keys, initial))
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"definition": "concatenated fp32 aux_specific tensors; Euclidean L2", "aux_tensors": len(aux_keys),
               "aux_params": int(initial.numel()), "weights_frames": WEIGHTS,
               "r120_regression": {k: gate[k] for k in EXPECTED_R120}, "rows": rows}
    (OUT / "param_distance_rounds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_figure(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
