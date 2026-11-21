"""2026-09-20: verify actual exported/dispatched weights for task 0.

Unlike file-integrity checks, this detects untouched initial aux adapters and
checks every modality's exported tensor against the saved global state.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(r"D:\cvresearch\ViPT")
sys.path.insert(0, str(REPO))
from lib.config.vipt.config import cfg, update_config_from_file
from lib.models.vipt import build_viptrack
from lib.federated.param_groups import build_param_groups, AUX_SPECIFIC, FROZEN
from lib.federated.aggregator import GlobalState


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--expect-broken", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.run_dir)
    update_config_from_file(str(REPO / "experiments/vipt/repro_rgbt.yaml"))
    torch.manual_seed(42)
    initial = build_viptrack(cfg, training=True).state_dict()
    groups = build_param_groups(initial.keys())
    weights = {}
    for aux in ("thermal", "depth", "event"):
        cp = torch.load(root / f"checkpoints/ViPTrack_{aux}_ep0020.pth.tar",
                        map_location="cpu", weights_only=False)
        assert cp["fed"] == dict(variant="fedavg", aux_type=aux, round=20, seed=42)
        weights[aux] = cp["net"]
        assert set(cp["net"]) == set(initial)
        assert all(torch.isfinite(t).all() for t in cp["net"].values())
    thermal = weights["thermal"]
    aux_keys = [k for k, g in groups.items() if g == AUX_SPECIFIC]
    frozen = [k for k, g in groups.items() if g == FROZEN]
    other = [k for k, g in groups.items() if g not in (AUX_SPECIFIC, FROZEN)]
    same = [k for k in aux_keys if torch.equal(thermal[k], initial[k])]
    unchanged_frozen = sum(torch.equal(thermal[k], initial[k]) for k in frozen)
    identical = all(torch.equal(thermal[k], weights[a][k])
                    for a in ("depth", "event") for k in thermal)
    assert len(aux_keys) == 26
    assert len(same) == (26 if args.expect_broken else 0), same
    assert identical
    assert unchanged_frozen == len(frozen)
    saved = torch.load(root / "fed_state_r0020.pt", map_location="cpu", weights_only=False)
    assert saved["round"] == 20 and saved["seed"] == 42 and saved["variant"] == "fedavg"
    state = GlobalState.from_flat_dict(saved["state"])
    assert all(torch.equal(weights[a][k], t)
               for a in weights for k, t in state.view_for(a).items())
    shared_reaches_export = all(torch.equal(thermal[k], state.shared[k]) for k in aux_keys)
    assert shared_reaches_export == (not args.expect_broken)
    record = dict(run_dir=str(root), expect_broken=args.expect_broken,
                  aux_tensors=len(aux_keys), aux_unchanged=len(same),
                  other_trainable_changed=sum(not torch.equal(thermal[k], initial[k]) for k in other),
                  frozen_unchanged=unchanged_frozen, frozen_total=len(frozen),
                  all_three_modalities_tensor_identical=identical,
                  export_matches_dispatched_view=True,
                  shared_aux_reaches_export=shared_reaches_export,
                  aux_max_abs_delta={k:float((thermal[k]-initial[k]).abs().max()) for k in aux_keys})
    Path(args.output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in record.items() if k != "aux_max_abs_delta"}, indent=2))
    print("EXPECTED OLD BUG CONFIRMED" if args.expect_broken else "OK")


if __name__ == "__main__":
    main()
