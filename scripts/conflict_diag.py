#!/usr/bin/env python
"""Layer-wise cross-modal update-conflict diagnostic for federated ViPT.

This lives outside ``ViPT`` deliberately.  It reuses the production client
training path, but never changes it.  A run stores local deltas first, so the
analysis can be re-run without loading data or a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch


def _repo() -> str:
    root = os.environ.get("CVR_ROOT", r"D:\cvresearch")
    return os.path.join(root, "ViPT")


REPO = _repo()
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.federated.aggregator import GlobalState, NO_AUX  # noqa: E402
from lib.federated.client import FederatedClient  # noqa: E402
from lib.federated.client_data import load_split  # noqa: E402
from lib.federated.param_groups import TRAINABLE_GROUPS, build_param_groups  # noqa: E402
from lib.federated.server import DEFAULT_SPLIT, build_everything  # noqa: E402
from lib.federated.warmup import WarmupScheduler  # noqa: E402


LAYER_RE = re.compile(r"(?:prompt_blocks|prompt_norms)\.(\d+)\.")


def unit_for_key(key: str, group: str) -> str:
    """Return the pre-registered ``layer × parameter group`` analysis unit."""
    if "patch_embed_prompt." in key:
        return "patch_embed_prompt/%s" % group
    match = LAYER_RE.search(key)
    if not match:
        raise ValueError("cannot derive prompt layer from %r" % key)
    return "layer_%02d/%s" % (int(match.group(1)), group)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a, b) / denom) if float(denom) else float("nan")


def cancellation_ratio(items):
    """||weighted mean(delta)|| / weighted mean(||delta||)."""
    total = float(sum(w for w, _ in items))
    if total <= 0:
        return float("nan")
    mean = sum((d.float() * (w / total) for w, d in items))
    norm_mean = sum(w * float(torch.linalg.vector_norm(d.float())) for w, d in items) / total
    return float(torch.linalg.vector_norm(mean) / norm_mean) if norm_mean else float("nan")


def pair_type(aux_a: str, aux_b: str) -> str:
    if aux_a == NO_AUX and aux_b == NO_AUX:
        return "rgb-rgb"
    if aux_a == NO_AUX or aux_b == NO_AUX:
        return "sensor-rgb"
    if aux_a == aux_b:
        return "same-%s" % aux_a
    return "cross-" + "-".join(sorted((aux_a, aux_b)))


def _flat_unit(delta, keys):
    return torch.cat([delta[k].detach().cpu().reshape(-1) for k in keys])


def _save_delta(path, update, dispatched, seed, round_idx):
    valid = OrderedDict((k, v.detach().cpu() - dispatched[k].detach().cpu())
                        for k, v in update.state.items()
                        if update.param_mask is None or update.param_mask.get(k, True))
    payload = {"client_id": update.client_id, "aux_type": update.aux_type,
               "num_samples": update.num_samples, "seed": seed,
               "round": round_idx, "delta": valid}
    torch.save(payload, path)
    return payload


def run_local_deltas(args):
    checkpoint = torch.load(args.fed_state, map_location="cpu", weights_only=False)
    if "state" not in checkpoint:
        raise ValueError("checkpoint lacks state: %s" % args.fed_state)
    round_idx = int(checkpoint.get("round", checkpoint.get("round_idx", _round_from_name(args.fed_state))))
    state = GlobalState.from_flat_dict(checkpoint["state"])
    net, clients, _cfg, _settings = build_everything(args)
    plan = WarmupScheduler.no_warmup(120, modality_aware=True).resolve(round_idx)
    os.makedirs(args.out, exist_ok=True)
    wanted_noise = set(_split_csv(args.noise_clients))
    results = []
    for client in clients:
        for base_seed in (42, 43) if client.client_id in wanted_noise else (42,):
            # A new client instance prevents optimizer/data-loader state from leaking
            # between the two stochastic repetitions.
            clone = FederatedClient({"client_id": client.client_id, "dataset": client.dataset_name,
                "aux_type": client.aux_type, "num_samples": client.num_samples,
                "missing_prob": client.missing_prob, "sequence_ids": client.sequence_ids},
                client.cfg, client.settings, base_seed=base_seed)
            dispatched = state.view_for(clone.aux_type)
            update, stats = clone.local_train(net, dispatched, plan, round_idx,
                                               local_steps=args.local_steps, device=args.device)
            seed = int(stats.get("seed", 0))
            output = os.path.join(args.out, "delta_%s_s%d.pt" % (clone.client_id, seed))
            results.append(_save_delta(output, update, dispatched, seed, round_idx))
            print("%s seed=%d tensors=%d" % (clone.client_id, seed, len(results[-1]["delta"])))
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"fed_state": os.path.abspath(args.fed_state), "round": round_idx,
                   "local_steps": args.local_steps, "clients": [c.client_id for c in clients],
                   "noise_clients": sorted(wanted_noise), "plan": plan.as_dict()}, f,
                  ensure_ascii=False, indent=2)
    return results


def analyze(delta_files, out_csv=None):
    docs = [torch.load(p, map_location="cpu", weights_only=False) for p in delta_files]
    primary = {d["client_id"]: d for d in docs if d["seed"] == _expected_seed(42, d)}
    repeats = {d["client_id"]: d for d in docs if d["seed"] == _expected_seed(43, d)}
    if len(primary) < 2:
        raise ValueError("need at least two seed-42 delta files")
    # Do not infer the universe from an arbitrary client: a pure-RGB file has
    # no aux_specific tensors.  The union is the protocol's full upload schema.
    all_keys = sorted({k for d in primary.values() for k in d["delta"]})
    groups = build_param_groups(all_keys)
    units = defaultdict(list)
    for key, group in groups.items():
        if group not in TRAINABLE_GROUPS:
            raise ValueError("unexpected non-trainable delta key %s" % key)
        units[unit_for_key(key, group)].append(key)
    rows = []
    for unit, keys in sorted(units.items()):
        present = {cid: _flat_unit(d["delta"], [k for k in keys if k in d["delta"]])
                   for cid, d in primary.items() if any(k in d["delta"] for k in keys)}
        cosines = defaultdict(list)
        ids = sorted(present)
        for i, cid_a in enumerate(ids):
            for cid_b in ids[i + 1:]:
                common = [k for k in keys if k in primary[cid_a]["delta"] and k in primary[cid_b]["delta"]]
                if common:
                    cosines[pair_type(primary[cid_a]["aux_type"], primary[cid_b]["aux_type"])].append(
                        cosine(_flat_unit(primary[cid_a]["delta"], common), _flat_unit(primary[cid_b]["delta"], common)))
        same = [v for n, vs in cosines.items() if n.startswith("same-") for v in vs]
        cross = [v for n, vs in cosines.items() if n.startswith("cross-") for v in vs]
        noise = [cosine(_flat_unit(primary[cid]["delta"], common), _flat_unit(repeats[cid]["delta"], common))
                 for cid in present if cid in repeats
                 for common in [[k for k in keys if k in primary[cid]["delta"] and k in repeats[cid]["delta"]]] if common]
        for label, values in cosines.items():
            rows.append(_row(unit, label, values, same, cross, noise, present, primary))
        # Include cancellation even for a unit without enough pairs.
        if not cosines:
            rows.append(_row(unit, "none", [], same, cross, noise, present, primary))
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def _row(unit, label, values, same, cross, noise, present, docs):
    weighted = [(docs[cid]["num_samples"], vec) for cid, vec in present.items()]
    by_mod = defaultdict(list)
    for cid, vec in present.items(): by_mod[docs[cid]["aux_type"]].append((docs[cid]["num_samples"], vec))
    return {"unit": unit, "pair_type": label, "cosine_mean": _mean(values), "pair_count": len(values),
            "same_modality_mean": _mean(same), "cross_modality_mean": _mean(cross),
            "conflict_score": _mean(same) - _mean(cross) if same and cross else float("nan"),
            "noise_ceiling_mean": _mean(noise), "cancellation_all": cancellation_ratio(weighted),
            "cancellation_by_modality": json.dumps({k: cancellation_ratio(v) for k, v in sorted(by_mod.items())})}


def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def _split_csv(s): return [x for x in s.split(",") if x]
def _round_from_name(path):
    m = re.search(r"r(\d+)", os.path.basename(path)); return int(m.group(1)) if m else 0
def _expected_seed(base, doc):
    from lib.federated.client import derive_seed
    return derive_seed(base, doc["round"], doc["client_id"])


def self_test(args):
    assert cosine(torch.tensor([1., 0]), torch.tensor([0., 1])) == 0.0
    assert abs(cancellation_ratio([(1, torch.tensor([1., 0])), (1, torch.tensor([-1., 0]))])) < 1e-8
    assert abs(cancellation_ratio([(1, torch.tensor([1., 0])), (1, torch.tensor([1., 0]))]) - 1) < 1e-8
    assert unit_for_key("backbone.prompt_blocks.3.conv1x1.weight", "fusion") == "layer_03/fusion"
    assert unit_for_key("backbone.patch_embed_prompt.weight", "aux_specific") == "patch_embed_prompt/aux_specific"
    # Actual model inventory guards against a parser silently omitting a new key.
    net, _clients, _cfg, _settings = build_everything(args)
    groups = build_param_groups(net.state_dict().keys())
    trainable = [k for k, g in groups.items() if g in TRAINABLE_GROUPS]
    assigned = [unit_for_key(k, groups[k]) for k in trainable]
    assert len(trainable) == len(assigned) == len(set(trainable))
    assert len(trainable) == 110, "expected 110 trainable tensors, got %d" % len(trainable)
    # Regression: the first file is pure RGB and lacks aux_specific, but the
    # union-based analysis must retain that unit from sensor-client files.
    with tempfile.TemporaryDirectory() as td:
        shared = "backbone.prompt_blocks.0.conv1x1.weight"
        aux = "backbone.patch_embed_prompt.weight"
        toy = [("rgbt_000", NO_AUX, {shared: torch.tensor([1.])}),
               ("rgbt_002", "thermal", {shared: torch.tensor([1.]), aux: torch.tensor([1.])}),
               ("rgbd_002", "depth", {shared: torch.tensor([1.]), aux: torch.tensor([-1.])})]
        paths = []
        for cid, modality, delta in toy:
            seed = _expected_seed(42, {"round": 20, "client_id": cid})
            path = os.path.join(td, cid + ".pt")
            torch.save({"client_id": cid, "aux_type": modality, "num_samples": 1,
                        "seed": seed, "round": 20, "delta": delta}, path); paths.append(path)
        assert any(r["unit"].endswith("/aux_specific") for r in analyze(paths)), "aux unit lost when first file is RGB-only"
    print("self-test passed: %d trainable tensors assigned exactly once" % len(trainable))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fed-state")
    p.add_argument("--local-steps", type=int, default=50)
    p.add_argument("--clients", default="")
    p.add_argument("--out", required=False, default="conflict_diag_out")
    p.add_argument("--noise-clients", default="rgbt_002,rgbd_002,rgbe_002,rgbt_000")
    p.add_argument("--split", default=DEFAULT_SPLIT); p.add_argument("--config", default="repro_rgbt")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--workers", type=int, default=0)
    # build_everything shares the server's construction contract and expects
    # these two server-only fields even though this diagnostic never uses FedProx.
    p.add_argument("--variant", default="modality"); p.add_argument("--prox-mu", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true"); p.add_argument("--analyze", nargs="*", default=[])
    args = p.parse_args(argv)
    if args.self_test: self_test(args); return 0
    if args.analyze:
        analyze(args.analyze, os.path.join(args.out, "conflict_units.csv")); return 0
    if not args.fed_state: p.error("--fed-state is required unless --self-test or --analyze is used")
    run_local_deltas(args)
    files = sorted(str(p) for p in Path(args.out).glob("delta_*.pt"))
    analyze(files, os.path.join(args.out, "conflict_units.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
