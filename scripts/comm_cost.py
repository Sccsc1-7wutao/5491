#!/usr/bin/env python
"""Reproducible communication accounting for the ViPT federated variants.

Uses the real ViPT state_dict and lib.federated.param_groups.  CPU only.
Run with D:\\Conda\\envs\\sdstrack\\python.exe scripts\\comm_cost.py.
"""
from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(r"D:\cvresearch")
VIPT = ROOT / "ViPT"
SDSTRACK = ROOT / "SDSTrack"
sys.path.insert(0, str(VIPT))

import torch
from lib.config.vipt.config import cfg, update_config_from_file
from lib.federated.fed_wrapper import FedViPTWrapper
from lib.federated.param_groups import (AUX_SPECIFIC, FUSION, NORM, RGB_SHARED,
                                        TRAINABLE_GROUPS, build_param_groups)
from lib.models.vipt.ostrack_prompt import build_viptrack
from lib.train.base_functions import update_settings

GROUPS = (AUX_SPECIFIC, RGB_SHARED, FUSION, NORM)
SENSOR_CLIENTS, RGB_ONLY_CLIENTS = 9, 3
ROUNDS = 120


def fmt_bytes(n: int) -> str:
    return f"{n:,} B ({n / 1e6:.3f} MB; {n / 2**20:.3f} MiB)"


def build_real_vipt():
    # The same model/config path used by the real-path regression test. No CUDA.
    update_config_from_file(str(VIPT / "experiments" / "vipt" / "repro_rgbt.yaml"))
    settings = SimpleNamespace(device="cpu", local_rank=-1, num_template=1,
                               num_search=1)
    update_settings(settings, cfg)
    return build_viptrack(cfg, training=True).cpu()


def count_selected(state, groups, selected):
    return sum(int(state[k].numel()) for k, g in groups.items() if g in selected)


def assert_actual_masks(model, groups):
    """Exercise FedViPTWrapper's production upload mask for both client types."""
    wrapper = FedViPTWrapper(model, groups)
    sensor = wrapper.param_mask_for("thermal", aux_present=True)
    rgb = wrapper.param_mask_for("none", aux_present=False)
    assert all(sensor[k] for k in sensor)
    assert all(rgb[k] == (groups[k] != AUX_SPECIFIC) for k in rgb)
    return {"sensor_true": [k for k, v in sensor.items() if v],
            "rgb_true": [k for k, v in rgb.items() if v]}


def try_sdstrack_count():
    """Build the local SDSTrack code and apply its actual optimizer freeze rule."""
    old_path = list(sys.path)
    old_cwd = os.getcwd()
    old_lib = {k: v for k, v in list(sys.modules.items()) if k == "lib" or k.startswith("lib.")}
    try:
        for k in old_lib:
            sys.modules.pop(k, None)
        sys.path.insert(0, str(SDSTRACK))
        os.chdir(SDSTRACK)  # its supplied YAML has a repository-relative checkpoint path
        from lib.config.sdstrack.config import cfg as s_cfg, update_config_from_file as s_update
        from lib.models.sdstrack import build_sdstrack
        s_update(str(SDSTRACK / "experiments" / "sdstrack" / "cvpr2024_rgbt.yaml"))
        net = build_sdstrack(s_cfg, training=True).cpu()
        trainable = ("Adapter", "multimodal")
        tuned = ("backbone.norm",)
        for name, p in net.named_parameters():
            p.requires_grad = (any(x in name for x in trainable) or
                               any(x in name for x in tuned))
        return sum(p.numel() for p in net.parameters() if p.requires_grad), None
    except Exception as exc:  # Report a failed local build rather than inventing a number.
        return None, repr(exc)
    finally:
        for k in [k for k in list(sys.modules) if k == "lib" or k.startswith("lib.")]:
            sys.modules.pop(k, None)
        sys.modules.update(old_lib)
        sys.path[:] = old_path
        os.chdir(old_cwd)


def assert_fedper_aggregation_mask(groups):
    """Run the FedPer server's real pre-aggregation filter on a CPU stub."""
    fedper = ROOT / "ViPT_fedper"
    old_path = list(sys.path)
    old_lib = {k: v for k, v in list(sys.modules.items()) if k == "lib" or k.startswith("lib.")}
    try:
        for k in old_lib:
            sys.modules.pop(k, None)
        sys.path.insert(0, str(fedper))
        from lib.federated.aggregator import ClientUpdate
        from lib.federated.server import FederatedServer
        trainable_groups = OrderedDict((k, g) for k, g in groups.items()
                                       if g in TRAINABLE_GROUPS)
        update = ClientUpdate("stub", 1, "thermal",
                              OrderedDict((k, torch.zeros(1)) for k in trainable_groups),
                              OrderedDict((k, True) for k in trainable_groups))
        filtered = FederatedServer._without_private_aux(
            SimpleNamespace(param_groups=groups), update)
        assert all(groups[k] != AUX_SPECIFIC for k in filtered.state)
        assert len(filtered.state) == sum(g != AUX_SPECIFIC for g in trainable_groups.values())
        return len(filtered.state)
    finally:
        for k in [k for k in list(sys.modules) if k == "lib" or k.startswith("lib.")]:
            sys.modules.pop(k, None)
        sys.modules.update(old_lib)
        sys.path[:] = old_path


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = build_real_vipt()
    state = model.state_dict()
    groups = build_param_groups(state.keys())
    masks = assert_actual_masks(model, groups)
    fedper_filtered_tensors = assert_fedper_aggregation_mask(groups)
    # state_dict also includes integer BatchNorm counters, but all tensors that
    # can be communicated by the wrapper are fp32.
    elem_bytes = {k: int(state[k].element_size()) for k, g in groups.items()
                  if g in TRAINABLE_GROUPS}
    assert set(elem_bytes.values()) == {4}, elem_bytes
    per_group = OrderedDict()
    for group in GROUPS:
        keys = [k for k, g in groups.items() if g == group]
        params = sum(int(state[k].numel()) for k in keys)
        per_group[group] = {"tensors": len(keys), "params": params, "bytes_fp32": params * 4}
    trainable = sum(v["params"] for v in per_group.values())
    full = sum(p.numel() for p in model.parameters())
    shared = count_selected(state, groups, {RGB_SHARED, FUSION, NORM})

    # This is the aggregation-side mask requested in the task. In FedPer,
    # _without_private_aux removes aux_specific before Aggregator.aggregate;
    # private aux remains client-local by the FedPer algorithm.
    variants = OrderedDict(
        fedavg={"sensor": set(GROUPS), "rgb": {RGB_SHARED, FUSION, NORM}},
        fedprox={"sensor": set(GROUPS), "rgb": {RGB_SHARED, FUSION, NORM}},
        modality={"sensor": set(GROUPS), "rgb": {RGB_SHARED, FUSION, NORM}},
        bucket_groups={"sensor": set(GROUPS), "rgb": {RGB_SHARED, FUSION, NORM}},
        fedper={"sensor": {RGB_SHARED, FUSION, NORM}, "rgb": {RGB_SHARED, FUSION, NORM}},
        local={"sensor": set(), "rgb": set()},
    )
    rows = []
    for name, masks_for_variant in variants.items():
        sensor_b = count_selected(state, groups, masks_for_variant["sensor"]) * 4
        rgb_b = count_selected(state, groups, masks_for_variant["rgb"]) * 4
        total_one_way = SENSOR_CLIENTS * sensor_b + RGB_ONLY_CLIENTS * rgb_b
        rows.append({"variant": name, "sensor_bytes": sensor_b, "rgb_bytes": rgb_b,
                     "round_upload": total_one_way, "round_download": total_one_way,
                     "total_120_bidirectional": ROUNDS * 2 * total_one_way})
    sdstrack_params, sdstrack_error = try_sdstrack_count()
    out = {"per_group": per_group, "trainable_params": trainable,
           "vipt_total_params": full, "shared_params": shared,
           "masks": {k: len(v) for k, v in masks.items()},
           "fedper_filtered_tensors": fedper_filtered_tensors, "rows": rows,
           "full_finetune_bytes_per_client_direction": full * 4,
           "full_finetune_total_120_bidirectional": ROUNDS * 12 * 2 * full * 4,
           "sdstrack_trainable_params": sdstrack_params,
           "sdstrack_error": sdstrack_error}
    result_path = VIPT / "output" / "eval_results" / "comm_cost_data.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
