"""Export an official ViPT checkpoint from a saved federated state.

Example:
    D:\\Conda\\envs\\sdstrack\\python.exe scripts\\fed_state_to_ckpt.py ^
      --fed-state ViPT\\output\\fed\\long120\\fedavg\\fed_state_r0120.pt ^
      --aux-type thermal --output ViPT\\output\\fed\\long120\\fedavg\\checkpoints\\from_state_thermal_ep0120.pth.tar

The saved federated state only contains trainable prompt tensors.  This script
rebuilds the frozen ViPT reference model, then overlays the requested modality
view exactly as FederatedServer.export_checkpoint does.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import torch


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ViPT"))
sys.path.insert(0, REPO)

from lib.config.vipt.config import cfg, update_config_from_file  # noqa: E402
from lib.federated.aggregator import GlobalState  # noqa: E402
from lib.models.vipt import build_viptrack  # noqa: E402


def export_from_state(fed_state_path, aux_type, output_path, config="repro_rgbt"):
    """Write a full `{'net': state_dict}` checkpoint from one saved state."""
    blob = torch.load(fed_state_path, map_location="cpu", weights_only=False)
    if "state" not in blob:
        raise KeyError("fed-state has no global 'state' payload: %s" % fed_state_path)
    if "local_states" in blob:
        raise ValueError("local fed-state has per-client models; select one client explicitly")

    os.chdir(REPO)
    update_config_from_file(os.path.join("experiments", "vipt", config + ".yaml"))
    torch.manual_seed(int(blob.get("seed", 42)))
    reference = build_viptrack(cfg, training=True).state_dict()
    view = GlobalState.from_flat_dict(blob["state"]).view_for(aux_type)
    missing = [key for key in view if key not in reference]
    if missing:
        raise KeyError("federated keys absent from ViPT: %s" % missing[:5])

    full = OrderedDict((key, value.detach().clone())
                       for key, value in reference.items())
    for key, value in view.items():
        full[key] = value.detach().clone().to(full[key].dtype)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({"net": full, "epoch": int(blob.get("round", 0)),
                "fed": {"variant": blob.get("variant"), "aux_type": aux_type,
                        "round": int(blob.get("round", 0)),
                        "seed": int(blob.get("seed", 42)),
                        "source_state": os.path.abspath(fed_state_path)}},
               output_path)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fed-state", required=True)
    parser.add_argument("--aux-type", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="repro_rgbt")
    args = parser.parse_args(argv)
    path = export_from_state(args.fed_state, args.aux_type, args.output, args.config)
    print("checkpoint -> %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
