"""Which modality bank does FedAvg's single shared aux adapter resemble? (r120, CPU only)"""
import sys
import torch

sys.path.insert(0, r"D:\cvresearch\ViPT")
from lib.federated.param_groups import build_param_groups  # noqa: E402

B = r"D:\cvresearch\ViPT\output\checkpoints\train\vipt"
def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)["net"]

F = load(B + r"\batch1_fedavg\ViPTrack_ep0120.pth.tar")
T = load(B + r"\batch1_modality\ViPTrack_ep0120.pth.tar")
D = load(B + r"\batch1_modality_depth\ViPTrack_ep0120.pth.tar")
E = load(B + r"\batch1_modality_event\ViPTrack_ep0120.pth.tar")
groups = build_param_groups(F.keys())
for grp in ("aux_specific", "rgb_shared", "fusion", "norm"):
    keys = [k for k, g in groups.items() if g == grp]
    vec = lambda sd: torch.cat([sd[k].flatten().float() for k in keys])
    f, t, d, e = vec(F), vec(T), vec(D), vec(E)
    # shares of aux-present frames in the pooled data (CLAUDE.md audit table)
    p = 0.503 * t + 0.429 * d + 0.068 * e
    n = lambda x: x.norm().item()
    print(f"[{grp}] {len(keys)} tensors, {f.numel()} params")
    print(f"   |F-T|={n(f-t):.4f} |F-D|={n(f-d):.4f} |F-E|={n(f-e):.4f} |F-pooledTDE|={n(f-p):.4f}")
    print(f"   |T-D|={n(t-d):.4f} |T-E|={n(t-e):.4f} |D-E|={n(d-e):.4f}")
