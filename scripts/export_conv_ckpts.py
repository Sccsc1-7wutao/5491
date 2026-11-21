"""Export thermal/depth checkpoints for the convergence curves (r20/r60/r100 of the batch-1 FedAvg and modality(bucket) runs).

Regression first: the r120 state exported by the same function must equal the existing batch-1 final checkpoint tensor-by-tensor.
Outputs: ViPT\\output\\checkpoints\\train\\vipt\\conv_<v>_r<NNNN>_<thermal|depth>\\ViPTrack_ep0120.pth.tar
Run: D:\\Conda\\envs\\sdstrack\\python.exe D:\\cvresearch\\scripts\\export_conv_ckpts.py
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from fed_state_to_ckpt import export_from_state  # noqa: E402

FED = r"D:\cvresearch\ViPT\output\fed\batch1"
CK = r"D:\cvresearch\ViPT\output\checkpoints\train\vipt"


def same(a, b):
    na, nb = torch.load(a, map_location="cpu", weights_only=False)["net"], torch.load(b, map_location="cpu", weights_only=False)["net"]
    if na.keys() != nb.keys():
        return False, "key sets differ"
    bad = [k for k in na if not torch.equal(na[k], nb[k])]
    return (not bad), (bad[:3] if bad else "identical")


def main():
    # regression on the final round (thermal for both methods, depth for the bucket run if a reference exists)
    for v, m, ref in (("fedavg", "thermal", os.path.join(CK, "batch1_fedavg", "ViPTrack_ep0120.pth.tar")),
                      ("modality", "thermal", os.path.join(CK, "batch1_modality", "ViPTrack_ep0120.pth.tar"))):
        if not os.path.isfile(ref):
            print("regression SKIPPED (no reference checkpoint):", v, m, ref); continue
        tmp = os.path.join(tempfile.gettempdir(), f"reg_{v}_{m}.pth.tar")
        export_from_state(os.path.join(FED, f"fed_state_{v}_r0120.pt"), m, tmp)
        ok, info = same(tmp, ref)
        print(f"regression {v}/{m}: {'PASS' if ok else 'FAIL'} ({info})")
        if not ok:
            raise SystemExit("export procedure does not reproduce the batch-1 checkpoint; stop")
        os.remove(tmp)
    done = 0
    for v in ("fedavg", "modality"):
        for rd in ("0020", "0060", "0100"):
            st = os.path.join(FED, v, f"fed_state_r{rd}.pt")
            for m in ("thermal", "depth"):
                out = os.path.join(CK, f"conv_{v}_r{rd}_{m}", "ViPTrack_ep0120.pth.tar")
                if not os.path.isfile(out):
                    export_from_state(st, m, out)
                done += 1
                print("ok", out)
    print("exported", done)


if __name__ == "__main__":
    main()
