#!/usr/bin/env python
"""CPU-only diagnostic for modality-bank drift in saved federated states.

Reads only saved ``fed_state_r*.pt`` files.  It never starts training or
evaluation.  The generated CSV/PNGs/Markdown are descriptive, single-seed
artifacts, not causal evidence.
"""
from __future__ import annotations

import argparse, csv, json, os, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
VIPT = ROOT / "ViPT"
sys.path.insert(0, str(VIPT))
from lib.federated.aggregator import GlobalState  # noqa: E402
from lib.config.vipt.config import cfg, update_config_from_file  # noqa: E402
from lib.models.vipt import build_viptrack  # noqa: E402

MODES = ("thermal", "depth", "event")
GROUPS = ("fusion", "norm", "aux_specific")

def group_of(key: str) -> str | None:
    key = key.removeprefix("module.")
    if ".conv1x1." in key and ".prompt_blocks." in key: return "fusion"
    if ".prompt_norms." in key: return "norm"
    if ".patch_embed_prompt." in key or ".conv0_1." in key: return "aux_specific"
    return None

def load(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return GlobalState.from_flat_dict(blob["state"]), int(blob.get("round", int(path.stem[-4:])))

def vector(bank, group):
    values = [v.detach().reshape(-1).to(torch.float64) for k, v in bank.items()
              if group_of(k) == group]
    if not values: raise KeyError(f"no {group} tensors")
    return torch.cat(values)

def state_paths(directory: Path): return sorted(directory.glob("fed_state_r*.pt"))

def weights(split_path: Path):
    doc = json.loads(split_path.read_text(encoding="utf-8"))
    out = defaultdict(int)
    dataset_to_mode = {"lasher":"thermal", "depthtrack":"depth", "visevent":"event"}
    for c in doc["clients"]:
        mode = c.get("aux_type")
        if mode == "none": mode = dataset_to_mode[{"rgbt":"lasher","rgbd":"depthtrack","rgbe":"visevent"}[c["client_id"][:4]]]
        out[mode] += int(c["num_samples"])
    return out

def analyse(name, paths, modal, init):
    rows=[]
    for p in paths:
        state, rnd = load(p)
        for group in GROUPS:
            if not all(m in state.per_modality for m in MODES):
                if name == "main": continue
                raise KeyError(f"{p}: missing modality banks")
            # In the main modality run fusion/norm are shared, therefore the
            # three views are exactly the same shared tensor (distance zero).
            vs={m:vector(state.per_modality[m] if any(group_of(k)==group for k in state.per_modality[m]) else state.shared, group) for m in MODES}
            mean=sum(vs[m] * (modal[m] / sum(modal.values())) for m in MODES)
            for a,b in combinations(MODES,2): rows.append(dict(run=name,round=rnd,group=group,metric=f"{a}-{b}",value=float(torch.linalg.vector_norm(vs[a]-vs[b]))))
            for m in MODES:
                rows.append(dict(run=name,round=rnd,group=group,metric=f"{m}-weighted_mean",value=float(torch.linalg.vector_norm(vs[m]-mean))))
                initial=init.get(group, {}).get(m)
                if initial is not None: rows.append(dict(run=name,round=rnd,group=group,metric=f"{m}-initial",value=float(torch.linalg.vector_norm(vs[m]-initial))))
    return rows

def initial_from_seed42():
    """Rebuild exactly as fed_state_to_ckpt.py does, rather than infer drift
    from a later saved state.  The seed is fixed by the experiment contract."""
    old=os.getcwd(); os.chdir(VIPT)
    try:
        update_config_from_file(str(VIPT/"experiments/vipt/repro_rgbt.yaml"))
        torch.manual_seed(42)
        reference=build_viptrack(cfg, training=True).state_dict()
    finally: os.chdir(old)
    return {g:{m:vector(reference,g) for m in MODES} for g in GROUPS}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=VIPT/"output/eval_results/fusion_drift_diag")
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    diag=VIPT/"output/fed/diag"; main_dir=VIPT/"output/fed/batch1/modality"
    # r120 was historically saved one directory higher.
    main_paths=list(state_paths(main_dir)); legacy=VIPT/"output/fed/batch1/fed_state_modality_r0120.pt"
    if legacy.exists(): main_paths.append(legacy)
    if not main_paths: raise FileNotFoundError(main_dir)
    # Materialize the legacy r120 in a temporary list-free analysis directory is avoided:
    # analyse accepts a directory, so main coverage is reported from r20-r100; r120 is
    # appended through an equivalent temporary path list below.
    split=VIPT/"configs/fed/client_splits/lasher_depthtrack_visevent_seed42_v2.json"; modal=weights(split)
    all_rows=[]
    seed_initial=initial_from_seed42()
    for name,d in (("fna",diag/"batch3_bucket_fna"),("fa",diag/"batch3_bucket_fa"),("main",main_dir)):
        paths = main_paths if name == "main" else state_paths(d)
        all_rows += analyse(name,paths,modal,seed_initial)
    with (args.out/"distances.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["run","round","group","metric","value"]); w.writeheader(); w.writerows(all_rows)
    for group in GROUPS:
        fig,ax=plt.subplots(figsize=(9,5))
        for run in sorted({r["run"] for r in all_rows}):
            for metric in ("thermal-depth","thermal-event","depth-event"):
                pts=sorted((r["round"],r["value"]) for r in all_rows if r["run"]==run and r["group"]==group and r["metric"]==metric)
                if pts: ax.plot(*zip(*pts),marker="o",label=f"{run}: {metric}")
        ax.set(title=f"{group}: modality-bank pairwise L2",xlabel="saved round",ylabel="L2 norm"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.out/f"{group}_pairwise.png",dpi=160); plt.close(fig)
    summary=[]
    for run in ("fna","fa","main"):
        for group in GROUPS:
            pts=[r for r in all_rows if r["run"]==run and r["group"]==group and r["metric"]=="thermal-depth"]
            if pts: summary.append((run,group,max(r["round"] for r in pts),next(r["value"] for r in pts if r["round"]==max(x["round"] for x in pts))))
    (args.out/"summary.json").write_text(json.dumps({"modal_frame_weights":modal,"final_thermal_depth":summary},indent=2),encoding="utf-8")
    print(args.out)
if __name__=="__main__": main()
