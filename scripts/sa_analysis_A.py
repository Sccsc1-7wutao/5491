"""A-class read-only analyses (2026-10-06): bucket composition per method/modality, communication three-way split.
Run: D:\Conda\envs\sdstrack\python.exe D:\cvresearch\scripts\sa_analysis_A.py  -> eval_results/SA_analysis_A.json
"""
import io, json, sys
from pathlib import Path
import torch
sys.path.insert(0, r"D:\cvresearch\ViPT")
from lib.federated.param_groups import build_param_groups, AUX_SPECIFIC, RGB_SHARED, FUSION, NORM

V = Path(r"D:\cvresearch\ViPT")
split = json.load(open(V / "configs/fed/client_splits/lasher_depthtrack_visevent_seed42_v2.json"))
cl = split["clients"]
mod_of = {"lasher": "thermal", "depthtrack": "depth", "visevent": "event"}
out = {"composition": {}, "comm": {}}
for ds, mod in mod_of.items():
    cs = [c for c in cl if c["dataset"] == ds]
    sensor = [c for c in cs if c["aux_type"] != "none"]
    pure = [c for c in cs if c["aux_type"] == "none"]
    fr = lambda xs: sum(c["num_samples"] for c in xs)
    tot_all = sum(c["num_samples"] for c in cl)
    out["composition"][mod] = {
        "sensor_clients": [(c["client_id"], c["num_samples"]) for c in sensor],
        "pure_rgb_clients": [(c["client_id"], c["num_samples"]) for c in pure],
        "sensor_frames": fr(sensor), "pure_rgb_frames": fr(pure), "dataset_frames": fr(cs),
        "pure_rgb_share_of_dataset": round(fr(pure) / fr(cs), 4),
        # what trains the modality's aux_specific / fusion / norm bank under each method
        "aux_specific_bank_frames": {"fedavg(pooled, all 12)": tot_all, "independent(4 clients of this modality, all groups)": fr(cs),
                                     "bucket/fna/S-A (sensor clients only)": fr(sensor)},
        "fusion_norm_bank_frames": {"fedavg(pooled, all 12)": tot_all, "independent": fr(cs), "bucket(aux_specific only: pooled all 12)": tot_all,
                                    "fna(dataset bucket incl. pure RGB)": fr(cs), "S-A(sensor clients only)": fr(sensor)},
    }
out["composition"]["rgb_shared_pool_frames_all_methods_except_independent"] = sum(c["num_samples"] for c in cl)

# communication
sd = torch.load(V / r"output/checkpoints/train/vipt/batch1_modality/ViPTrack_ep0120.pth.tar", map_location="cpu", weights_only=False)["net"]
groups = build_param_groups(sd.keys())
cnt = {}
for k, g in groups.items():
    if g in (AUX_SPECIFIC, RGB_SHARED, FUSION, NORM):
        cnt[g] = cnt.get(g, 0) + sd[k].numel()
total = sum(cnt.values())
sensor_up = total
pure_up = total - cnt[AUX_SPECIFIC]
def ser(keys):
    b = io.BytesIO(); torch.save({k: sd[k] for k in keys}, b); return b.tell()
all_keys = [k for k, g in groups.items() if g in cnt]
pure_keys = [k for k in all_keys if groups[k] != AUX_SPECIFIC]
n_sensor = sum(1 for c in cl if c["aux_type"] != "none"); n_pure = len(cl) - n_sensor
rounds = 120
ser_all, ser_pure = ser(all_keys), ser(pure_keys)
out["comm"] = {
    "trainable_params_by_group": cnt, "trainable_params_total": total,
    "theory_bytes_fp32": {"sensor_client_upload": sensor_up * 4, "pure_rgb_client_upload": pure_up * 4, "download_full_view_per_client": total * 4},
    "serialized_bytes_torch_save": {"sensor_client_upload": ser_all, "pure_rgb_client_upload": ser_pure},
    "clients": {"sensor": n_sensor, "pure_rgb": n_pure},
    "per_round_theory_MB": round((n_sensor * sensor_up + n_pure * pure_up) * 4 / 1e6 + len(cl) * total * 4 / 1e6, 2),
    "total_120_rounds_theory_GB": round(rounds * ((n_sensor * sensor_up + n_pure * pure_up) * 4 + len(cl) * total * 4) / 1e9, 3),
    "total_120_rounds_serialized_GB_upload_plus_full_download": round(rounds * (n_sensor * ser_all + n_pure * ser_pure + len(cl) * ser_all) / 1e9, 3),
    "note": "all federated variants here upload the same groups (S-A/bucket/fna change aggregation, not what is uploaded); identical comm for FedAvg, bucket, fna, S-A",
}
Path(r"D:\cvresearch\ViPT\output\eval_results\SA_analysis_A.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8"); print(json.dumps(out, ensure_ascii=False, indent=1))
