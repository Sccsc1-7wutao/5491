"""Full comparison table, ALL numbers from the local machine (thermal local caliber), per seed. Writes eval_results/full_comparison_local.html
Personalized baselines (Local/FedPer/Ditto) = mean over the 3 sensor clients (project protocol). '—' = not run / pending.
Run: D:\Conda\envs\sdstrack\python.exe D:\cvresearch\scripts\full_table_local.py
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, r"D:\cvresearch\scripts")
import judge_SA as j

EV = j.EV


def th(tag):
    v = j.thermal_vec(tag); return None if v is None else float(v.mean())


def ev(tag):
    v = j.event_vec(tag); return None if v is None else float(v.mean())


def dp(tag):
    return j.depth_F(tag)


def mean3(f, tags):
    xs = [f(t) for t in tags]
    return None if any(x is None for x in xs) else float(np.mean(xs))


def row(thermal, event, depth):
    return (thermal, event, depth)


T3 = lambda p: [f"{p}rgbt_00{i}" for i in (1, 2, 3)]
E3 = lambda p: [f"{p}rgbe_00{i}" for i in (0, 2, 3)]
D3 = lambda p: [f"{p}rgbd_00{i}" for i in (1, 2, 3)]
S42 = {  # method -> (thermal, event, depth)
    "FedAvg": (th("batch1_fedavg"), ev("batch1_fedavg_event"), dp("batch1_fedavg")),
    "FedProx μ=0.01": (th("batch1_fedprox"), ev("batch1_fedprox_event"), dp("batch1_fedprox")),
    "FedProx μ=0.1": (th("batch2_fedprox_mu01"), ev("batch2_fedprox_mu01_event"), dp("batch2_fedprox_mu01")),
    "FedPer（3 客户端均值）": (mean3(th, T3("lc_fedper_")), mean3(ev, E3("batch2_fedper_")), mean3(dp, D3("batch2_fedper_"))),
    "Ditto（3 客户端均值）": (mean3(th, T3("lc_ditto_")), mean3(ev, E3("batch4_ditto_")), mean3(dp, D3("batch4_ditto_"))),
    "Local（3 客户端均值）": (mean3(th, T3("batch2_local_")), mean3(ev, E3("batch2_local_")), mean3(dp, D3("batch2_local_"))),
    "独立联邦": (th("batch2_indep_thermal"), ev("batch2_indep_event_event"), dp("batch2_indep_depth")),
    "分桶（aux_specific）": (th("batch1_modality"), ev("batch1_modality_event"), dp("batch1_modality")),
    "分桶 + warm-up（ours）": (th("batch1_ours"), ev("batch1_ours_event"), dp("batch1_ours")),
    "fna": (th("batch3_bucket_fna"), ev("batch3_bucket_fna_event"), dp("batch3_bucket_fna")),
    "S-A": (th("sa42_thermal"), ev("sa42_event"), dp("sa42_depth")),
}
MULTI = {  # method -> seed -> tags
    "FedAvg": {0: ("s0fa", ), 123: ("s123fa", )}, "分桶": {0: ("s0bk", ), 123: ("s123bk", )}, "独立联邦": {0: ("s0id", ), 123: ("s123id", )},
    "fna": {0: ("s0fn", ), 123: ("s123fn", )}, "S-A": {0: ("s0sa", ), 123: ("s123sa", )},
}


def fmt(x):
    return "—" if x is None else f"{x:.2f}"


h = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>完整对比总表</title><style>",
     ":root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--mute:#5d6878;--line:#e2e6ec;--hl:#e4f5ea}@media(prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1a2029;--ink:#e6eaf0;--mute:#9aa6b6;--line:#2c3542;--hl:#17301f}}",
     "body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}main{max-width:900px;margin:0 auto;padding:18px 16px 50px}",
     "h1{font-size:1.5rem}h2{font-size:1.15rem;border-left:4px solid #2b5fd9;padding-left:10px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0;overflow-x:auto}",
     "table{width:100%;border-collapse:collapse;font-size:.93rem}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--mute);font-size:.85rem}tr.hl td{background:var(--hl);font-weight:600}.s{color:var(--mute);font-size:.88rem}",
     "</style></head><body><main><h1>完整对比总表（全部本机口径）</h1>",
     f"<p class='s'>生成于 {time.strftime('%Y-%m-%d %H:%M')}。热红外 AUC / 事件 SR / 深度 F，第 120 轮。个性化方法（Local、FedPer、Ditto）为 3 个传感器客户端的均值。“—”表示没有跑或还在评。热红外全部在本机 4080S 评估，没有混入 B 实例的数字。</p>",
     "<div class='card'><h2>seed 42（原划分）全部方法</h2><table><tr><th>方法</th><th>热红外</th><th>事件</th><th>深度</th></tr>"]
for k, (a, b, c) in S42.items():
    cls = " class='hl'" if k == "S-A" else ""
    h.append(f"<tr{cls}><td>{k}</td><td>{fmt(a)}</td><td>{fmt(b)}</td><td>{fmt(c)}</td></tr>")
h.append("</table></div>")
h.append("<div class='card'><h2>逐 seed：FedAvg / 分桶 / 独立联邦 / fna / S-A</h2><table><tr><th>方法</th><th>seed</th><th>热红外</th><th>事件</th><th>深度</th></tr>")
base42 = {"FedAvg": S42["FedAvg"], "分桶": S42["分桶（aux_specific）"], "独立联邦": S42["独立联邦"], "fna": S42["fna"], "S-A": S42["S-A"]}
allv = {}
for m in MULTI:
    for seed in (42, 0, 123):
        if seed == 42:
            vals = base42[m]
        else:
            p = MULTI[m][seed][0]
            vals = (th(f"{p}_thermal"), ev(f"{p}_event"), dp(f"{p}_depth"))
        allv[(m, seed)] = vals
        cls = " class='hl'" if m == "S-A" else ""
        h.append(f"<tr{cls}><td>{m}</td><td>{seed}</td><td>{fmt(vals[0])}</td><td>{fmt(vals[1])}</td><td>{fmt(vals[2])}</td></tr>")
h.append("<tr><td colspan='5' class='s'>各方法三个 seed 的均值（仅当三个 seed 都有数时才算）</td></tr>")
for m in MULTI:
    cols = []
    for i in range(3):
        xs = [allv[(m, s)][i] for s in (42, 0, 123)]
        cols.append(None if any(x is None for x in xs) else float(np.mean(xs)))
    h.append(f"<tr><td>{m}（均值）</td><td>42/0/123</td><td>{fmt(cols[0])}</td><td>{fmt(cols[1])}</td><td>{fmt(cols[2])}</td></tr>")
h.append("</table></div>")
h.append("<p class='s'>说明：fna 的 seed 0、123 和 S-A 的 seed 0、123 仍在跑或评，评完会补上。数据来源：judge_SA.py 与 full_table_local.py（读取 eval_results 与各工作区的评估输出）。</p></main></body></html>")
Path(r"D:\cvresearch\ViPT\output\eval_results\full_comparison_local.html").write_text("\n".join(h), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
for k, v in S42.items(): print(k, [fmt(x) for x in v])
for k, v in allv.items(): print(k, [fmt(x) for x in v])
