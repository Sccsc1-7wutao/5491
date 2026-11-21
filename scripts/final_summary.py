"""Final multi-seed summary page (for the report to Klint). Everything is computed from the evaluation outputs; rerun after each new result.
Judgement rule (frozen, direction 2026-10-06): all judgements use the multi-seed MEAN over seeds 42/0/123; per-seed numbers are always listed.
Run: D:\\Conda\\envs\\sdstrack\\python.exe D:\\cvresearch\\scripts\\final_summary.py  -> eval_results/final_summary_for_klint.html
"""
import io
import sys
import time
import contextlib
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, r"D:\cvresearch\scripts")
import judge_SA as j

import os
with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):          # full_table_local prints a lot and refreshes full_comparison_local.html
    import full_table_local as ft

SEEDS = (42, 0, 123)
MODS = ("thermal", "event", "depth")
MODNAME = {"thermal": "热红外 AUC", "event": "事件 SR", "depth": "深度 F"}
METHODS = ["fedavg", "bucket", "indep", "fna", "S-A"]
LABEL = {"fedavg": "FedAvg", "bucket": "分桶（aux_specific）", "indep": "独立联邦", "fna": "fna（三组分桶，纯 RGB 参与）", "S-A": "S-A（三组分桶，排除纯 RGB）"}

vec = {}
for m, by in j.RUNS.items():
    for s, (t, e, d) in by.items():
        vec[(m, s, "thermal")] = j.thermal_vec(t)
        vec[(m, s, "event")] = j.event_vec(e)
        vec[(m, s, "depth")] = j.depth_F(d)


def score(m, s, k):
    v = vec.get((m, s, k))
    if v is None:
        return None
    return float(v.mean()) if isinstance(v, np.ndarray) else float(v)


def mean_of(m, k):
    xs = [score(m, s, k) for s in SEEDS]
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)) if xs else None), len(xs)


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def mean_diff(ref, k):
    """S-A minus ref over the seeds both have; returns (mean diff, n, per-seed list, t-CI or None)"""
    diffs, vecs = [], []
    for s in SEEDS:
        a, b = score("S-A", s, k), score(ref, s, k)
        if a is None or b is None:
            continue
        diffs.append((s, a - b))
        va, vb = vec.get(("S-A", s, k)), vec.get((ref, s, k))
        if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray) and va.size == vb.size:
            vecs.append(va - vb)
    if not diffs:
        return None
    ci = None
    if vecs and len(vecs) == len(diffs):
        d = np.mean(np.stack(vecs), axis=0)
        se = d.std(ddof=1) / np.sqrt(d.size)
        tc = stats.t.ppf(0.975, d.size - 1)
        ci = (d.mean() - tc * se, d.mean() + tc * se)
    return float(np.mean([x for _, x in diffs])), len(diffs), diffs, ci


# ------------------------------------------------------------------ verdict
sa_means = {k: mean_of("S-A", k) for k in MODS}
complete = all(sa_means[k][1] == 3 for k in MODS)
passes = {k: (None if sa_means[k][0] is None else sa_means[k][0] >= j.LINES[k]) for k in MODS}
if complete:
    verdict = "三个 seed 都已评完。S-A 三 seed 均值：" + "；".join(
        f"{MODNAME[k]} {sa_means[k][0]:.2f}（线 {j.LINES[k]:.2f}，{'过' if passes[k] else '<b>未过</b>'}）" for k in MODS) + "。"
    verdict += " <b>三条线全部满足。</b>" if all(passes.values()) else " <b>有未满足的线，如实报告。</b>"
else:
    got = max(sa_means[k][1] for k in MODS)
    verdict = f"<b>最终判定待定：S-A 目前只有 {got} 个 seed，下面的均值是暂时值。</b>" + "暂时均值：" + "；".join(
        f"{MODNAME[k]} {fmt(sa_means[k][0])}（n={sa_means[k][1]}）" for k in MODS) + "。"

# ------------------------------------------------------------------ html
css = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--mute:#5d6878;--line:#e2e6ec;--accent:#2b5fd9;--ok:#14803c;--okbg:#e4f5ea;--warn:#a15c00;--warnbg:#fff2d9;--bad:#b3261e;--badbg:#fde7e5;--bluebg:#e6edfd}
@media(prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1a2029;--ink:#e6eaf0;--mute:#9aa6b6;--line:#2c3542;--accent:#7aa2ff;--ok:#5ed38a;--okbg:#17301f;--warn:#f0b04a;--warnbg:#33290f;--bad:#ff8a80;--badbg:#38191a;--bluebg:#1b2740}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:920px;margin:0 auto;padding:18px 16px 60px}h1{font-size:1.55rem;margin:8px 0 4px}h2{font-size:1.15rem;border-left:4px solid var(--accent);padding-left:10px;margin:0 0 10px}
h3{font-size:1rem;margin:14px 0 6px}.meta{color:var(--mute);font-size:.9rem;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0;overflow-x:auto}.lead{border-left:6px solid var(--accent);background:var(--bluebg)}
table{width:100%;border-collapse:collapse;font-size:.92rem;margin:6px 0}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;white-space:normal}th{color:var(--mute);font-size:.85rem}
tr.hl td{background:var(--okbg);font-weight:600}.tag{display:inline-block;padding:1px 9px;border-radius:99px;font-size:.78rem;font-weight:700;margin-right:4px}
.ok{background:var(--okbg);color:var(--ok)}.part{background:var(--warnbg);color:var(--warn)}.no{background:var(--badbg);color:var(--bad)}.wait{background:var(--bluebg);color:var(--accent)}
ul,ol{padding-left:1.25rem;margin:6px 0}li{margin:5px 0}.small{color:var(--mute);font-size:.88rem}code{background:var(--bg);padding:1px 6px;border-radius:5px;font-size:.86em}
"""
h = [f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>多 seed 结果汇总</title><style>{css}</style></head><body><main>",
     "<h1>S-A 多 seed 结果汇总（原划分）</h1>",
     f"<div class='meta'>生成于 {time.strftime('%Y-%m-%d %H:%M')} · 全部第 120 轮 · 热红外全部本机口径 · <b>所有判断以三个 seed（42 / 0 / 123）的均值为准，逐 seed 数字照列</b> · 原划分与任何新协议分开报告</div>",
     f"<section class='card lead'><h2>一句话</h2><p>{verdict}</p><p class='small'>冻结判定线：热红外 AUC ≥ 48.0；事件 SR ≥ 56.07；深度 F ≥ 55.0（深度线待学长确认）。</p></section>"]

# 0. glossary (what each method is, in plain words)
h.append("""<section class='card'><h2>先说清楚：各方法是什么</h2>
<p class='small'>设定：联邦学习，12 个客户端（热红外 LasHeR、事件 VisEvent、深度 DepthTrack 各 4 个），骨干是 ViPT，只训练提示参数（共 839,628 个）。每个数据集里有 1 个<b>纯 RGB 客户端</b>（没有传感器数据，只有 RGB），其余是<b>传感器客户端</b>。提示参数分四组：<code>aux_specific</code>（664,416，和传感器模态强相关）、<code>rgb_shared</code>（73,836）、<code>fusion</code>（82,944）、<code>norm</code>（18,432）。</p>
<table><tr><th>方法</th><th>一句话解释</th></tr>
<tr><td>FedAvg</td><td>所有客户端、所有参数一起平均，不区分模态。</td></tr>
<tr><td>分桶（aux_specific）</td><td>只把 <code>aux_specific</code> 按模态分开聚合（每个模态一个桶，只在该模态的传感器客户端之间平均），其余三组仍然全部客户端一起平均。我们之前的“主方法”。</td></tr>
<tr><td>独立联邦</td><td>三个模态各自单独做一个联邦，每个只用自己数据集的 4 个客户端（含那个纯 RGB），互不共享。</td></tr>
<tr><td><b>fna</b></td><td><b>fna = 把 <u>f</u>usion、<u>n</u>orm、<u>a</u>ux_specific 三组都按数据集分桶</b>（名字取自这三个组的首字母）。每个数据集一个桶，<b>纯 RGB 客户端也在自己数据集的桶里一起平均</b>。这是在分桶基础上多分了 fusion 和 norm 两组。</td></tr>
<tr><td><b>S-A</b>（selective participation）</td><td><b>和 fna 完全一样，只多一条规则：纯 RGB 客户端不参与 fusion 和 norm 这两组的分桶聚合</b>（它们仍然接收自己数据集的桶，仍然参与 rgb_shared，aux_specific 不变）。</td></tr></table>
<p><b>所以 S-A 与 fna 只差一件事：纯 RGB 客户端是否参与 fusion/norm 的分桶聚合。</b>S-A−fna 的差，就是这条“排除规则”的单独效果。目前 fna 只有 seed 42，seed 0 和 123 正在评估。</p></section>""")

# 1. verdict table
h.append("<section class='card'><h2>1. 对冻结判定线（三 seed 均值）</h2><table><tr><th>指标</th><th>S-A 均值</th><th>n</th><th>判定线</th><th>结果</th><th>逐 seed（42 / 0 / 123）</th></tr>")
for k in MODS:
    mv, n = sa_means[k]
    if mv is None:
        res = "<span class='tag wait'>待定</span>"
    elif n < 3:
        res = f"<span class='tag wait'>暂时：{'过' if mv >= j.LINES[k] else '未过'}</span>"
    else:
        res = "<span class='tag ok'>过</span>" if mv >= j.LINES[k] else "<span class='tag no'>未过</span>"
    per = " / ".join(fmt(score("S-A", s, k)) for s in SEEDS)
    h.append(f"<tr><td>{MODNAME[k]}</td><td>{fmt(mv)}</td><td>{n}</td><td>{j.LINES[k]:.2f}</td><td>{res}</td><td>{per}</td></tr>")
h.append("</table><p class='small'>单个 seed 低于线不算失败，判据是均值；但逐 seed 数字全部列出，不挑。</p></section>")

# 2. means table
h.append("<section class='card'><h2>2. 各方法三 seed 均值</h2><table><tr><th>方法</th><th>n</th><th>热红外</th><th>事件</th><th>深度</th></tr>")
for m in METHODS:
    cells, ns = [], []
    for k in MODS:
        mv, n = mean_of(m, k); cells.append(mv); ns.append(n)
    cls = " class='hl'" if m == "S-A" else ""
    h.append(f"<tr{cls}><td>{LABEL[m]}</td><td>{'/'.join(map(str, ns))}</td><td>{fmt(cells[0])}</td><td>{fmt(cells[1])}</td><td>{fmt(cells[2])}</td></tr>")
h.append("</table><p class='small'>n 不是 3 的行为暂时值（分别对应热红外/事件/深度已有的 seed 数）。</p>")
h.append("<h3>逐 seed 全部数字</h3><table><tr><th>方法</th><th>seed</th><th>热红外</th><th>事件</th><th>深度</th></tr>")
for m in METHODS:
    for s in SEEDS:
        cls = " class='hl'" if m == "S-A" else ""
        h.append(f"<tr{cls}><td>{LABEL[m]}</td><td>{s}</td><td>{fmt(score(m, s, 'thermal'))}</td><td>{fmt(score(m, s, 'event'))}</td><td>{fmt(score(m, s, 'depth'))}</td></tr>")
h.append("</table></section>")

# 3. mean differences
h.append("<section class='card'><h2>3. S-A 相对各对照的均值差</h2><p class='small'>差 = S-A − 对照，只用 S-A 与对照都有的 seed。区间为多 seed 逐序列差取平均后的 95% t 区间，<b>只覆盖测试序列的抽样，不覆盖训练 seed 与客户端划分的波动</b>；深度是官方 F 值，无逐序列区间（单 seed 的深度自助区间见 <code>depth_pairs_SA.md</code>）。</p>")
h.append("<table><tr><th>指标</th><th>对照</th><th>n</th><th>均值差</th><th>95% 区间</th><th>各 seed 的差（42 / 0 / 123）</th></tr>")
for k in MODS:
    for ref in ("bucket", "indep", "fedavg", "fna"):
        r = mean_diff(ref, k)
        if r is None:
            h.append(f"<tr><td>{MODNAME[k]}</td><td>{LABEL[ref]}</td><td>0</td><td>—</td><td>—</td><td>待出</td></tr>"); continue
        md, n, diffs, ci = r
        dd = {s: x for s, x in diffs}
        per = " / ".join("—" if s not in dd else f"{dd[s]:+.2f}" for s in SEEDS)
        cis = "—" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
        sig = ""
        if ci is not None:
            sig = " <span class='tag ok'>区间不跨 0</span>" if (ci[0] > 0 or ci[1] < 0) else ""
        h.append(f"<tr><td>{MODNAME[k]}</td><td>{LABEL[ref]}</td><td>{n}</td><td>{md:+.2f}{sig}</td><td>{cis}</td><td>{per}</td></tr>")
h.append("</table></section>")

# 4. other methods (seed 42 only)
h.append("<section class='card'><h2>4. 其他对比方法（只有 seed 42）</h2><table><tr><th>方法</th><th>热红外</th><th>事件</th><th>深度</th></tr>")
for name in ("FedProx μ=0.01", "FedProx μ=0.1", "FedPer（3 客户端均值）", "Ditto（3 客户端均值）", "Local（3 客户端均值）", "分桶 + warm-up（ours）"):
    a, b, c = ft.S42[name]
    h.append(f"<tr><td>{name}</td><td>{fmt(a)}</td><td>{fmt(b)}</td><td>{fmt(c)}</td></tr>")
h.append("</table><p class='small'>这些方法没有补多 seed，不参与均值判定；个性化方法为 3 个传感器客户端的均值。FedPer、Ditto 的热红外已在本机重评，与其余数字同口径。</p></section>")

# 5. claims
h.append("""<section class='card'><h2>5. 论文说法：成立 / 不成立 / 待定</h2>
<p class='small'>下面是已核实部分的口径；依赖尚未评完数据的说法在生成后需要按第 3 节的数字人工复核。</p>
<ul>
<li><span class='tag ok'>成立</span>分桶（aux_specific）优于 FedAvg：事件上每个 seed 都 +3 以上，深度和热红外方向一致。</li>
<li><span class='tag ok'>成立</span>S-A 优于 FedAvg：热红外、事件逐 seed 区间不跨 0。</li>
<li><span class='tag part'>线索</span>纯 RGB 参与三组分桶（fna）在热红外、事件上有害，排除后恢复（S-A−fna，seed 42：热红外 +5.09、事件 +2.51、深度 −0.17）。跨 seed 是否稳定，看第 3 节 fna 行。</li>
<li><span class='tag part'>看第 3 节</span>S-A 对分桶、对独立联邦：以三 seed 均值与区间为准。此前两个 seed 显示热红外略高，事件、深度持平。</li>
<li><span class='tag no'>不成立</span>通信量少 17.6 倍是我们的贡献（来自 ViPT 只训提示参数）。</li>
<li><span class='tag no'>不成立</span>缺失模态更鲁棒（置零评估不支持；S-A、fna 未测）。</li>
<li><span class='tag no'>不成立</span>纯 RGB 客户端普遍有害；首创按模态聚合（有先例）。</li>
<li><span class='tag wait'>待定</span>换一份划分后是否成立：划分只有一份，未做另一份划分的验证。</li>
</ul></section>""")

# 6. setup & caveats
h.append("""<section class='card'><h2>6. 设置、口径与局限</h2>
<ul>
<li><b>划分：</b><code>lasher_depthtrack_visevent_seed42_v2.json</code>（12 个客户端，每个数据集 4 个，其中纯 RGB 各 1 个；纯 RGB 占各数据集帧数：热红外 48%、事件 85%、深度 3%）。<b>只有这一份划分。</b></li>
<li><b>训练：</b>120 轮，每轮 50 步，12 个客户端全部参与；seed 为 42、0、123；代码 ViPT_bucketgroups 提交 e47644b（lib 与 configs 去行尾指纹 <code>6ccd0d9ce9d8d779</code>），FedAvg 用主仓库 b5bfa29。</li>
<li><b>评估：</b>全部在本机 4080S，第 120 轮，不挑检查点；热红外为本机口径；训练在云端 4090，不同机器的训练非确定性计入 seed 波动，未单独校正。</li>
<li><b>独立联邦：</b>每个模态只用自己的 4 个客户端（含该数据集的纯 RGB 客户端）。</li>
<li><b>统计：</b>逐序列区间不覆盖训练 seed 与划分的波动；每个方法只有 3 个 seed，均值的稳定性有限。</li>
<li>深度判定线 55.0 是我方提议的容忍线，学长尚未确认；深度对 seed 敏感（独立联邦 54.02 / 56.71 / 55.12）。</li>
</ul></section>""")

h.append("""<section class='card'><h2>7. 需要学长确认的</h2><ol>
<li>深度判定线 55.0 是否可以。</li>
<li>论文定位：以三 seed 均值为准，S-A 与分桶、独立联邦的关系怎么表述。</li>
<li>是否做另一份随机划分的验证（冻结规则草案 <code>partition2_freeze_rules_draft.html</code>，云端机器已释放，要做需重新租机）。</li>
<li>名字：FedMPT 与 CVPR 2026 那篇同名，正文用 Fed-MAPTrack。</li></ol></section>""")
h.append("</main></body></html>")
out = Path(r"D:\cvresearch\ViPT\output\eval_results\final_summary_for_klint.html")
out.write_text("\n".join(h), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print("written", out)
print(verdict.replace("<b>", "").replace("</b>", ""))
