"""Build a standalone HTML report from the four-variant pilot run.

Self-contained: inline SVG chart, no CDN, sendable over WeChat as a file.

Design rule for this report: it must read honestly whichever way the numbers
land. "No detectable difference between the variants" is a real and important
result for the experiment-matrix decision, not a failure to be buried -- so the
verdict text is derived from the measured effect size and its standard error,
never assumed.
"""

import argparse
import html as _h
import json
import os
from collections import OrderedDict, defaultdict

VARIANTS = ["local", "fedavg", "modality", "ours"]
LABEL = {"local": "Local（不聚合）", "fedavg": "FedAvg", "modality": "Modality-aware",
         "ours": "Ours（+ warm-up）"}
COLOR = {"local": "#8a8a90", "fedavg": "#c4562a", "modality": "#2f6fb5",
         "ours": "#2e7d52"}


# ------------------------------------------------------------------ loading

def load(root, variant):
    path = os.path.join(root, variant, "round_metrics.jsonl")
    if not os.path.isfile(path):
        return None, None
    per_client, per_round = {}, OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            per_round[rec["round"]] = rec
            for c in rec.get("clients", []):
                per_client[(rec["round"], c["client_id"])] = c
    return per_client, per_round


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = [x for x in xs if x is not None and x == x]
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def common_rounds(runs):
    """Rounds every variant has actually completed.

    A paired comparison must stand on the same rounds for all variants;
    using the global round list would silently give a still-running variant an
    empty window and produce NaN.
    """
    sets = [{r for (r, _) in pc} for (pc, _) in runs.values()]
    return sorted(set.intersection(*sets)) if sets else []


def per_client_window(per_client, rounds, window):
    last = set(sorted(rounds)[-window:])
    grp = defaultdict(list)
    aux = {}
    for (r, cid), rec in per_client.items():
        if r in last:
            grp[cid].append(rec)
            aux[cid] = rec.get("aux_type", "?")
    return ({cid: {"IoU": mean(v.get("mean_IoU") for v in recs),
                   "loss": mean(v.get("mean_Loss/total") for v in recs),
                   "aux_type": aux[cid]}
             for cid, recs in grp.items()})


def paired(base, other, key="IoU"):
    cids = sorted(set(base) & set(other))
    d = [other[c][key] - base[c][key] for c in cids]
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float("nan"), se=float("nan"), t=float("nan"),
                    win=0, lose=0)
    m = mean(d)
    se = stdev(d) / (n ** 0.5)
    return dict(n=n, mean=m, se=se,
                t=(m / se if se and se > 0 else float("nan")),
                win=sum(1 for x in d if x > 0), lose=sum(1 for x in d if x < 0))


# -------------------------------------------------------------------- chart

def svg_chart(runs, rounds):
    W, H = 840, 330
    ml, mr, mt, mb = 56, 128, 24, 42
    pw, ph = W - ml - mr, H - mt - mb

    series = OrderedDict()
    for v, (pc, _) in runs.items():
        ys = []
        for r in rounds:
            ys.append(mean(rec.get("mean_IoU") for (rr, _), rec in pc.items() if rr == r))
        series[v] = ys

    vals = [y for ys in series.values() for y in ys if y == y]
    if not vals:
        return "<p>没有可画的数据。</p>"
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.15, 0.004)
    lo, hi = lo - pad, hi + pad
    x0, x1 = min(rounds), max(rounds)

    def px(r):
        return ml + (r - x0) / max(x1 - x0, 1) * pw

    def py(v):
        return mt + (hi - v) / (hi - lo) * ph

    p = ['<svg viewBox="0 0 %d %d" width="100%%" xmlns="http://www.w3.org/2000/svg" '
         'font-family="system-ui,sans-serif" font-size="11">' % (W, H),
         '<rect width="%d" height="%d" fill="#fff"/>' % (W, H)]
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = py(v)
        p.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ececef"/>'
                 % (ml, y, ml + pw, y))
        p.append('<text x="%d" y="%.1f" text-anchor="end" fill="#666">%.3f</text>'
                 % (ml - 8, y + 4, v))
    step = max(1, (x1 - x0) // 10)
    for r in range(x0, x1 + 1, step):
        p.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#666">%d</text>'
                 % (px(r), mt + ph + 18, r))
    p.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#444">轮次</text>'
             % (ml + pw / 2, H - 6))
    p.append('<text x="14" y="%d" fill="#444" transform="rotate(-90 14 %d)" '
             'text-anchor="middle">本地 IoU</text>' % (mt + ph / 2, mt + ph / 2))

    for i, (v, ys) in enumerate(series.items()):
        pts = [(px(r), py(y)) for r, y in zip(rounds, ys) if y == y]
        if not pts:
            continue
        d = " ".join(("M" if k == 0 else "L") + "%.1f %.1f" % xy
                     for k, xy in enumerate(pts))
        p.append('<path d="%s" fill="none" stroke="%s" stroke-width="1.9"/>'
                 % (d, COLOR[v]))
        ly = mt + 14 + i * 18
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2.4"/>'
                 % (ml + pw + 12, ly, ml + pw + 32, ly, COLOR[v]))
        p.append('<text x="%d" y="%d" fill="#333">%s</text>'
                 % (ml + pw + 38, ly + 4, _h.escape(LABEL[v])))
    p.append('</svg>')
    return "".join(p)


# ------------------------------------------------------------------- report

CSS = """
:root{--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e6;--bg:#fff;--soft:#f7f7f9;
 --acc:#2f6fb5;--ok:#2e7d52;--warn:#b45309;--bad:#b3261e;
 --mono:ui-monospace,"Cascadia Code",Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:#f4f4f6;color:var(--fg);
 font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
main{max-width:920px;margin:0 auto;background:var(--bg);padding:48px 46px 72px;
 box-shadow:0 1px 4px rgba(0,0,0,.08)}
h1{font-size:27px;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin:0 0 26px}
h2{font-size:20px;margin:42px 0 12px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px}
p{margin:12px 0}
code{background:var(--soft);padding:1px 5px;border-radius:4px;font:13px var(--mono)}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#fafafb;font-weight:600}
tr:nth-child(even) td{background:#fcfcfd}
td.num{text-align:right;font-family:var(--mono)}
.callout{border-left:3px solid var(--acc);background:var(--soft);padding:12px 16px;
 margin:18px 0;border-radius:0 6px 6px 0}
.callout.bad{border-color:var(--bad);background:#fdf5f5}
.callout.warn{border-color:var(--warn);background:#fdfaf3}
.callout.ok{border-color:var(--ok);background:#f2f9f5}
.callout p:first-child{margin-top:0}.callout p:last-child{margin-bottom:0}
figure{margin:22px 0}figcaption{color:var(--mut);font-size:12px;text-align:center;margin-top:6px}
ul,ol{margin:12px 0;padding-left:24px}li{margin:5px 0}
@media(max-width:720px){main{padding:26px 16px 44px}
 table{display:block;overflow-x:auto;white-space:nowrap}}
@media print{body{background:#fff}main{box-shadow:none;padding:0;max-width:none}}
"""


def build(root, window, out_path, noise_floor=0.0053):
    runs = OrderedDict()
    for v in VARIANTS:
        pc, pr = load(root, v)
        if pc:
            runs[v] = (pc, pr)
    if not runs:
        raise SystemExit("no round_metrics.jsonl under %s" % root)

    rounds = sorted({r for (pc, _) in runs.values() for (r, _) in pc})
    shared_rounds = common_rounds(runs)
    if not shared_rounds:
        raise SystemExit("variants share no completed round; nothing to pair")
    wins = OrderedDict((v, per_client_window(pc, shared_rounds, window))
                       for v, (pc, _) in runs.items())
    n_clients = len(next(iter(wins.values())))

    cfg = next(iter(runs.values()))[1][rounds[0]]
    local_steps = cfg.get("local_steps", "?")

    base = "fedavg" if "fedavg" in wins else list(wins)[0]
    comps = OrderedDict((v, paired(wins[base], w)) for v, w in wins.items() if v != base)

    # Verdict derived from the data, never assumed -- and `local` is excluded
    # from it. Local IoU is measured on each client's own data, and `local`
    # never aggregates, so it fits its own distribution better by construction.
    # Counting that as "the mechanism works" would be reading an artifact of the
    # metric as a result.
    judged = OrderedDict((v, c) for v, c in comps.items() if v != "local")
    biggest = (max(judged.values(),
                   key=lambda c: abs(c["mean"]) if c["mean"] == c["mean"] else 0)
               if judged else None)
    detectable = [v for v, c in judged.items()
                  if c["t"] == c["t"] and abs(c["t"]) >= 2]

    H = []
    a = H.append
    a('<!doctype html><html lang="zh"><head><meta charset="utf-8">')
    a('<meta name="viewport" content="width=device-width,initial-scale=1">')
    a('<title>联邦变体试点结果</title><style>%s</style></head><body><main>' % CSS)
    a('<h1>联邦变体试点结果</h1>')
    a('<p class="sub">%d 轮 · 每轮 %s 本地步 · %d 个客户端 · 同一 seed · '
      'ViPT + LasHeR/DepthTrack/VisEvent</p>' % (len(rounds), local_steps, n_clients))

    # ---- what this is and is not
    a('<div class="callout warn"><p><strong>先说清楚这份结果能答什么、不能答什么。</strong></p>'
      '<ul>'
      '<li><strong>能答:</strong>四个变体之间的<strong>效应量级</strong>——'
      '差 0.3 分还是 3 分。这正是定实验矩阵需要的数。</li>'
      '<li><strong>不能答:</strong>论文主表的数字。这里的 IoU 是<strong>本地训练 IoU</strong>,'
      '不是 LasHeR 基准分。要基准分必须把导出的 checkpoint 走官方评估。</li>'
      '<li><strong>不公平的对比:</strong><code>ours</code> vs <code>modality</code>。'
      'warm-up 的四个相位被压缩进 %d 轮,不是论文的 120 轮设定。</li>'
      '</ul></div>' % len(rounds))

    # ---- chart
    a('<h2>1. 逐轮轨迹</h2>')
    a('<figure>%s<figcaption>每轮所有客户端的平均本地 IoU。'
      '轮次 r 的值反映的是「第 r-1 轮聚合出来的模型好不好」,在新批次上测'
      '</figcaption></figure>' % svg_chart(runs, rounds))

    # ---- headline
    a('<h2>2. 最后 %d 轮的配对比较</h2>' % window)
    a('<p>四个变体共用 seed,client 的采样种子由 (seed, 轮次, client_id) 派生,'
      '所以同一个 (轮次, 客户端) 在四次运行里看到的是<strong>同一批数据</strong>。'
      '差异只来自它们拿到的模型 —— 可以做配对比较。</p>')

    a('<table><tr><th>变体</th><th>平均本地 IoU</th><th>平均 loss</th></tr>')
    for v, w in wins.items():
        a('<tr><td>%s</td><td class="num">%.4f</td><td class="num">%.4f</td></tr>'
          % (_h.escape(LABEL[v]), mean(x["IoU"] for x in w.values()),
             mean(x["loss"] for x in w.values())))
    a('</table>')

    a('<h3>相对 %s 的配对差异(每个客户端一个数,n=%d)</h3>'
      % (_h.escape(LABEL[base]), n_clients))
    if "local" in wins and base != "local":
        a('<div class="callout bad"><p><strong>读这张表前必须知道一件事:'
          '<code>local</code> 在这个指标上赢是几乎注定的,不是它更好。</strong></p>'
          '<p>本地 IoU measured 在每个客户端<strong>自己的数据</strong>上。'
          '<code>local</code> 从不聚合、只拟合自己那几十个序列,'
          '在自己的分布上当然占优。真正的问题是<strong>泛化到 LasHeR 基准</strong>,'
          '那里 <code>local</code> 应该输 —— 因为它从来没用上其他客户端的数据。</p>'
          '<p>所以 <code>local</code> 这一行只能当作「本地过拟合程度」的参考,'
          '<strong>不能当作方法对比</strong>。有意义的比较是 '
          '<code>modality</code> / <code>ours</code> 对 <code>fedavg</code> —— '
          '它们三个都做聚合,面对的是同一个泛化压力。</p></div>')

    a('<table><tr><th>变体</th><th>ΔIoU</th><th>标准误</th><th>t</th>'
      '<th>胜/负</th><th>判定</th></tr>')
    for v, c in comps.items():
        if c["t"] == c["t"] and abs(c["t"]) >= 2:
            verdict = '<strong style="color:var(--ok)">可辨</strong>'
        else:
            verdict = '<span style="color:var(--mut)">噪声内</span>'
        a('<tr><td>%s</td><td class="num">%+.4f</td><td class="num">%.4f</td>'
          '<td class="num">%+.2f</td><td class="num">%d/%d</td><td>%s</td></tr>'
          % (_h.escape(LABEL[v]), c["mean"], c["se"], c["t"], c["win"], c["lose"],
             verdict))
    a('</table>')
    a('<p style="color:var(--mut);font-size:13px">|t| &lt; 2 表示该差异与噪声不可区分。'
      '胜/负是逐客户端计数 —— 接近一半一半就是抛硬币。</p>')

    # ---- by modality
    a('<h2>3. 按客户端模态拆开</h2>')
    a('<p>模态感知聚合的机制预期是:<strong>有传感器的客户端受益,纯 RGB 客户端基本不受影响</strong>'
      '(它本来就不写任何辅助适配器)。这张表是检验这个预期的地方。</p>')
    a('<table><tr><th>变体</th>')
    auxes = sorted({x["aux_type"] for x in wins[base].values()})
    for ax in auxes:
        a('<th>%s</th>' % _h.escape("纯 RGB" if ax == "none" else ax))
    a('</tr>')
    for v, w in wins.items():
        if v == base:
            continue
        a('<tr><td>%s</td>' % _h.escape(LABEL[v]))
        for ax in auxes:
            cids = [c for c in set(wins[base]) & set(w) if w[c]["aux_type"] == ax]
            d = mean(w[c]["IoU"] - wins[base][c]["IoU"] for c in cids)
            a('<td class="num">%+.4f <span style="color:var(--mut)">(n=%d)</span></td>'
              % (d, len(cids)))
        a('</tr>')
    a('</table>')

    # Mechanistic check, derived from the data: modality-aware aggregation
    # changes how AUX_SPECIFIC tensors are averaged. An RGB-only client writes
    # none of those, so the prediction is a near-zero delta for it and a
    # positive one for sensor clients. Stating it only when the data says so.
    if "modality" in wins and base == "fedavg":
        w = wins["modality"]
        b = wins[base]
        def _d(ax):
            cs = [c for c in set(b) & set(w) if w[c]["aux_type"] == ax]
            return mean(w[c]["IoU"] - b[c]["IoU"] for c in cs) if cs else float("nan")
        d_none = _d("none")
        sensor = [_d(ax) for ax in auxes if ax != "none"]
        sensor = [x for x in sensor if x == x]
        if d_none == d_none and sensor and abs(d_none) < 0.25 * mean(sensor):
            a('<div class="callout ok"><p><strong>机制核对通过。</strong>'
              '模态感知聚合只改变<strong>辅助适配器</strong>怎么平均。'
              '纯 RGB 客户端根本不写这些参数,所以机制预测它的差异应该接近零 —— '
              '实测 <strong>%+.4f</strong>,而有传感器的客户端平均 <strong>%+.4f</strong>。</p>'
              '<p>效应<strong>恰好出现在机制说它该出现的地方,且只出现在那里</strong>。'
              '这比总体差异更能说明这不是噪声或某个无关因素造成的。</p></div>'
              % (d_none, mean(sensor)))

    # ---- verdict
    a('<h2>4. 这对实验矩阵意味着什么</h2>')
    a('<p>ViPT 复现时测出的<strong>测量噪声地板是 AUC 1.0 分</strong>'
      '(配对标准误 %.4f)。两个模型在 LasHeR 245 个序列上差距小于这个值,就无法判定谁更好。</p>'
      % noise_floor)

    # `ours` below `modality` is the expected consequence of compressing the
    # warm-up, not a sign the warm-up hurts. Say so only when it actually happens.
    if ("ours" in comps and "modality" in comps
            and comps["ours"]["mean"] == comps["ours"]["mean"]
            and comps["ours"]["mean"] < comps["modality"]["mean"]):
        n_warm = sum(1 for r in shared_rounds
                     if runs["ours"][1].get(r, {}).get("phase") == "rgb_common")
        a('<div class="callout warn"><p><strong>为什么完整方法 <code>ours</code> '
          '反而低于消融 <code>modality</code>?这是压缩 warm-up 的必然结果,'
          '不是 warm-up 有害。</strong></p>'
          '<p><code>ours</code> 的前 %d 轮处在 <code>rgb_common</code> 相位,'
          '此时<strong>辅助适配器是冻结的</strong> —— 也就是它比 <code>modality</code> '
          '少了 %d 轮训练适配器的机会。在论文的 120 轮设定里,'
          'warm-up 占的是前 20 轮、后面还有 100 轮,代价被摊薄;'
          '压进 20 轮就变成占了 %.0f%%。</p>'
          '<p><strong>所以这个试点无法判断 warm-up 的价值,只能判断模态感知的价值。</strong>'
          '要评 warm-up 必须按 120 轮跑。</p></div>'
          % (n_warm, n_warm, 100.0 * n_warm / max(len(shared_rounds), 1)))

    if detectable:
        a('<div class="callout ok"><p><strong>试点里 %s 与 %s 的差异是可辨的。</strong>'
          '最大效应 ΔIoU = %+.4f。</p>'
          '<p>这说明机制本身有效应。下一步该做的是把导出的 checkpoint 走一遍官方 LasHeR 评估,'
          '看这个效应换算到基准分上有多大 —— 只有那个数能决定主表撑不撑得住。</p></div>'
          % ("、".join(_h.escape(LABEL[v]) for v in detectable),
             _h.escape(LABEL[base]), biggest["mean"] if biggest else float("nan")))
    else:
        a('<div class="callout bad"><p><strong>在这个试点规模下,没有任何变体与 %s 的差异是可辨的。</strong>'
          '最大效应 ΔIoU = %+.4f,|t| 全部小于 2。</p>'
          '<p>这<strong>不等于</strong>方法无效 —— 20 轮只有复现数据预算的约 2 个 epoch,'
          '而且本地 IoU 不是基准分。但它确实说明:<strong>照当前设置放大到完整实验,'
          '很可能得不到能写进论文的差距。</strong></p>'
          '<p>所以需要先做的是<strong>把效应做大</strong>,而不是直接开跑完整矩阵:'
          '提高缺模态客户端比例、加重 Non-IID(调小 Dirichlet α)、'
          '或减少每轮本地步数让聚合策略的影响更突出。</p></div>'
          % (_h.escape(LABEL[base]), biggest["mean"] if biggest else float("nan")))

    a('<h2>5. 怎么复现这次试点</h2>')
    a('<p style="color:var(--mut);font-size:13px">原始日志:'
      '<code>output/fed/pilot20/&lt;variant&gt;/round_metrics.jsonl</code>,每轮一行,'
      '含相位、可训练组、loss 权重、聚合统计和每个客户端的 loss/IoU。</p>')
    a('<pre style="background:#1e1e22;color:#e6e6e6;padding:14px 16px;border-radius:7px;'
      'overflow-x:auto;font:12.5px/1.65 var(--mono)">'
      'bash scripts/run_pilot.sh\n'
      'python scripts/compare_pilot.py --window %d</pre>' % window)

    a('</main></body></html>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(H))
    return out_path, comps, detectable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"D:\cvresearch\ViPT\output\fed\pilot20")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--out", default=r"D:\cvresearch\ViPT\output\eval_results\pilot_results.html")
    args = p.parse_args()
    path, comps, detectable = build(args.root, args.window, args.out)
    print("wrote %s  (%.1f KB)" % (path, os.path.getsize(path) / 1024))
    for v, c in comps.items():
        print("  %-10s dIoU %+.4f  SE %.4f  t %+.2f  %d/%d"
              % (v, c["mean"], c["se"], c["t"], c["win"], c["lose"]))
    print("  detectable: %s" % (detectable or "none"))


if __name__ == "__main__":
    main()
