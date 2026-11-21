"""Generate a standalone, sendable copy of the ViPT reproduction run report.

Why: the report lives as a hosted Claude Doc, which cannot simply be handed to
someone outside the account. This writes a self-contained .html (opens in any
browser, no network, chart included as inline SVG) and a .md, so the report can
go over WeChat as a file.

Content mirrors the doc as of rev 15, including the user's edits.
"""

import json
import os

OUT_DIR = r"D:\cvresearch\ViPT\output\eval_results"
CURVE = os.path.join(OUT_DIR, "train_curve.json")


def build_svg():
    """Inline SVG of the training curve. Drawn by hand so the file needs no CDN."""
    d = json.load(open(CURVE))
    tr = [(e, l) for e, l, _ in d["train"]]
    va = [(e, l) for e, l, _ in d["val"]]
    vi = [(e, i) for e, _, i in d["val"]]

    W, H = 820, 360
    ml, mr, mt, mb = 60, 62, 28, 44
    pw, ph = W - ml - mr, H - mt - mb

    xs = [e for e, _ in tr]
    x0, x1 = min(xs), max(xs)
    losses = [v for _, v in tr] + [v for _, v in va]
    y0, y1 = min(losses) - 0.02, max(losses) + 0.02
    i0, i1 = min(v for _, v in vi) - 0.004, max(v for _, v in vi) + 0.004

    def px(e):
        return ml + (e - x0) / (x1 - x0) * pw

    def py(v):
        return mt + (y1 - v) / (y1 - y0) * ph

    def pi(v):
        return mt + (i1 - v) / (i1 - i0) * ph

    def path(points, fy):
        return " ".join(("M" if k == 0 else "L") + "%.1f %.1f" % (px(e), fy(v))
                        for k, (e, v) in enumerate(points))

    parts = ['<svg viewBox="0 0 %d %d" width="100%%" xmlns="http://www.w3.org/2000/svg" '
             'font-family="system-ui,sans-serif" font-size="11">' % (W, H)]
    parts.append('<rect width="%d" height="%d" fill="#fff"/>' % (W, H))

    # grid + left axis labels (loss)
    for k in range(5):
        v = y0 + (y1 - y0) * k / 4
        y = py(v)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e8e8e8"/>'
                     % (ml, y, ml + pw, y))
        parts.append('<text x="%d" y="%.1f" text-anchor="end" fill="#666">%.2f</text>'
                     % (ml - 8, y + 4, v))
    # right axis labels (IoU)
    for k in range(5):
        v = i0 + (i1 - i0) * k / 4
        parts.append('<text x="%d" y="%.1f" fill="#666">%.3f</text>'
                     % (ml + pw + 8, pi(v) + 4, v))
    # x labels
    for e in (1, 10, 20, 30, 40, 50, 60):
        parts.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#666">%d</text>'
                     % (px(e), mt + ph + 18, e))
    parts.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#444">Epoch</text>'
                 % (ml + pw / 2, H - 8))

    # LR drop marker
    parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#bbb" '
                 'stroke-dasharray="4 3"/>' % (px(48), mt, px(48), mt + ph))
    parts.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#999">LR drop</text>'
                 % (px(48), mt - 8))

    parts.append('<path d="%s" fill="none" stroke="#2f6fb5" stroke-width="1.6"/>'
                 % path(tr, py))
    parts.append('<path d="%s" fill="none" stroke="#c4562a" stroke-width="1.8"/>'
                 % path(va, py))
    parts.append('<path d="%s" fill="none" stroke="#3e8e5a" stroke-width="1.5" '
                 'stroke-dasharray="5 3"/>' % path(vi, pi))
    for e, v in va:
        parts.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="#c4562a"/>' % (px(e), py(v)))

    lx, ly = ml + 14, mt + 12
    for k, (name, colour) in enumerate((("train loss", "#2f6fb5"),
                                        ("val loss", "#c4562a"),
                                        ("val IoU (right axis)", "#3e8e5a"))):
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2"/>'
                     % (lx, ly + k * 15, lx + 18, ly + k * 15, colour))
        parts.append('<text x="%d" y="%d" fill="#333">%s</text>'
                     % (lx + 24, ly + k * 15 + 4, name))
    parts.append('</svg>')
    return "".join(parts)


SECTIONS = []


def sec(md):
    SECTIONS.append(md.strip("\n"))


sec("""
# ViPT 复现运行报告

2026-09-18 · LasHeR RGB-T · 联邦多模态跟踪项目 阶段 1

## 一页结论

ViPT 在 LasHeR 上的复现过关,可以往上搭联邦层。但不是逐点复现,差距落在测量分辨率的边缘。

| 指标 | 我们 ep0055 | 我们 ep0060 | 论文 ViPT | 差 |
| --- | --- | --- | --- | --- |
| PR (20 px) | 63.8 | 63.4 | 65.1 | −1.3 |
| SR(即 AUC) | 51.5 | 51.1 | 52.5 | −1.0 |

判「过关」不是因为看着差不多。四条独立检查都指向同一结论:评分代码对 MATLAB 逐位一致、差距摊在 245 个序列上而非少数崩掉、权重加载无混用、差距量级与已知误差源相符。

### 最需要决策的一件事

这轮跑出了一个比复现结果本身更重要的数字:测量噪声地板是 AUC 1.0 分、PR 1.4 分。两个模型差距小于这个值,就无法判定谁更好。

直接后果:联邦四个变体(Local / FedAvg / Modality-aware / Ours)如果最终只差 0.5 到 1 分,在当前实验设置下证明不了。出路只有两条:把效应做大,或多 seed 跑并报均值 ± 方差(GPU 时间 ×3 起)。

这件事现在定比实验全跑完再定便宜得多。
""")

sec("""
## 训练运行

60 epoch 跑完,无中断,耗时 10 小时 46 分(2026-09-17 15:26 到 2026-09-18 02:12),平均每 epoch 约 10.8 分钟。

| 项 | 值 |
| --- | --- |
| 配置 | `experiments/vipt/repro_rgbt.yaml` = 官方 `deep_rgbt.yaml` + `ACCUM_ITER: 2` |
| 批量 | BATCH_SIZE 32 × 累积 2 = 等效 64,对齐论文的 2-GPU 设置 |
| 硬件 | 单卡 RTX 4080 SUPER,16376 MiB |
| 学习率 | 4e-4,第 48 epoch 下降 |
| 采样 | 训练 60000 / 验证 10000 样本每 epoch |
| 可训练参数 | 839,628 / 93,361,053 = 0.90%(只调 prompt,主干冻结) |
| checkpoint | ep0005 到 ep0060 养 12 个,全部保留 |

@@CHART@@

两个值得看的地方。一是第 48 epoch 学习率下降后,train loss 从 1.37 再降一档到 1.33,说明学习率调度确实生效了。二是 val loss 全程低于 train loss,这是 ViPT 这类只调 0.9% 参数的方法的正常现象,不是泄露。

val 最优在 ep55(loss 1.24957 / IoU 0.72668),末轮 ep60 是 1.26953 / 0.72271。两个都拿去评了。
""")

sec("""
## 评估运行

两个 checkpoint 各跑了一遍全量测试集,各 245 个序列、220,703 帧。

| 项 | ep0055 | ep0060 |
| --- | --- | --- |
| 耗时 | 50.6 分 | 51.4 分 |
| 序列完成 | 245 / 245 | 245 / 245 |
| 跳过序列 | 0 | 0 |
| 日志中出现的 checkpoint 种类 | 1 | 1 |
| 平均帧率 | — | 20.67 FPS(单序列 5.14 到 34.34) |

协议是官方的 `RGBT_workspace/test_rgbt_mgpus.py`,4 进程并发,显存峰值约 4.0 GB / 16 GB。

完整性校验做了三项,都过了:245 个结果文件全在且无空文件;日志里只出现过一种 checkpoint 路径,没有混用;没有任何序列被“已存在则跳过”的逻辑跳掉。

还有一个效率观察。官方把模型构造写在逐序列的函数里,每个序列都要重新 build 模型并载入 381 MB checkpoint,245 次约白花 6 分钟。复现阶段不改它,但后面自己写联邦评估循环时不要照抄这个结构。
""")

sec("""
## 结果对照

ep0055 在四个指标上都略高于 ep0060,但这个差异不显著(见「测量噪声地板」)。

| checkpoint | PR (20px) | NPR (0.20) | SR50 | AUC | 选择依据 |
| --- | --- | --- | --- | --- | --- |
| ep0055 | 0.6383 | 0.6057 | 0.6193 | 0.5149 | val 最优 |
| ep0060 | 0.6341 | 0.6016 | 0.6139 | 0.5114 | 官方 `TEST.EPOCH: 60` |
| 论文 ViPT | 0.651 | — | — | 0.525 | `assets/results_lasher.png` |

论文数字是从官方仓库的结果图里读的(图例写着 `[0.651] ViPT` 和 `[0.525] ViPT`),不是凭记忆写的。

### 一个写表格会踩的坑

论文表里的 SR 是成功率曲线的 AUC,不是 SR@0.5。

我们的 SR@0.5 是 0.6193,AUC 是 0.5149 —— 差 10 个点。写表格时用错会凭空多出 10 分,看起来像大幅超过论文。官方 toolkit 的 `Readme.txt` 自己也在提醒这件事,说的就是要单独用 `ranking_type='AUC'` 重画成功率曲线。
""")

sec("""
## 评分工具凭什么可信

官方评分工具是 MATLAB 的,本机没有 MATLAB 也没有 Octave。自己另写一套指标会让数字和文献不可比,所以做法是逐行移植,再用官方自带的数据反向验证。

工具包里打包了 12 个已发表 tracker 在全部 1224 个序列上的结果,**以及 MATLAB 当年算出的曲线数组**。拿我们的 Python 跑同样的输入,逐元素比:

| 项 | 值 |
| --- | --- |
| 对比规模 | 12 tracker × 1224 序列 × (21 + 51) 阈值 = 1,058,304 个数 |
| 最大绝对差 | 0.000e+00 |

而且算出的 PR/SR 正好是 LasHeR benchmark 论文公布的值(mfDiMP 52.23 / 47.03、DMCNet 55.68 / 46.70 等),双重确认。

### 第一遍没对上,原因值得记住

第一次验证差 1.09e-2。原因是 MATLAB 的 `0:0.05:1` 和 `np.arange` 不是同一个数组。

MATLAB 的 colon 不是用 `a + i*d` 逐个累加,而是从两端各填一半以保证端点精确。对 `0:0.05:1`,两种构造在 5 个位置上给出不同的 double。其中索引 13 是致命的:朴素构造给 0.65000000000000002,MATLAB 给 0.64999999999999991,**低于** double(0.65)。

跟踪框是整数,IoU 会出现 13/20 这种恰好等于 double(0.65) 的值,于是 `重叠率 > 阈值` 判定相反。实测在 mfDiMP 的一个序列上差了整 4 帧。

关键在于:**汇总的 PR/SR/AUC 在两位小数上看不出任何区别**(都是 52.23 / 47.03),只有逐序列曲线才露出来。所以光读源码写一遍是不够的,必须找到金标准输出来对。这条对以后移植任何指标都适用。
""")

sec("""
## 差距归因

四条检查各自独立,结论是链路健康。

| 判据 | 结果 |
| --- | --- |
| 评分器对不对 | 对 MATLAB 逐位一致,不是嫌疑人 |
| 差距是否集中在少数崩掉的序列 | 不集中。去掉最差 10 个序列 → 0.531 / 0.657,反超论文 |
| 权重是否真加载 | 两轮日志各只出现一种 checkpoint 路径,0 跳过序列 |
| 差距是否大于噪声 | 只是勉强。噪声地板 AUC 1.0 / PR 1.4,差距是 1.0 / 1.3 |

AUC 分布是正常形状:70 个序列高于 0.7,15 个低于 0.1。低分尾巴是跟踪任务的正常失败,不是 bug。中位 0.522 高于均值 0.511,也是正常左尾。如果是链路坏了,差距会是 10 到 40 分且集中在特定模式上。

### 候选原因(尚未逐一排除)

1. 单卡加梯度累积 vs 论文的 2×A100。等效 batch 对齐了,但优化轨迹不完全等价
2. 随机种子与数据加载顺序
3. 我们对 LasHeR 训练集做过帧名重命名(310 / 979 序列不匹配 ViPT 硬编码的帧名格式)。修对齐应该是帮助而非损害,但这使我们的训练数据与作者当年的不完全相同
4. epoch 选择

### 能一刀切开的对照实验

用官方发布的 `ViPT_deep_rgbt.pth` 走我们这条评估链路。若它跑出 65.1,则链路没问题、是我们训练弱;若它也跑出 63.4,则是链路或数据差异,我们的训练其实没问题。

`ViPT/models/` 目前是空的,需要下载约 370 MB 且要梯子。这件事做不做都不影响论文 —— 联邦变体本来就该跟我们自己的 baseline 比,同一条链路、同一份数据、同一套评分,才是可归因的比较。
""")

sec("""
## 测量噪声地板

这是这轮跑出的最重要的数字,比复现结果本身更影响后面的工作。

把 ep55 和 ep60 当成两个模型做配对比较(同一批 245 个序列),结果是:

| 指标 | ep55 更好 | 更差 | 平手 | 配对均差 | t |
| --- | --- | --- | --- | --- | --- |
| AUC | 121 序列 | 120 | 4 | +0.0035 | +0.67 |
| PR | 101 序列 | 90 | 54 | +0.0043 | +0.61 |

121 对 120 就是抛硬币。t 绝对值远小于 2,两个 checkpoint 统计上不可区分。所以不要因为 ep55 分高就声称 val 选模型有效 —— 这次的数据不支持这个说法。

由此得到的地板:

| 量 | 值 | 含义 |
| --- | --- | --- |
| AUC 配对标准误 | 0.0053 | 两模型差距小于 1.0 分判不出来 |
| PR 配对标准误 | 0.0069 | 差距小于 1.4 分判不出来 |
| baseline AUC 的 95% 区间 | 0.5149 ± 0.0291 | 序列间 std 0.2325,n = 245 |

### 对实验矩阵的硬约束

联邦四个变体若最终只差 0.5 到 1 分,在 LasHeR 245 个序列上证明不了。两条出路:

- 把效应做大。例如把缺模态比例拉高、Non-IID 加重,让 baseline 真的掉下来
- 多 seed 跑并报均值 ± 方差。GPU 时间 ×3 起

另外两件顺带的结论。一是别再逐 epoch 刷分 —— 相邻 epoch 的差异全在噪声里,而且用 test 分数挑 epoch 本身就不可发表。二是不管最终选哪个 checkpoint,必须对每个联邦变体用同一套规则,否则比较是被操纵的。
""")

sec("""
## 过程中发现并处置的问题

### 数据标注异常

对 245 + 979 个序列做了三方核对(官方 GT 行数、数据集 `visible.txt` 行数、图片张数),发现两个异常,都无害。

一是测试集的 `advancedredcup`,它的 `visible.txt` 只剩 15 行,而图片和官方 GT 都是 1411 帧,且首行框宽与官方差 1 px。影响可忽略:初始框宽差 1 px,而打分时第 0 帧本来就被 GT 覆盖不计分。

二是训练集的 `cameraman_1202` 根本没有 `visible.txt`。它本来就不在训练列表里,也不在联邦划分里,所以没进训练、也没影响划分的帧数权重。

除这两个以外,全部序列的标注行数与图片张数一致。**结论是打分要用 toolkit 的 GT,不用数据集目录里的标注。**

### 结果目录会静默混用不同 epoch

官方脚本的结果目录只按配置名命名,`--epoch` 不进路径;而已存在的结果文件会被跳过。所以用 ep30 跑过几个序列、再用 ep60 跑全量,那几个序列会留着 ep30 的结果,指标就是两个模型的混合,而且不报任何错。

这次跑 ep55 前先把 ep60 的 245 个结果整目录挪走了,两轮结果都完整保留。以后重新评估前必须清空结果目录,并核对日志里打印的 checkpoint 路径。
""")

sec("""
## 产出物清单

路径都相对于 `D:\\cvresearch`。

| 路径 | 是什么 |
| --- | --- |
| `ViPT/output/eval_results/README.md` | 结果、噪声地板、复现命令 |
| `ViPT/output/eval_results/lasher_ep{55,60}_curves.npz` | 逐序列曲线,含序列名 |
| `ViPT/output/eval_results/train_curve.json` | 逐 epoch 训练与验证曲线 |
| `ViPT/RGBT_workspace/results/LasHeR/repro_rgbt_ep{55,60}/` | 两轮各 245 个预测框 |
| `ViPT/output/checkpoints/train/vipt/repro_rgbt/` | 12 个 checkpoint,ep0005 到 ep0060 |
| `scripts/lasher_eval.py` | 官方指标的 Python 移植 |
| `scripts/lasher_eval_validate.py` | 对 MATLAB 逐位验证的脚本 |
| `tools/LasHeR_Toolkit/` | 官方工具包,含 GT、实验对比数据 |
| `ViPT/lib/federated/` | 联邦层,89 个测试全过 |
| `CLAUDE.md` | 项目上下文,已同步本报告全部结论 |
""")

sec("""
## 下一步与待决事项

正在做:联邦训练主循环 `client.py` 和 `server.py`。数据侧 `client_data.py` 已铺好,不占 GPU。设计上把 seed 做成一等参数,不管实验矩阵最后怎么定都用得上。

之后接四个 baseline。退化路径已就绪:关掉模态感知就是 FedAvg,关掉 warm-up 就是无 warm-up。

### 待决

| 事项 | 代价 | 建议 |
| --- | --- | --- |
| 实验矩阵怎么砍,要不要多 seed | 影响后面所有 GPU 时间 | 优先级最高,需和刘翔宇定 |
| 下官方权重做对照 | 手动下载 + 51 分钟 GPU | 可做可不做,不影响论文 |
| 删 `data/_zips/` 的 495 GB 压缩包 | 盘从 176 GB 空闲回到 671 GB | 建议删,删前先核对解压产物 |
""")


def md_to_html(md, svg):
    """Small markdown subset: headings, tables, lists, bold, inline code, paragraphs."""
    import html as _h
    import re

    def inline(s):
        s = _h.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        return s

    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "@@CHART@@":
            out.append('<figure>%s<figcaption>output/repro_rgbt_console.log · 60 epoch'
                       '</figcaption></figure>' % svg)
            i += 1
        elif ln.startswith("### "):
            out.append("<h3>%s</h3>" % inline(ln[4:]))
            i += 1
        elif ln.startswith("## "):
            out.append("<h2>%s</h2>" % inline(ln[3:]))
            i += 1
        elif ln.startswith("# "):
            out.append("<h1>%s</h1>" % inline(ln[2:]))
            i += 1
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            body = ["<table>"]
            for r, cells in enumerate(rows):
                if r == 1 and all(set(c) <= set("-: ") for c in cells):
                    continue
                tag = "th" if r == 0 else "td"
                body.append("<tr>" + "".join("<%s>%s</%s>" % (tag, inline(c), tag)
                                             for c in cells) + "</tr>")
            body.append("</table>")
            out.append("".join(body))
        elif re.match(r"^(\d+\.|-) ", ln):
            ordered = bool(re.match(r"^\d+\.", ln))
            items = []
            while i < len(lines) and re.match(r"^(\d+\.|-) ", lines[i]):
                items.append(re.sub(r"^(\d+\.|-) ", "", lines[i]))
                i += 1
            t = "ol" if ordered else "ul"
            out.append("<%s>%s</%s>" % (t, "".join("<li>%s</li>" % inline(x)
                                                   for x in items), t))
        elif ln.strip() == "":
            i += 1
        else:
            out.append("<p>%s</p>" % inline(ln))
            i += 1
    return "\n".join(out)


CSS = """
:root{--fg:#1d1d1f;--mut:#666;--line:#e2e2e4;--acc:#2f6fb5;--bg:#fff}
*{box-sizing:border-box}
body{margin:0;background:#f6f6f7;color:var(--fg);
 font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
main{max-width:860px;margin:0 auto;background:var(--bg);padding:48px 44px 64px;
 box-shadow:0 1px 3px rgba(0,0,0,.08)}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:20px;margin:40px 0 12px;padding-top:18px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px;color:#333}
p{margin:12px 0}
code{background:#f1f2f4;padding:1px 5px;border-radius:4px;
 font:13px ui-monospace,"Cascadia Code",Consolas,monospace;word-break:break-all}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left;vertical-align:top}
th{background:#fafafb;font-weight:600}
tr:nth-child(even) td{background:#fcfcfd}
ul,ol{margin:12px 0;padding-left:24px}
li{margin:5px 0}
figure{margin:20px 0}
figcaption{color:var(--mut);font-size:12px;margin-top:6px;text-align:center}
strong{font-weight:650}
@media(max-width:700px){main{padding:26px 16px 40px}
 table{display:block;overflow-x:auto;white-space:nowrap}}
@media print{body{background:#fff}main{box-shadow:none;max-width:none;padding:0}}
"""


def main():
    svg = build_svg()
    md = "\n\n".join(SECTIONS)

    md_path = os.path.join(OUT_DIR, "ViPT_reproduction_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md.replace("@@CHART@@",
                           "![训练曲线](见 HTML 版,或 train_curve.json)"))

    html_path = os.path.join(OUT_DIR, "ViPT_reproduction_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                "<title>ViPT 复现运行报告</title><style>%s</style></head><body><main>%s"
                "</main></body></html>" % (CSS, md_to_html(md, svg)))

    for p in (html_path, md_path):
        print("%s  %.1f KB" % (p, os.path.getsize(p) / 1024))


if __name__ == "__main__":
    main()
