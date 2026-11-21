"""Draft scene-based client partitions + validation of the derived VisEvent illumination label (data only, no training).

(A) VisEvent: mean luminance of vis_imgs per sequence (8 evenly spaced frames) + recording hour parsed from dvSave-YYYY_MM_DD_HH_MM_SS names.
    Validation on the 320 TEST sequences, which carry the official LI (low illumination) attribute: AUC of (darkness score) vs LI.
(B) LasHeR: exclusive scene groups from the official attributes (night LI / heavy-occlusion / bright HI / normal), sizes by sequences and frames.
(C) DepthTrack: indoor / wild + object-category grouping balanced by frames.
Outputs: ViPT\\output\\eval_results\\final_supp\\scene_split_draft.json and scene_split_draft.md
Run: D:\\Conda\\envs\\sdstrack\\python.exe D:\\cvresearch\\scripts\\scene_split_draft.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

try:
    import cv2
    def read_gray(p):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE); return None if im is None else im.astype(np.float32)
except ImportError:  # pragma: no cover
    from PIL import Image
    def read_gray(p):
        return np.asarray(Image.open(p).convert("L"), dtype=np.float32)

SPLIT = r"D:\cvresearch\ViPT\configs\fed\client_splits\lasher_depthtrack_visevent_seed42_v2.json"
DATA = r"D:\cvresearch\data"
ATTR = r"D:\cvresearch\ViPT\output\eval_results\batch3_supp\lasher_attrs"
TK = r"D:\cvresearch\tools\VisEvent_SOT_Benchmark"
OUT = r"D:\cvresearch\ViPT\output\eval_results\final_supp"
os.makedirs(OUT, exist_ok=True)

split = json.load(open(SPLIT, encoding="utf-8"))
seqs_of = defaultdict(list)
for c in split["clients"]:
    seqs_of[c["dataset"]] += c["sequence_ids"]
md = ["# 按场景划分客户端：可行性与草案（仅数据，未训练）", ""]
draft = {}


def nframes_dir(path):
    return len(os.listdir(path))


def auc(score, label):  # P(score_pos > score_neg), ties 0.5
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    g = (pos[:, None] > neg[None, :]).mean(); t = (pos[:, None] == neg[None, :]).mean()
    return float(g + 0.5 * t)


# ---------------- (A) VisEvent ----------------
def lum(root, seq, k=8):
    d = os.path.join(root, seq, "vis_imgs"); fs = sorted(os.listdir(d)); n = len(fs)
    idx = np.unique(np.linspace(0, n - 1, min(k, n)).astype(int))
    vals = [read_gray(os.path.join(d, fs[i])) for i in idx]
    vals = [v for v in vals if v is not None]
    return float(np.mean([v.mean() for v in vals])), float(np.mean([v.std() for v in vals])), n


def hour_of(name):
    m = re.match(r"^dvSave-\d{4}_\d{2}_\d{2}_(\d{2})_(\d{2})_(\d{2})", name)
    return int(m.group(1)) + int(m.group(2)) / 60 if m else None


test_root = os.path.join(DATA, "visevent", "test_subset"); train_root = os.path.join(DATA, "visevent", "train_subset")
tseq = [x.strip() for x in open(os.path.join(TK, "sequence_evaluation_config", "VisEvent_testing_subset.txt")).read().splitlines() if x.strip()]
tl = np.array([lum(test_root, s) for s in tseq])
LI = np.array([np.loadtxt(os.path.join(TK, "annos", "att", s + ".txt"), ndmin=1)[4] for s in tseq])  # index 4 = LI
dark_score = -tl[:, 0]
a_lum = auc(dark_score, LI)
hours = np.array([np.nan if hour_of(s) is None else hour_of(s) for s in tseq])
has_h = ~np.isnan(hours)
a_hour = auc(np.where(has_h, np.abs(hours - 12), np.nan)[has_h], LI[has_h]) if has_h.sum() > 10 else float("nan")
md += ["## VisEvent：估算的亮度标签能否复现官方低照度（LI）标注（在 320 条测试序列上验证）", "",
       f"- 官方 LI=1 的序列：{int(LI.sum())} / {len(tseq)}",
       f"- **平均亮度作为'偏暗'分数对 LI 的 AUC = {a_lum:.3f}**（0.5=无信息，1=完美）",
       f"- 带录制时间的序列 {int(has_h.sum())} 条；'离正午越远越暗'对 LI 的 AUC = {a_hour:.3f}",
       f"- LI=1 的平均亮度 {tl[LI == 1, 0].mean():.1f}，LI=0 的 {tl[LI == 0, 0].mean():.1f}（0–255）", ""]

train_seqs = seqs_of["visevent"]
trl = {}
for s in train_seqs:
    try:
        trl[s] = lum(train_root, s)
    except Exception as e:  # keep going
        print("lum failed", s, e)
vals = np.array([trl[s][0] for s in trl]); frs = np.array([trl[s][2] for s in trl]); names = list(trl)
q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
grp = np.where(vals <= q1, "dark", np.where(vals <= q2, "mid", "bright"))
md += ["## VisEvent 训练集（当前划分内 500 条）：按平均亮度三等分", "", "| 组 | 序列数 | 帧数 | 平均亮度范围 |", "|---|---:|---:|---|"]
vis_groups = {}
for g in ("dark", "mid", "bright"):
    m = grp == g; vis_groups[g] = [names[i] for i in np.flatnonzero(m)]
    md.append(f"| {g} | {int(m.sum())} | {int(frs[m].sum())} | {vals[m].min():.0f}–{vals[m].max():.0f} |")
md.append("")
draft["visevent"] = {"rule": "mean luminance of 8 frames per sequence, tertiles over the training sequences in the current split",
                     "validation_auc_vs_official_LI_on_test": round(a_lum, 3), "groups": {g: {"sequences": len(v), "frames": int(frs[[names.index(s) for s in v]].sum())} for g, v in vis_groups.items()}}

# ---------------- (B) LasHeR ----------------
order = [x.strip() for x in open(os.path.join(ATTR, "Attributes_order.txt"), encoding="utf-8").read().split(",") if x.strip()]
ix = {a: i for i, a in enumerate(order)}
lroot = os.path.join(DATA, "lasher", "trainingset")
rows = []
for s in seqs_of["lasher"]:
    v = np.fromstring(open(os.path.join(ATTR, "AttriSeqsTxt", s + ".txt"), encoding="utf-8").read(), sep=",")
    f = max(nframes_dir(os.path.join(lroot, s, d)) for d in os.listdir(os.path.join(lroot, s)) if os.path.isdir(os.path.join(lroot, s, d)))
    rows.append((s, v, f))
V = np.stack([r[1] for r in rows]); F = np.array([r[2] for r in rows]); S = [r[0] for r in rows]
li = V[:, ix["LI"]] == 1; hi = V[:, ix["HI"]] == 1; ho = V[:, ix["HO"]] == 1; to = V[:, ix["TO"]] == 1
g_night = li
g_occ = (~li) & (ho | to)
g_bright = (~li) & (~(ho | to)) & hi
g_norm = ~(g_night | g_occ | g_bright)
md += ["## LasHeR 草案：互斥的场景分组（官方属性）", "", "优先级：夜间(LI) > 遮挡(HO 或 TO，且非 LI) > 高照度(HI，且非 LI、非遮挡) > 其余。", "",
       "| 组 | 序列数 | 帧数 |", "|---|---:|---:|"]
las_groups = {}
for name, m in (("night(LI)", g_night), ("occlusion(HO|TO)", g_occ), ("bright(HI)", g_bright), ("normal", g_norm)):
    las_groups[name] = [S[i] for i in np.flatnonzero(m)]
    md.append(f"| {name} | {int(m.sum())} | {int(F[m].sum())} |")
md.append("")
md.append("三个有传感器客户端 + 一个纯RGB 客户端有两种分配方式：把规模最大的 normal 当纯RGB（与当前'纯RGB 最大'的结构一致），或把规模最小的 bright 当纯RGB（更接近缺模态通常是少数的直觉）。两种都要作为备选写进协议。")
md.append("")
draft["lasher"] = {"rule": "exclusive: night(LI) > occlusion(HO|TO,!LI) > bright(HI,!LI,!occ) > normal",
                   "groups": {k: {"sequences": len(v), "frames": int(sum(F[S.index(s)] for s in v)), "sequence_ids": v} for k, v in las_groups.items()}}

# ---------------- (C) DepthTrack ----------------
droot = os.path.join(DATA, "depthtrack", "train")
dt = []
for s in seqs_of["depthtrack"]:
    n = max(nframes_dir(os.path.join(droot, s, d)) for d in os.listdir(os.path.join(droot, s)) if os.path.isdir(os.path.join(droot, s, d)))
    m = re.match(r"^(.*?)(\d*)_(indoor|wild)$", s)
    dt.append((s, m.group(1) if m else s, m.group(3) if m else "unknown", n))
cats = defaultdict(lambda: [0, 0])
for s, c, sc, n in dt:
    if sc == "indoor":
        cats[c][0] += 1; cats[c][1] += n
wild = [x for x in dt if x[2] == "wild"]
md += ["## DepthTrack 草案：室内/野外 + 物体类别", "",
       f"- 野外：{len(wild)} 条序列，{sum(x[3] for x in wild)} 帧；室内共 {sum(v[0] for v in cats.values())} 条、{sum(v[1] for v in cats.values())} 帧、{len(cats)} 个物体类别；无后缀 {sum(1 for x in dt if x[2] == 'unknown')} 条",
       "- 把室内类别按帧数贪心分成两组，与野外组合成三个有传感器客户端：", ""]
bins = [[], []]; tot = [0, 0]
for c, (ns, nf) in sorted(cats.items(), key=lambda kv: -kv[1][1]):
    k = int(np.argmin(tot)); bins[k].append(c); tot[k] += nf
md += [f"  - 室内-A：{len(bins[0])} 个类别，{tot[0]} 帧", f"  - 室内-B：{len(bins[1])} 个类别，{tot[1]} 帧", f"  - 野外：{sum(x[3] for x in wild)} 帧", "",
       "注意：按类别分组是一个任意的切法（没有场景含义），只用来凑出三个规模均衡的客户端；真正有场景含义的只有室内/野外。", ""]
draft["depthtrack"] = {"rule": "indoor/wild by name; indoor categories balanced into two groups by frames", "wild_frames": int(sum(x[3] for x in wild)),
                       "indoor_A_frames": int(tot[0]), "indoor_B_frames": int(tot[1])}

json.dump(draft, open(os.path.join(OUT, "scene_split_draft.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(os.path.join(OUT, "scene_split_draft.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
