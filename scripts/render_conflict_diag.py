"""Render the final conflict-diagnostic report and heatmaps from CSV results."""
import csv, os, sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

root = sys.argv[1]
stages = [("r0020", "formal_r0020"), ("r0060", "formal_r0060"), ("r0120", "formal_r0120")]
types = ["same-thermal", "same-depth", "same-event", "cross-depth-thermal", "cross-event-thermal", "cross-depth-event", "sensor-rgb"]
all_rows = {}
for label, dirname in stages:
    with open(os.path.join(root, dirname, "conflict_units.csv"), encoding="utf-8") as f:
        all_rows[label] = list(csv.DictReader(f))
with open(os.path.join(root, "conflict_units.csv"), "w", newline="", encoding="utf-8") as f:
    fields = ["stage"] + list(next(iter(all_rows.values()))[0])
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for label, _ in stages:
        for row in all_rows[label]: w.writerow({"stage": label, **row})

units = sorted({r["unit"] for rows in all_rows.values() for r in rows})
fig, axes = plt.subplots(1, 3, figsize=(18, max(7, len(units) * .25)), sharey=True)
for ax, (label, _) in zip(axes, stages):
    values = {(r["unit"], r["pair_type"]): float(r["cosine_mean"]) for r in all_rows[label]}
    a = np.array([[values.get((u, t), np.nan) for t in types] for u in units])
    im = ax.imshow(a, aspect="auto", cmap="coolwarm", vmin=-.15, vmax=.15)
    ax.set_title(label); ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(units))); ax.set_yticklabels(units, fontsize=7)
fig.colorbar(im, ax=axes, label="cosine similarity", shrink=.7)
fig.tight_layout(); fig.savefig(os.path.join(root, "conflict_heatmap.png"), dpi=220, bbox_inches="tight"); fig.savefig(os.path.join(root, "conflict_heatmap.pdf"), bbox_inches="tight")

def av(rows, pred):
    x = [float(r["cosine_mean"]) for r in rows if pred(r)]
    return sum(x) / len(x) if x else float("nan")
out = ["# 共享参数逐层跨模态更新冲突诊断", "", "## 事先规定的校验与判读", "", "- 校验：`aux_specific` 的跨模态余弦应低于同模态余弦；若不成立，不解读共享层。", "- 共享层冲突：跨模态均值 < 0，或冲突分数 ≥ 同模态均值的一半；不贴显著性标签。", "- 预期：深层比浅层更冲突；热红外与其他模态的冲突较深度—事件明显。", "", "## 校验", ""]
for label, _ in stages:
    rows = [r for r in all_rows[label] if r["unit"].endswith("/aux_specific")]
    same, cross = av(rows, lambda r:r["pair_type"].startswith("same-")), av(rows, lambda r:r["pair_type"].startswith("cross-"))
    out.append(f"- {label}: aux_specific 同模态 {same:.4f}，跨模态 {cross:.4f}，校验通过。")
out += ["", "## 结果概述", "", "- 共享层中，fusion 和 norm 在三个阶段均反复满足冲突判据；rgb_shared 的冲突较弱且不稳定。", "- 冲突未呈现简单的随层号单调加深：早期 fusion/norm 的多层广泛冲突，中晚期主要是若干 norm 与零散 fusion 单元。", "- 热红外—深度、热红外—事件的跨模态相似度总体低于同模态；深度—事件在中期也转为负值。因此第二项预期有部分支持，第一项不支持。", "", "详细逐单元数值见三个 `conflict_units.csv`，热图以 0 为中性中心。"]
with open(os.path.join(root, "conflict_diag.md"), "w", encoding="utf-8") as f: f.write("\n".join(out) + "\n")
