"""LasHeR attribute-stratified descriptive analysis.

Read-only inputs: archived official attribute vectors and per-sequence curve NPZs.
Outputs: output/eval_results/batch3_supp/{lasher_attributes.csv,lasher_attributes.md}.
Run: D:\Conda\envs\sdstrack\python.exe D:\cvresearch\scripts\lasher_attributes.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats


VIPT = Path(r"D:\cvresearch\ViPT")
OUT = VIPT / "output" / "eval_results" / "batch3_supp"
ATTR_ROOT = OUT / "lasher_attrs"
CURVES = VIPT / "output" / "eval_results"
VARIANTS = (
    "batch1_fedavg", "batch1_modality", "batch1_modality_a05", "batch1_ours",
    "batch1_fedprox", "batch2_modality_a025", "batch2_modality_a075",
    "batch2_indep_thermal",
)
PAIRS = (
    ("batch1_modality", "batch1_fedavg", "modality_minus_fedavg"),
    ("batch2_indep_thermal", "batch1_modality", "indep_thermal_minus_modality"),
    ("batch2_indep_thermal", "batch1_fedavg", "indep_thermal_minus_fedavg"),
    ("batch1_ours", "batch1_modality", "ours_minus_modality"),
    ("batch2_modality_a075", "batch1_modality", "a075_minus_modality"),
)
EXPECTED_AUC = {
    "batch1_fedavg": 0.4691,
    "batch1_modality": 0.4756,
    "batch2_indep_thermal": 0.4872,
}


def load_scores(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    names = data["names"].astype(str)
    return names, {"AUC": data["curves_ovl"].mean(axis=1), "PR": data["curves_err"][:, 20]}


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    diff = a - b
    n = int(diff.size)
    se = float(diff.std(ddof=1) / np.sqrt(n))
    t = float(diff.mean() / se) if se else float("nan")
    p = float(stats.ttest_1samp(diff, 0).pvalue) if se else float("nan")
    ci_half = float(stats.t.ppf(0.975, n - 1) * se)
    return {
        "diff": float(diff.mean()), "se": se, "t": t, "p_uncorrected": p,
        "ci95_low": float(diff.mean() - ci_half), "ci95_high": float(diff.mean() + ci_half),
        "n": n, "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def holm(pvalues: list[float]) -> list[float]:
    """Holm adjusted p-values, retaining original order."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not ATTR_ROOT.is_dir():
        raise RuntimeError(f"attribute directory absent: {ATTR_ROOT}")
    attributes = [x.strip() for x in (ATTR_ROOT / "Attributes_order.txt").read_text(encoding="utf-8").split(",") if x.strip()]
    attr_files = {p.stem: p for p in (ATTR_ROOT / "AttriSeqsTxt").glob("*.txt")}
    # Keep the prespecified order, then automatically include later batch runs.
    discovered = sorted(p.name.removesuffix("_curves.npz") for p in CURVES.glob("batch*_curves.npz"))
    variants = tuple(dict.fromkeys((*VARIANTS, *discovered)))
    present = [v for v in variants if (CURVES / f"{v}_curves.npz").is_file()]
    missing = [v for v in VARIANTS if v not in present]
    if not present:
        raise RuntimeError("no requested curve archives found")

    names, first = load_scores(CURVES / f"{present[0]}_curves.npz")
    if names.size != 245:
        raise RuntimeError(f"expected 245 LasHeR test sequences, got {names.size}")
    duplicate_names = sorted({x for x in names if (names == x).sum() > 1})
    absent_attrs = [x for x in names if x not in attr_files]
    if duplicate_names or absent_attrs:
        raise RuntimeError(f"attribute/name mismatch; duplicate={duplicate_names}, absent={absent_attrs}")
    vectors = np.stack([np.fromstring(attr_files[x].read_text(encoding="utf-8"), sep=",") for x in names])
    if vectors.shape != (names.size, len(attributes)) or not np.isin(vectors, [0, 1]).all():
        raise RuntimeError(f"invalid attribute matrix shape/content: {vectors.shape}")

    values: dict[str, dict[str, np.ndarray]] = {present[0]: first}
    for variant in present[1:]:
        other_names, scores = load_scores(CURVES / f"{variant}_curves.npz")
        if not np.array_equal(names, other_names):
            raise RuntimeError(f"sequence order differs: {variant}")
        values[variant] = scores
    for variant, expected in EXPECTED_AUC.items():
        if variant not in values:
            raise RuntimeError(f"required acceptance variant missing: {variant}")
        actual = float(values[variant]["AUC"].mean())
        if abs(actual - expected) > 5e-5:
            raise RuntimeError(f"overall AUC acceptance failed: {variant}={actual:.8f}, expected={expected:.4f}")
    core = paired(values["batch2_indep_thermal"]["AUC"], values["batch1_modality"]["AUC"])
    if abs(core["diff"] - 0.0116) > 5e-5 or abs(core["ci95_low"] - 0.0055) > 5e-4 or abs(core["ci95_high"] - 0.0177) > 5e-4:
        raise RuntimeError(f"indep_thermal-modality acceptance failed: {core}")

    rows: list[dict[str, object]] = []
    for metric in ("AUC", "PR"):
        for col, attribute in enumerate(attributes):
            indices = np.flatnonzero(vectors[:, col] == 1)
            row: dict[str, object] = {"metric": metric, "attribute": attribute, "n": int(indices.size), "note": "sequence-level; exploratory/descriptive only"}
            for variant in present:
                row[f"{variant}_mean"] = float(values[variant][metric][indices].mean())
            for left, right, label in PAIRS:
                if left in values and right in values:
                    result = paired(values[left][metric][indices], values[right][metric][indices])
                    row.update({f"{label}_{key}": val for key, val in result.items()})
            rows.append(row)
    # Holm correction is a reference only: 19 attributes per metric and comparison family.
    for metric in ("AUC", "PR"):
        metric_rows = [r for r in rows if r["metric"] == metric]
        for _, _, label in PAIRS:
            usable = [r for r in metric_rows if f"{label}_p_uncorrected" in r]
            for row, adjusted in zip(usable, holm([float(r[f"{label}_p_uncorrected"]) for r in usable])):
                row[f"{label}_p_holm_19"] = adjusted
    fields = list(rows[0])
    with (OUT / "lasher_attributes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    def pp(x: float) -> str: return f"{100 * x:.2f}"
    auc_rows = [r for r in rows if r["metric"] == "AUC"]
    lines = [
        "# LasHeR 属性分析（探索性、描述性）", "",
        "全部为单一训练 seed 的逐序列 AUC/PR 分层汇总。属性是事后分组（19 个官方属性）；未校正的 t 或 p 不能作为显著性结论，也不支持因果解释。Holm 列仅作 19 属性家族的校正参照。", "",
        "## 标注与完整性核对", "",
        f"官方 `Attributes_order.txt` 含 {len(attributes)} 个属性：" + "、".join(attributes) + "。",
        f"曲线档的 {names.size} 条测试序列均与属性文件逐名匹配；无重名、无缺失。属性文件库共 {len(attr_files)} 条，分析只使用测试集 245 条。",
        "", "| 属性 | n |", "|---|---:|",
    ]
    lines += [f"| {r['attribute']} | {r['n']} |" for r in auc_rows]
    lines += ["", "## 全体 245 条序列的回归核对", "", "| 变体 | AUC | PR |", "|---|---:|---:|"]
    for variant in present:
        lines.append(f"| {variant} | {pp(values[variant]['AUC'].mean())} | {pp(values[variant]['PR'].mean())} |")
    lines += [
        "",
        f"验收通过：FedAvg {pp(values['batch1_fedavg']['AUC'].mean())}、modality {pp(values['batch1_modality']['AUC'].mean())}、indep_thermal {pp(values['batch2_indep_thermal']['AUC'].mean())} AUC；indep_thermal − modality = +{pp(float(core['diff']))} pp，95% t 区间 [{pp(float(core['ci95_low']))}, +{pp(float(core['ci95_high']))}]，胜/负/平={core['wins']}/{core['losses']}/{core['ties']}。",
        "", "## 属性内 AUC（百分点）", "",
        "下表的 Δ 是左变体减右变体；`|t|≥2?` 仅标识未校正的描述性阈值，并非显著性结论。完整 AUC/PR、未校正 p、Holm-19 参照、区间、胜负均在 CSV。", "",
        "| 属性 | n | FedAvg | modality | indep thermal | Δ indep−modality | t | 胜/负 | \|t\|≥2? | a075−modality |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in auc_rows:
        d = float(r.get("indep_thermal_minus_modality_diff", float("nan")))
        t = float(r.get("indep_thermal_minus_modality_t", float("nan")))
        flag = "是（未校正）" if abs(t) >= 2 else "否/不可辨"
        lines.append(f"| {r['attribute']} | {r['n']} | {pp(float(r['batch1_fedavg_mean']))} | {pp(float(r['batch1_modality_mean']))} | {pp(float(r['batch2_indep_thermal_mean']))} | {d*100:+.2f} | {t:+.2f} | {r['indep_thermal_minus_modality_wins']}/{r['indep_thermal_minus_modality_losses']} | {flag} | {float(r['a075_minus_modality_diff'])*100:+.2f} |")
    ranked = sorted(auc_rows, key=lambda r: float(r["indep_thermal_minus_modality_diff"]))
    lines += ["", "## 描述性观察", ""]
    for r in ranked[:2] + ranked[-2:]:
        d = float(r["indep_thermal_minus_modality_diff"])
        t = float(r["indep_thermal_minus_modality_t"])
        phrase = "|t|≥2（未校正）" if abs(t) >= 2 else "不可辨"
        lines.append(f"- `{r['attribute']}`（n={r['n']}）中，indep_thermal − modality 为 {d*100:+.2f} pp（t={t:+.2f}，{phrase}）；这是 19 个事后属性分组中的描述性比较。")
    lines += [
        f"- 全体 245 条序列的 indep_thermal − modality 为 +{pp(float(core['diff']))} pp；属性子集的差异不能用于推断热红外独立联邦为何更好。",
        "- 属性可重叠，同一序列可被计入多个行；因此各行不是互斥分桶，也不可将 n 直接相加。",
        "",
        "## 文件说明", "",
        "`lasher_attributes.csv` 每行是一个属性×指标（AUC 或 PR），包括所有已发现变体均值和 5 个预定义配对的均差、t、未校正 p、Holm-19 参照、95% t 区间、胜/负/平。重跑时会自动纳入 `batch*_curves.npz` 的新增变体；缺失的预期变体会如实列在下方。",
        f"缺失的预期曲线档：{', '.join(missing) if missing else '无'}。",
    ]
    (OUT / "lasher_attributes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'lasher_attributes.csv'}")
    print(f"wrote {OUT / 'lasher_attributes.md'}")


if __name__ == "__main__":
    main()
