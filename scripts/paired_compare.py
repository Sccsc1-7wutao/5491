"""Paired comparison for LasHeR evaluation curve archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load(path: str):
    data = np.load(path, allow_pickle=False)
    names = data["names"].astype(str)
    return names, data["curves_ovl"].mean(axis=1), data["curves_err"][:, 20]


def bootstrap_ci(x: np.ndarray, seed: int, draws: int = 20_000):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(draws, x.size))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def compare(a_path: str, b_path: str, seed: int = 42):
    names_a, auc_a, pr_a = load(a_path)
    names_b, auc_b, pr_b = load(b_path)
    if not np.array_equal(names_a, names_b):
        raise ValueError("sequence order differs between inputs")
    rows = []
    for metric, a, b in (("AUC", auc_a, auc_b), ("PR", pr_a, pr_b)):
        d = b - a
        se = d.std(ddof=1) / np.sqrt(d.size)
        tcrit = stats.t.ppf(0.975, d.size - 1)
        rows.append({
            "metric": metric,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "mean_diff_b_minus_a": float(d.mean()),
            "se": float(se),
            "t_ci95": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
            "p_two_sided": float(stats.ttest_1samp(d, 0).pvalue),
            "bootstrap_ci95": [float(v) for v in bootstrap_ci(d, seed)],
            "wins_b": int((d > 0).sum()),
            "losses_b": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
        })
    return {"a": str(Path(a_path)), "b": str(Path(b_path)), "n": int(names_a.size), "metrics": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="baseline *_curves.npz")
    parser.add_argument("b", help="comparison *_curves.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(compare(args.a, args.b, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
