"""Compare the four federated variants from their round logs.

Why a paired comparison is valid here
-------------------------------------
Every variant runs with the same seed, and client.py derives its sample seed
from (base_seed, round, client_id). So for a given (round, client) the four runs
see the SAME batches. The difference between two variants at that point is the
model they were handed, which is exactly what we want to measure.

What the numbers mean, precisely
--------------------------------
`mean_IoU` for round r is measured DURING that round's local steps, i.e. on a
model that was just loaded from the state aggregated at the end of round r-1.
So it reads as "how good was last round's aggregation", on fresh batches.
It is not a held-out benchmark score -- for that the exported checkpoints have
to go through the official LasHeR evaluation.

Two caveats this script does NOT paper over:
  1. Rounds are not independent of each other (one trajectory), so a t-test
     across all (round, client) pairs would overstate significance. The headline
     comparison therefore uses the LAST `--window` rounds averaged per client,
     giving one number per client -- 12 independent-ish pairs.
  2. 20 rounds compresses the 120-round warm-up schedule, so `ours` vs
     `modality` is not a fair test of the warm-up. Printed as a reminder.
"""

import argparse
import json
import os
from collections import OrderedDict, defaultdict

VARIANTS = ["local", "fedavg", "modality", "ours"]


def load(root, variant):
    """{(round, client_id): {metric: value}} for one variant."""
    path = os.path.join(root, variant, "round_metrics.jsonl")
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for c in rec.get("clients", []):
                out[(rec["round"], c["client_id"])] = c
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def per_client_window(data, rounds, window):
    """Average each client's metric over the last `window` rounds."""
    last = sorted(rounds)[-window:]
    by_client = defaultdict(dict)
    for (r, cid), rec in data.items():
        if r in last:
            by_client[cid][r] = rec
    out = {}
    for cid, recs in by_client.items():
        out[cid] = {
            "IoU": mean(v.get("mean_IoU", float("nan")) for v in recs.values()),
            "loss": mean(v.get("mean_Loss/total", float("nan")) for v in recs.values()),
            "aux_type": next(iter(recs.values())).get("aux_type", "?"),
        }
    return out


def paired(a, b, key):
    """b - a per client, over the clients both have."""
    cids = sorted(set(a) & set(b))
    d = [b[c][key] - a[c][key] for c in cids]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), n, 0, 0
    m = mean(d)
    se = stdev(d) / (n ** 0.5)
    wins = sum(1 for x in d if x > 0)
    losses = sum(1 for x in d if x < 0)
    return m, se, n, wins, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"D:\cvresearch\ViPT\output\fed\pilot20")
    p.add_argument("--window", type=int, default=5,
                   help="average the last N rounds per client (default 5)")
    p.add_argument("--baseline", default="fedavg",
                   help="variant the others are compared against")
    args = p.parse_args()

    runs = OrderedDict()
    for v in VARIANTS:
        d = load(args.root, v)
        if d is None:
            print("missing: %s" % os.path.join(args.root, v, "round_metrics.jsonl"))
            continue
        runs[v] = d
    if not runs:
        return 1

    rounds = sorted({r for d in runs.values() for (r, _) in d})
    print("rounds: %d..%d   variants: %s" % (rounds[0], rounds[-1], list(runs)))
    print("")

    # ---------------------------------------------------- trajectory
    print("=== per-round mean IoU over all clients ===")
    head = "round " + "".join("%12s" % v for v in runs)
    print(head)
    for r in rounds:
        row = "%5d " % r
        for v, d in runs.items():
            vals = [rec.get("mean_IoU") for (rr, _), rec in d.items() if rr == r]
            vals = [x for x in vals if x is not None]
            row += "%12.4f" % mean(vals) if vals else "%12s" % "-"
        print(row)

    # ---------------------------------------------------- headline
    print("")
    print("=== last %d rounds, averaged per client (n=%d clients) ==="
          % (args.window, len(set(cid for (_, cid) in next(iter(runs.values()))))))
    win = OrderedDict((v, per_client_window(d, rounds, args.window))
                      for v, d in runs.items())
    print("%-10s %10s %10s" % ("variant", "mean IoU", "mean loss"))
    for v, w in win.items():
        print("%-10s %10.4f %10.4f"
              % (v, mean(x["IoU"] for x in w.values()),
                 mean(x["loss"] for x in w.values())))

    if args.baseline not in win:
        print("\nbaseline %r not present" % args.baseline)
        return 0

    print("")
    print("=== paired vs %s (positive IoU delta = better) ===" % args.baseline)
    print("%-10s %12s %10s %10s %14s"
          % ("variant", "dIoU", "SE", "t", "win/lose"))
    base = win[args.baseline]
    for v, w in win.items():
        if v == args.baseline:
            continue
        m, se, n, wins, losses = paired(base, w, "IoU")
        t = m / se if se and se == se and se > 0 else float("nan")
        print("%-10s %+12.4f %10.4f %10.2f %14s"
              % (v, m, se, t, "%d/%d" % (wins, losses)))

    # ---------------------------------------------------- by modality
    print("")
    print("=== dIoU vs %s, split by client modality ===" % args.baseline)
    for v, w in win.items():
        if v == args.baseline:
            continue
        groups = defaultdict(list)
        for cid in sorted(set(base) & set(w)):
            groups[w[cid]["aux_type"]].append(w[cid]["IoU"] - base[cid]["IoU"])
        cells = "  ".join("%s %+.4f (n=%d)" % (k, mean(vs), len(vs))
                          for k, vs in sorted(groups.items()))
        print("  %-10s %s" % (v, cells))

    print("")
    print("Reading these numbers:")
    print("  * |t| < 2 means the difference is not distinguishable from noise.")
    print("  * IoU here is local training IoU on fresh batches, NOT a LasHeR")
    print("    benchmark score. For that, evaluate the exported checkpoints.")
    print("  * 20 rounds compresses the 120-round warm-up, so `ours` vs")
    print("    `modality` is NOT a fair test of the warm-up schedule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
