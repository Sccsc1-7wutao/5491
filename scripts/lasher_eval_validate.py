"""Validate scripts/lasher_eval.py against the MATLAB toolkit, element by element.

The toolkit ships tracking_results/ for 12 published trackers on all 1224
sequences AND tmp_mat/aveSuccessRatePlot_12alg_{overlap,error}_OPE.mat, which is
the exact ave_success_rate_plot array MATLAB produced from them. So we can check
our Python port against MATLAB output directly instead of trusting a reading of
the source.

Pass criterion: max absolute difference over every (tracker, sequence,
threshold) entry is at floating-point noise level. Anything larger means the
port is wrong and our own PR/SR numbers cannot be compared to the literature.
"""

import os
import sys
import time

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasher_eval import evaluate, _mean_curve, RANK_IDX_ERROR, RANK_IDX_OVERLAP  # noqa: E402

TOOLKIT = r"D:\cvresearch\tools\LasHeR_Toolkit"


def main():
    anno_dir = os.path.join(TOOLKIT, "annos")
    with open(os.path.join(TOOLKIT, "sequence_evaluation_config", "all_dataset.txt")) as f:
        seqs = [ln.strip() for ln in f if ln.strip()]
    print("sequences: %d" % len(seqs))

    mat_o = sio.loadmat(os.path.join(TOOLKIT, "tmp_mat", "aveSuccessRatePlot_12alg_overlap_OPE.mat"))
    mat_e = sio.loadmat(os.path.join(TOOLKIT, "tmp_mat", "aveSuccessRatePlot_12alg_error_OPE.mat"))
    names = [str(x[0][0]) for x in mat_o["name_tracker_all"]]
    ref_o = mat_o["ave_success_rate_plot"]      # (12, 1224, 21)
    ref_e = mat_e["ave_success_rate_plot"]      # (12, 1224, 51)
    print("reference: overlap %s  error %s" % (ref_o.shape, ref_e.shape))
    print("")

    only = sys.argv[1:] or names
    worst_o = worst_e = 0.0
    rows = []

    for k, tname in enumerate(names):
        if tname not in only:
            continue
        rdir = os.path.join(TOOLKIT, "tracking_results", tname + "_tracking_result")
        t0 = time.time()
        r = evaluate(seqs, anno_dir, rdir, norm_dst=False)
        dt = time.time() - t0

        if r["missing"] or r["failed"]:
            print("%-9s SKIP: missing=%d failed=%d  %s" % (
                tname, len(r["missing"]), len(r["failed"]), r["failed"][:2]))
            continue
        if len(r["names"]) != len(seqs):
            print("%-9s SKIP: scored %d of %d" % (tname, len(r["names"]), len(seqs)))
            continue

        d_o = np.abs(r["curves_ovl"] - ref_o[k]).max()
        d_e = np.abs(r["curves_err"] - ref_e[k]).max()
        worst_o = max(worst_o, d_o)
        worst_e = max(worst_e, d_e)

        # also compare the headline scalars the paper tables quote
        mc_e, _ = _mean_curve(r["curves_err"])
        mc_o, _ = _mean_curve(r["curves_ovl"])
        keep_e = ref_e[k].sum(axis=1) > np.finfo(float).eps
        keep_o = ref_o[k].sum(axis=1) > np.finfo(float).eps
        ref_pr = ref_e[k][keep_e].mean(axis=0)[RANK_IDX_ERROR]
        ref_sr = ref_o[k][keep_o].mean(axis=0)[RANK_IDX_OVERLAP]
        ref_auc = ref_o[k][keep_o].mean(axis=0).mean()

        rows.append((tname, mc_e[RANK_IDX_ERROR], ref_pr,
                     mc_o[RANK_IDX_OVERLAP], ref_sr, mc_o.mean(), ref_auc))
        print("%-9s  maxdiff overlap=%.3e  error=%.3e   (%.1fs)" % (tname, d_o, d_e, dt))

    print("")
    print("%-9s %-15s %-15s %-15s" % ("tracker", "PR20 ours/ref", "SR50 ours/ref", "AUC ours/ref"))
    for t, pr, rpr, sr, rsr, auc, rauc in rows:
        print("%-9s %6.2f /%6.2f   %6.2f /%6.2f   %6.2f /%6.2f" % (
            t, 100 * pr, 100 * rpr, 100 * sr, 100 * rsr, 100 * auc, 100 * rauc))

    print("")
    print("WORST absolute difference vs MATLAB:  overlap=%.3e  error=%.3e" % (worst_o, worst_e))
    ok = worst_o < 1e-9 and worst_e < 1e-9
    print("VERDICT: %s" % ("EXACT MATCH -- port is trustworthy" if ok
                           else "MISMATCH -- do not use these numbers"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
