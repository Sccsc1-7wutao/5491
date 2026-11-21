"""Pull finished LasHeR thermal evaluations from instance B, score them locally, and run the paired comparisons.

Usage (sdstrack env):
    python scripts\\thermal_pull_score.py B_indep_thermal_norgb [B_main_norgb_thermal B_fedavg_norgb_thermal B_norgb_thermal]

For every run: (1) require "END <run> files=245" in B's daemon logs, (2) tar + scp the 245 prediction files,
(3) score with the validated scripts\\lasher_eval.py (no scoring logic is re-implemented), (4) save <run>_curves.npz.
Then every comparison whose two sides both exist is printed (B-internal pairing only) and appended to Bres\\summary.md.
Reference archives (with pure-RGB clients) are copied once from the earlier scratchpad into Bres\\ref.
"""
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from paired_compare import bootstrap_ci  # noqa: E402

BRES = r"D:\cvresearch\ViPT\output\eval_results\Bres"
REF = os.path.join(BRES, "ref")
OLD = r"D:\Temp\claude\D--Downloads-cv--\e60c7863-bc89-49fa-b093-9a2b2f3527a0\scratchpad\Bres"
PY = r"D:\Conda\envs\sdstrack\python.exe"
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
B = ["-o", "ConnectTimeout=25", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-p", "26194"]
BHOST = "root@px1-jcy.matpool.com"
RES_DIR = "/root/cvresearch/ViPT/RGBT_workspace/results/LasHeR"

# label -> npz path (filled in below).  "with RGB" references first, pulled runs after.
REFS = {"fedavg": "batch1_fedavg", "bucket": "batch1_modality", "indep": "batch2_indep_thermal"}
PULLED = {
    "indep_norgb": "B_indep_thermal_norgb",      # batch9
    "main_norgb": "B_main_norgb_thermal",        # batch8 (aux-only bucketing, no pure-RGB)
    "fedavg_norgb": "B_fedavg_norgb_thermal",    # batch8
    "fna_norgb": "B_norgb_thermal",              # batch5 (aux+fusion+norm bucketing, no pure-RGB)
    "noaux_main_norgb": "B_noaux_main_norgb",    # zero-aux (missing modality) evaluation of main_norgb
    "noaux_fna_norgb": "B_noaux_fna_norgb",      # zero-aux evaluation of fna_norgb
}


def run(cmd, tries=6, sleep=30, capture=True):
    last = None
    for i in range(tries):
        try:
            p = subprocess.run(cmd, capture_output=capture, text=True, timeout=900)  # a hung connection counts as a failed try
        except subprocess.TimeoutExpired:
            time.sleep(sleep)
            continue
        if p.returncode == 0:
            return p
        last = p
        time.sleep(sleep)
    raise RuntimeError("command failed after %d tries: %s\n%s" % (tries, " ".join(cmd[:4]), ((last.stderr if last else "timeout") or "")[-300:]))


def npz_path(label):
    if label in REFS:
        return os.path.join(REF, REFS[label] + "_curves.npz")
    return os.path.join(BRES, PULLED[label] + "_curves.npz")


def prepare_refs():
    os.makedirs(REF, exist_ok=True)
    for name in REFS.values():
        dst = os.path.join(REF, name + "_curves.npz")
        src = os.path.join(OLD, name + "_curves.npz")
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)


def pull_and_score(run_name):
    out_npz = os.path.join(BRES, run_name + "_curves.npz")
    if os.path.exists(out_npz):
        print("[skip] %s already scored" % run_name)
        return True
    logs = run(["ssh"] + B + [BHOST, "cat /root/cvresearch/B_chain*.log /root/cvresearch/B_daemon*.log"]).stdout
    if ("END %s files=245" % run_name) not in logs:
        print("[wait] %s: no 'END %s files=245' in B logs yet" % (run_name, run_name))
        return False
    work = os.path.join(BRES, run_name)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    tgz = os.path.join(BRES, run_name + ".tgz")
    run(["ssh"] + B + [BHOST, "cd %s && tar czf /tmp/%s.tgz %s" % (RES_DIR, run_name, run_name)])
    run(["scp", "-q", "-P", "26194", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", "%s:/tmp/%s.tgz" % (BHOST, run_name), tgz])
    with tarfile.open(tgz) as t:
        t.extractall(work)
    pred = os.path.join(work, run_name)
    n = len([f for f in os.listdir(pred) if f.endswith(".txt")])
    if n != 245:
        raise RuntimeError("%s: %d prediction files after pull, expected 245" % (run_name, n))
    p = subprocess.run([PY, os.path.join(SCRIPTS, "lasher_eval.py"), "--results", pred, "--save-npz", out_npz],
                       capture_output=True, text=True)
    print(p.stdout.strip().splitlines()[-3:] if p.stdout else p.stderr[-400:])
    if p.returncode != 0 or not os.path.exists(out_npz):
        raise RuntimeError("scoring failed for %s: %s" % (run_name, p.stderr[-400:]))
    return True


def auc(path):
    d = np.load(path, allow_pickle=False)
    return d["names"].astype(str), d["curves_ovl"].mean(axis=1)


def stats_of(d, seed=42):
    se = d.std(ddof=1) / np.sqrt(d.size)
    tc = stats.t.ppf(0.975, d.size - 1)
    bl, bh = bootstrap_ci(d, seed)
    return d.mean() * 100, (d.mean() - tc * se) * 100, (d.mean() + tc * se) * 100, bl * 100, bh * 100, \
        int((d > 0).sum()), int((d < 0).sum())


def fmt(label, d):
    m, tl, th, bl, bh, w, l = stats_of(d)
    return "%-58s %+6.2f  t[%+.2f,%+.2f]  boot[%+.2f,%+.2f]  胜/负 %d/%d" % (label, m, tl, th, bl, bh, w, l)


def load_all():
    have = {}
    for label in list(REFS) + list(PULLED):
        p = npz_path(label)
        if os.path.exists(p):
            have[label] = auc(p)
    names0 = None
    for label, (names, _) in have.items():
        if names0 is None:
            names0 = names
        elif not np.array_equal(names0, names):
            raise RuntimeError("sequence order differs for %s" % label)
    return {k: v[1] for k, v in have.items()}


def report(a):
    lines = []
    lines.append("## 热红外 B 内部配对（%s）" % time.strftime("%m-%d %H:%M"))
    lines.append("各方法 AUC 均值（245 条）：" + "，".join("%s %.2f" % (k, v.mean() * 100) for k, v in a.items()))
    pairs = [  # (label, minuend, subtrahend)
        ("独立去纯RGB − 独立含纯RGB（去纯RGB的效应）", "indep_norgb", "indep"),
        ("独立去纯RGB − 分桶含纯RGB", "indep_norgb", "bucket"),
        ("独立去纯RGB − FedAvg含纯RGB", "indep_norgb", "fedavg"),
        ("分桶去纯RGB − 分桶含纯RGB（去纯RGB的效应，主方法）", "main_norgb", "bucket"),
        ("FedAvg去纯RGB − FedAvg含纯RGB（去纯RGB的效应）", "fedavg_norgb", "fedavg"),
        ("【公平检验】分桶去纯RGB − 独立去纯RGB（跨模态−独立）", "main_norgb", "indep_norgb"),
        ("分桶去纯RGB − FedAvg去纯RGB", "main_norgb", "fedavg_norgb"),
        ("fna去纯RGB − 独立去纯RGB", "fna_norgb", "indep_norgb"),
        ("fna去纯RGB − 分桶去纯RGB", "fna_norgb", "main_norgb"),
        ("缺模态：主方法去纯RGB 置零 − 完整输入", "noaux_main_norgb", "main_norgb"),
        ("缺模态：fna去纯RGB 置零 − 完整输入", "noaux_fna_norgb", "fna_norgb"),
        ("缺模态：置零下 主方法去纯RGB − fna去纯RGB", "noaux_main_norgb", "noaux_fna_norgb"),
    ]
    for label, x, y in pairs:
        if x in a and y in a:
            lines.append(fmt(label, a[x] - a[y]))
    if all(k in a for k in ("main_norgb", "bucket", "fedavg_norgb", "fedavg")):
        d = (a["main_norgb"] - a["bucket"]) - (a["fedavg_norgb"] - a["fedavg"])
        lines.append(fmt("【交互项】主方法去纯RGB效应 − FedAvg去纯RGB效应", d))
    return "\n".join(lines)


def main():
    runs = sys.argv[1:] or list(PULLED.values())
    os.makedirs(BRES, exist_ok=True)
    prepare_refs()
    for r in runs:
        try:
            pull_and_score(r)
        except Exception as e:  # keep going so one failure does not hide the others
            print("[ERROR] %s: %s" % (r, e))
    text = report(load_all())
    print(text)
    with open(os.path.join(BRES, "summary.md"), "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


if __name__ == "__main__":
    main()
