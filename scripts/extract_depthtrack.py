#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DepthTrack 解压:202 个独立 zip → data/depthtrack/{train,test}/

和 LasHeR/VisEvent 不同,这批是**普通 zip**(每个 zip 一个序列),
不是分卷,Python 的 zipfile 能直接处理,不需要 WinRAR。

zip 内部结构(已核对 adapter02_indoor.zip):
    <序列名>/color/00000001.jpg …     8 位补零,从 1 开始
    <序列名>/depth/00000001.png …
    <序列名>/groundtruth.txt
    <序列名>/sequence
顶层就是序列名,所以直接解到 train/ 或 test/ 即可得到
    depthtrack/train/<序列名>/color/…
正是 depthtrack.py:111 期望的 `color/{:08}.jpg` / `depth/{:08}.png`。

目标序列数:train 100+52=152(对应 data_specs 的 146 train + 6 val),test 50。

用法:
    python extract_depthtrack.py
    python extract_depthtrack.py --workers 4
"""

import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

SRC = r"D:\cvresearch\data\_zips\depthtrack"
DST = r"D:\cvresearch\data\depthtrack"
GROUPS = [("train_part1", "train"), ("train_part2", "train"), ("test", "test")]
EXPECT = {"train": 152, "test": 50}


def stamp():
    return datetime.now().strftime("%H:%M:%S")


def log(m):
    print(f"[{stamp()}] {m}", flush=True)


def one(zp, out_root):
    """解压一个 zip;若目标序列目录已存在且非空则跳过。"""
    name = os.path.splitext(os.path.basename(zp))[0]
    seq_dir = os.path.join(out_root, name)
    try:
        if os.path.isdir(seq_dir) and os.listdir(seq_dir):
            return "skip", name, None
        with zipfile.ZipFile(zp) as z:
            # 先验一遍压缩包本身,坏了立刻知道,而不是解出半个序列
            bad = z.testzip()
            if bad is not None:
                return "fail", name, f"压缩包损坏,首个坏条目 {bad}"
            z.extractall(out_root)
        return "ok", name, None
    except Exception as e:
        return "fail", name, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    args = ap.parse_args()

    jobs = []
    for grp, split in GROUPS:
        d = os.path.join(args.src, grp)
        if not os.path.isdir(d):
            log(f"!! 找不到 {d}")
            continue
        out = os.path.join(args.dst, split)
        os.makedirs(out, exist_ok=True)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".zip"):
                jobs.append((os.path.join(d, fn), out))

    log(f"共 {len(jobs)} 个 zip 待处理,并发 {args.workers}")
    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "fail": 0}
    fails = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, zp, out): zp for zp, out in jobs}
        done = 0
        for fu in as_completed(futs):
            st, name, err = fu.result()
            stats[st] += 1
            done += 1
            if st == "fail":
                fails.append((name, err))
                log(f"  !! {name}: {err}")
            if done % 25 == 0:
                log(f"  进度 {done}/{len(jobs)}")

    log(f"完成:解压 {stats['ok']}  跳过 {stats['skip']}  失败 {stats['fail']}  "
        f"用时 {(time.time()-t0)/60:.1f} 分")

    ok_all = True
    for split, exp in EXPECT.items():
        p = os.path.join(args.dst, split)
        n = len([d for d in os.listdir(p)
                 if os.path.isdir(os.path.join(p, d))]) if os.path.isdir(p) else 0
        mark = "✓" if n >= exp else "!!"
        log(f"{mark} {split}: {n} / {exp} 个序列")
        ok_all &= (n >= exp)

    if fails:
        log("失败清单:")
        for n, e in fails:
            log(f"    {n}: {e}")
        return 1
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
