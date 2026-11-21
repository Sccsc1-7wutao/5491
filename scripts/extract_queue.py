#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分卷压缩包串行解压队列

为什么要串行:
  四组分卷合计约 425 GB。并行解压会让机械层面的磁盘 I/O 互相抢,
  总时间不会变短,反而更容易出问题。一个一个来,每个解完立刻校验。

为什么用 WinRAR:
  这些都是**分卷 ZIP**(.z01…+.zip)。Python 的 zipfile 明确不支持
  ("zipfiles that span multiple disks are not supported"),
  7-Zip 没装,机器上只有 WinRAR。WinRAR 是 GUI 程序、不往 stdout 写东西,
  所以这里靠"解压后数子目录个数"来判断成败,而不是看它的输出。

  注意必须指向**末卷 .zip**(它含中央目录),不能指向 .z01。

用法:
    python extract_queue.py              # 跑完整队列
    python extract_queue.py --dry-run    # 只看计划,不动手
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

WINRAR = r"C:\Program Files\WinRAR\WinRAR.exe"
DATA = r"D:\cvresearch\data"
ZIPS = os.path.join(DATA, "_zips")

# (任务名, 末卷路径, 解压目标父目录, 解出来的顶层目录名, 期望序列数)
QUEUE = [
    ("LasHeR-test",
     os.path.join(ZIPS, r"LasHeR0327\TestingSet\testingset.zip"),
     os.path.join(DATA, "lasher"), "testingset", 245),
    ("VisEvent-train",
     os.path.join(ZIPS, "VisEvent_train.zip"),
     os.path.join(DATA, "visevent"), "train_subset", 500),
    ("VisEvent-test",
     os.path.join(ZIPS, "VisEvent_test.zip"),
     os.path.join(DATA, "visevent"), "test_subset", 320),
]


def stamp():
    return datetime.now().strftime("%H:%M:%S")


def log(m):
    print(f"[{stamp()}] {m}", flush=True)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_size(p):
    t = 0
    for root, _d, files in os.walk(p):
        for f in files:
            try:
                t += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return t


def winrar_running():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq WinRAR.exe", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return "WinRAR.exe" in out
    except Exception:
        return False


def wait_for_free(poll=30):
    """等当前可能在跑的解压结束,避免抢磁盘。"""
    n = 0
    while winrar_running():
        if n % 10 == 0:
            log("检测到 WinRAR 正在运行,等它结束…")
        time.sleep(poll)
        n += 1


def extract(name, archive, dest_parent, top, expect):
    if not os.path.exists(archive):
        log(f"!! {name}: 找不到末卷 {archive}")
        return False

    out_dir = os.path.join(dest_parent, top)
    if os.path.isdir(out_dir):
        have = len([d for d in os.listdir(out_dir)
                    if os.path.isdir(os.path.join(out_dir, d))])
        if have >= expect:
            log(f"✓ {name}: 已有 {have} 个序列(>= 期望 {expect}),跳过")
            return True
        log(f"  {name}: 目标已存在但只有 {have}/{expect} 个序列,继续解压补齐")

    os.makedirs(dest_parent, exist_ok=True)
    log(f"→ {name}: 解压 {os.path.basename(archive)} → {dest_parent}")
    t0 = time.time()
    # -y 全部确认  -ibck 后台运行  末尾反斜杠表示目标是目录
    rc = subprocess.call([WINRAR, "x", "-y", "-ibck", archive, dest_parent + os.sep])
    dt = time.time() - t0

    if not os.path.isdir(out_dir):
        log(f"!! {name}: WinRAR rc={rc},但没生成 {out_dir}")
        return False

    seqs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    size = dir_size(out_dir)
    ok = len(seqs) >= expect
    log(f"{'✓' if ok else '!!'} {name}: rc={rc}  用时 {dt/60:.1f} 分  "
        f"{len(seqs)}/{expect} 个序列  {human(size)}")
    return ok


def make_visevent_trainlist():
    """visevent.py:48 要读 <root>/trainlist.txt,归档里没有,得自己生成。"""
    root = os.path.join(DATA, "visevent", "train_subset")
    if not os.path.isdir(root):
        log("  (train_subset 不存在,跳过 trainlist.txt 生成)")
        return
    dst = os.path.join(root, "trainlist.txt")
    seqs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for s in seqs:
            f.write(s + "\n")
    log(f"✓ 生成 {dst}({len(seqs)} 个序列)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("解压队列启动")
    for name, arc, parent, top, expect in QUEUE:
        log(f"  队列: {name:16} {os.path.basename(arc):22} → {parent}\\{top}  期望 {expect} 个序列")
    if args.dry_run:
        log("(--dry-run,不动手)")
        return 0

    wait_for_free()

    results = []
    for name, arc, parent, top, expect in QUEUE:
        results.append((name, extract(name, arc, parent, top, expect)))
        make_visevent_trainlist() if name == "VisEvent-train" else None

    log("=" * 56)
    for name, ok in results:
        log(f"{'✓' if ok else '!!'} {name}")
    bad = [n for n, ok in results if not ok]
    if bad:
        log(f"有 {len(bad)} 项没通过:{', '.join(bad)} —— 重跑本脚本会跳过已完成的")
        return 1
    log("全部解压完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
