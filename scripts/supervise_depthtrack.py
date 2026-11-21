#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DepthTrack 下载守护进程 —— 无人值守时用这个,不要直接跑 download_depthtrack.py

为什么需要:
  2026-09-09 夜间两次观察到下载进程被外部终止(exit 255、日志断在半行、
  无任何 Python 异常)。其中至少一次的原因已查明:项目目录当晚从
  D:\\Downloads\\cv科研 整体搬到了 D:\\cvresearch,正在运行的进程脚下的
  路径被抽走了。是否还有别的原因未确认。
  download_depthtrack.py 本身幂等可续传,所以外面套一层循环重启即可。

它做什么:
  1. 反复调用 download_depthtrack.py,直到它返回 0(全部完成并通过 MD5)
  2. 每轮之间等一会儿再来,避免打死循环
  3. 每轮记录已落盘体积,便于事后判断是真在推进还是原地打转
  4. 连续多轮零进展则停下,避免无意义地刷日志

用法:
    python supervise_depthtrack.py
    python supervise_depthtrack.py --workers 4 --max-rounds 100
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADER = os.path.join(HERE, "download_depthtrack.py")
DEFAULT_OUT = r"D:\cvresearch\data\_zips\depthtrack"
TARGET_BYTES = int(67.1 * 1024 ** 3)      # 67.1 GiB,用于打印进度百分比


def dir_size(p):
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def stamp():
    return datetime.now().strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=5,
                    help="并发数。夜间无人值守建议 4-5:实测 10-12 并发时"
                         "MD5 校验失败频发(经代理传输数据被损坏)")
    ap.add_argument("--max-rounds", type=int, default=200)
    ap.add_argument("--gap", type=int, default=20, help="每轮之间等待秒数")
    ap.add_argument("--stall-rounds", type=int, default=8,
                    help="连续多少轮零进展就放弃")
    args = ap.parse_args()

    if not os.path.exists(DOWNLOADER):
        print(f"!! 找不到下载器: {DOWNLOADER}")
        return 2

    os.makedirs(args.out, exist_ok=True)
    last_size = dir_size(args.out)
    stalled = 0

    print(f"[{stamp()}] 守护启动。起始 {human(last_size)} / {human(TARGET_BYTES)}"
          f"  并发={args.workers}  输出={args.out}", flush=True)

    for rnd in range(1, args.max_rounds + 1):
        t0 = time.time()
        try:
            rc = subprocess.call(
                [sys.executable, DOWNLOADER,
                 "--workers", str(args.workers), "--out", args.out]
            )
        except Exception as e:
            rc = -1
            print(f"[{stamp()}] 第 {rnd} 轮启动失败: {type(e).__name__}: {e}", flush=True)

        size = dir_size(args.out)
        gained = size - last_size
        pct = size / TARGET_BYTES * 100
        print(f"[{stamp()}] 第 {rnd} 轮结束 rc={rc}  用时 {(time.time()-t0)/60:.1f} 分  "
              f"本轮 +{human(max(gained, 0))}  累计 {human(size)} ({pct:.1f}%)",
              flush=True)

        if rc == 0:
            print(f"[{stamp()}] ✓ 下载器报告全部完成并通过 MD5 校验。守护退出。", flush=True)
            return 0

        stalled = stalled + 1 if gained <= 0 else 0
        last_size = size
        if stalled >= args.stall_rounds:
            print(f"[{stamp()}] !! 连续 {stalled} 轮零进展,停止。"
                  f"可能是代理断了或 Zenodo 不可达,醒来后人工看一眼。", flush=True)
            return 1

        time.sleep(args.gap)

    print(f"[{stamp()}] 达到最大轮数 {args.max_rounds},停止。", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
