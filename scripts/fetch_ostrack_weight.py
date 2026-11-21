#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 OSTrack 预训练权重(带续传的重试循环)

为什么需要这个:
  经代理访问 Google Drive 时,drive.google.com 可达,但实际发文件的
  doc-*.googleusercontent.com 连接经常被切断(SSLEOFError)。
  gdown 自带 -c/--continue 可以续传,这里套一层重试循环把它跑完。

为什么是这个文件:
  Google Drive 那个 folder 里有 6 个 OSTrack 变体。
  experiments/sdstrack/cvpr2024_rgbt.yaml 要求:
      BACKBONE.TYPE: vit_base_patch16_224_ce   -> 要 ce 版
      DATA.SEARCH.SIZE: 256                    -> 要 256 版
  所以唯一正确的是 vitb_256_mae_ce_32x4_ep300。

  下错了不会报错:build_sdstrack 用 load_state_dict(..., strict=False),
  对不上的 key 会被静默跳过,模型照跑,等于没加载预训练权重。

用法:
    python fetch_ostrack_weight.py
    python fetch_ostrack_weight.py --retries 50
"""

import argparse
import os
import subprocess
import sys
import time

# Google Drive folder 1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy 内各变体的 file id
VARIANTS = {
    "vitb_256_mae_32x4_ep300":            "1lpmc5DhZTIluKdvawvt-5iOpLZtcRjm4",
    "vitb_256_mae_ce_32x4_ep300":         "1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV",  # ← SDSTrack 用这个
    "vitb_256_mae_ce_32x4_got10k_ep100":  "1jwTLPiwj_r7KkAYIFdH9lkbt1JH5MhGC",
    "vitb_384_mae_32x4_ep300":            "1rTD1W_gr1VNsPg-rjQHN5t2VfI7_Kk3n",
    "vitb_384_mae_ce_32x4_ep300":         "19tj5NmwZr9ylDaqOwfNVOLijr389ZDBi",
    "vitb_384_mae_ce_32x4_got10k_ep100":  "1WKaXPz1l3SdQWIFzq18ByrdxFJXtHMzV",
}

WANT = "vitb_256_mae_ce_32x4_ep300"
EXPECT_MIN = 300 * 1024 * 1024      # 约 370 MB,低于 300 MB 视为没下完
PYLIBS = r"D:\Temp\claude\D--Downloads-cv--\df870986-0b78-47ca-bad4-e5ef1f5e0d67\scratchpad\pylibs"
DEFAULT_OUT = (r"D:\cvresearch\SDSTrack\pretrained"
               r"\vitb_256_mae_ce_32x4_ep300\OSTrack_ep0300.pth.tar")


def looks_done(path):
    """大小达标 + 是 torch 存档(zip 魔数 PK 或旧式 pickle),不是 HTML 拦截页。"""
    if not os.path.exists(path):
        return False, "文件不存在"
    n = os.path.getsize(path)
    if n < EXPECT_MIN:
        return False, f"只有 {n/1024/1024:.1f} MB,未下完"
    with open(path, "rb") as fp:
        head = fp.read(4)
    if head[:2] == b"PK":                      # torch.save 新格式是 zip
        return True, f"{n/1024/1024:.1f} MB,zip 格式正常"
    if head[:1] == b"\x80":                    # 旧式 pickle 协议
        return True, f"{n/1024/1024:.1f} MB,pickle 格式正常"
    if head[:1] in (b"<", b"{"):
        return False, "开头是 < 或 {,八成是 HTML 拦截页或错误 JSON"
    return True, f"{n/1024/1024:.1f} MB,魔数 {head!r}(非常见格式,建议人工确认)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--variant", default=WANT, choices=sorted(VARIANTS))
    ap.add_argument("--retries", type=int, default=40)
    args = ap.parse_args()

    file_id = VARIANTS[args.variant]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ok, why = looks_done(args.out)
    if ok:
        print(f"已存在且完整({why}),无需下载:\n  {args.out}")
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = PYLIBS + os.pathsep + env.get("PYTHONPATH", "")

    print(f"变体   : {args.variant}")
    print(f"file id: {file_id}")
    print(f"目标   : {args.out}\n")

    for attempt in range(1, args.retries + 1):
        have = os.path.getsize(args.out) if os.path.exists(args.out) else 0
        print(f"--- 第 {attempt}/{args.retries} 次;本地已有 "
              f"{have/1024/1024:.1f} MB ---", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "gdown", "-c", file_id, "-O", args.out],
                env=env, check=False, timeout=1800,
            )
        except subprocess.TimeoutExpired:
            print("  单次超过 30 分钟,重来")

        ok, why = looks_done(args.out)
        if ok:
            print(f"\n✓ 下载完成:{why}\n  {args.out}")
            return 0
        print(f"  尚未完成:{why}")
        time.sleep(min(5 * attempt, 60))

    print(f"\n!! 重试 {args.retries} 次仍未完成。最后状态:{looks_done(args.out)[1]}")
    print("   可以直接重跑本脚本继续续传。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
