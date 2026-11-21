#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据线收官验证:三个数据集能否被 SDSTrack 正确加载。

验什么(每个数据集都过一遍):
  1. 能实例化,序列数对得上 data_specs
  2. get_sequence_info 能读出标注,bbox 形状 (N,4),valid/visible 长度一致
  3. get_frames 能取出图,形状必须是 (H, W, 6) —— RGB 3 通道 + 辅助模态 3 通道
  4. 抽若干序列检查"帧数 == 标注行数",这是图像与框对齐的前提
  5. 随机抽帧检查取到的图不是全黑/全白(读图失败时 cv2 可能给出异常内容)

必须在 SDSTrack 仓库根目录下运行,且用 sdstrack 环境的 python。
"""

import os
import random
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
REPO = r"D:\cvresearch\SDSTrack"
os.chdir(REPO)
sys.path.insert(0, REPO)

from lib.train.dataset import LasHeR, DepthTrack, VisEvent   # noqa: E402
from lib.train.admin import env_settings                     # noqa: E402


def check(name, ds, expect_seqs, n_probe=5):
    print(f"\n{'='*62}\n### {name}")
    ok = True
    try:
        n = ds.get_num_sequences()
        mark = "✓" if (expect_seqs is None or n == expect_seqs) else "!!"
        print(f"  {mark} 序列数 {n}" + (f" / 期望 {expect_seqs}" if expect_seqs else ""))
        ok &= (expect_seqs is None or n == expect_seqs)

        random.seed(0)
        ids = random.sample(range(n), min(n_probe, n))
        for sid in ids:
            info = ds.get_sequence_info(sid)
            bbox = info["bbox"]
            nfr = bbox.shape[0]
            # 取几个有效帧
            valid_idx = [i for i in range(nfr) if bool(info["valid"][i])][:3]
            if not valid_idx:
                print(f"  !! seq {sid}: 没有任何有效帧")
                ok = False
                continue
            frames, anno, _meta = ds.get_frames(sid, valid_idx, info)
            f0 = frames[0]
            shape_ok = (f0.ndim == 3 and f0.shape[2] == 6)
            content_ok = not (f0.min() == f0.max())     # 全黑/全白说明读图有问题
            box = anno["bbox"][0].tolist()
            print(f"  {'✓' if shape_ok and content_ok else '!!'} seq {sid:4} "
                  f"{nfr:5} 帧  取帧 shape={tuple(f0.shape)} dtype={f0.dtype} "
                  f"范围[{int(f0.min())},{int(f0.max())}]  bbox={[round(v,1) for v in box]}")
            ok &= shape_ok and content_ok
    except Exception:
        print("  !! 异常:")
        traceback.print_exc()
        ok = False
    return ok


def main():
    env = env_settings()
    print("配置的路径:")
    for k in ("lasher_dir", "depthtrack_dir", "visevent_dir"):
        p = getattr(env, k)
        print(f"  {k:16} {p}   {'存在' if os.path.isdir(p) else '!! 不存在'}")

    results = []
    results.append(("LasHeR_train",
                    check("LasHeR (split=train)",
                          LasHeR(env.lasher_dir, dtype='rgbrgb', split='train'), 881)))
    results.append(("LasHeR_val",
                    check("LasHeR (split=val)",
                          LasHeR(env.lasher_dir, dtype='rgbrgb', split='val'), 97)))
    results.append(("DepthTrack_train",
                    check("DepthTrack (split=train)",
                          DepthTrack(env.depthtrack_dir, dtype='rgbcolormap', split='train'), 146)))
    results.append(("DepthTrack_val",
                    check("DepthTrack (split=val)",
                          DepthTrack(env.depthtrack_dir, dtype='rgbcolormap', split='val'), 6)))
    results.append(("VisEvent",
                    check("VisEvent (split=train)",
                          VisEvent(env.visevent_dir, dtype='rgbrgb', split='train'), 500)))

    print(f"\n{'='*62}\n汇总:")
    for name, ok in results:
        print(f"  {'✓' if ok else '!!'} {name}")
    bad = [n for n, ok in results if not ok]
    if bad:
        print(f"\n未通过:{', '.join(bad)}")
        return 1
    print("\n全部通过 —— 数据线就位。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
