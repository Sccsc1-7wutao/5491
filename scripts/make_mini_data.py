#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成三个数据集的极小伪造版本(Step 0.5),用于在真实数据到位前验证整条链路。

为什么需要
----------
真实数据合计约 500 GB,下载要几小时。但"dataset 能不能实例化、
DataLoader 能不能吐出 (B, 6, H, W)、模型能不能吃下去"这些问题
不需要真实数据就能答。用噪声图先把链路跑通,真数据到了直接换路径。

三个数据集的格式差异(全部从源码核对,不是猜的)
-----------------------------------------------
LasHeR      lib/train/dataset/lasher.py
    <seq>/visible/*.jpg           !! SDSTrack 用 glob('*{frame_id}.jpg')[0] 取帧,
    <seq>/infrared/*.jpg             不是 ViPT 那种硬编码 '{:06d}.jpg'。
                                     这里仍按 6 位补零生成 —— 补零后字典序 = 数值序,
                                     glob 的 [0] 才稳定拿到正确那一帧。
    <seq>/visible.txt             每帧一行 "x,y,w,h" 逗号分隔
    <seq>/infrared.txt            训练端不读,测试端要读
    有效帧判据: w > 0 且 h > 0
    读图 dtype='rgbrgb'

DepthTrack  lib/train/dataset/depthtrack.py
    <seq>/color/{:08d}.jpg        !! 从 1 开始
    <seq>/depth/{:08d}.png        !! 从 1 开始,16-bit 单通道
    <seq>/groundtruth.txt
    有效帧判据: w > 10 且 h > 10   !! 阈值不是 0
    序列名必须含下划线(_build_class_list 用 split('_')[0] 取类名)
    读图 dtype='rgbcolormap', depth_clip=True

VisEvent    lib/train/dataset/visevent.py
    <seq>/vis_imgs/*.bmp          sorted() 排序,文件名不要求补零
    <seq>/event_imgs/*.bmp        必须和 vis_imgs 同名
    <seq>/groundtruth.txt
    <seq>/absent_label.txt        每帧一行 0/1
    <root>/trainlist.txt          序列清单放在根目录
    有效帧判据: w > 5 且 h > 5
    读图 dtype='rgbrgb'

序列名一律从 SDSTrack 官方 data_specs 清单里取,不能自己起名 ——
dataset 类是按清单读目录的,名字对不上会直接找不到。

用法
----
    python make_mini_data.py                       # 三个数据集各 4 序列 × 20 帧
    python make_mini_data.py --seqs 2 --frames 10
    python make_mini_data.py --datasets lasher
    python make_mini_data.py --verify              # 生成后立即实例化验证
"""

import argparse
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv-python。请用 sdstrack 环境跑:\n"
             r"  D:\Conda\envs\sdstrack\python.exe scripts\make_mini_data.py")

SDSTRACK = r"D:\cvresearch\SDSTrack"
SPECS = os.path.join(SDSTRACK, "lib", "train", "data_specs")
DEFAULT_OUT = r"D:\cvresearch\data\_mini"

H, W = 256, 320          # 伪造图尺寸,够大以免裁剪逻辑踩边界
BOX_WH = 60              # 框边长,同时满足 >0 / >10 / >5 三种判据


def seq_names(spec_file, n):
    """从官方清单里取前 n 个序列名。"""
    p = os.path.join(SPECS, spec_file)
    with open(p, "r", encoding="utf-8") as fp:
        names = [ln.strip() for ln in fp if ln.strip()]
    if len(names) < n:
        raise SystemExit(f"{spec_file} 只有 {len(names)} 条,要不了 {n} 个")
    return names[:n]


def boxes(n, rng):
    """生成 n 帧的 (x, y, w, h),框在图内平滑游走。"""
    out = []
    x, y = W // 2 - BOX_WH // 2, H // 2 - BOX_WH // 2
    for _ in range(n):
        x = int(np.clip(x + rng.integers(-6, 7), 0, W - BOX_WH - 1))
        y = int(np.clip(y + rng.integers(-6, 7), 0, H - BOX_WH - 1))
        out.append((x, y, BOX_WH, BOX_WH))
    return out


def rgb_noise(rng):
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def imwrite(path, img):
    """cv2.imwrite 失败时返回 False 而不抛异常 —— 必须自己检查。

    最典型的失败原因是**路径含非 ASCII 字符**:OpenCV 在 Windows 上走 ANSI API,
    路径里有中文就静默写不进去(读同理,imread 返回 None)。
    2026-09-09 就是这么踩的:项目原本放在 D:\\Downloads\\cv<中文>\\ 下,
    生成器"成功"造出一堆空目录,直到 dataset 实例化才暴露。
    项目因此整体搬到了 D:\\cvresearch\\。
    """
    if not cv2.imwrite(path, img):
        raise IOError(
            "cv2.imwrite 失败: %s\n"
            "  最常见原因:路径含非 ASCII 字符(OpenCV 在 Windows 上不支持)。\n"
            "  其次:目录不存在 / 磁盘满 / 无写权限。" % path
        )


def write_txt(path, rows, fmt="{:.2f},{:.2f},{:.2f},{:.2f}"):
    with open(path, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(fmt.format(*r) + "\n")


# ---------------------------------------------------------------- LasHeR

def gen_lasher(root, n_seq, n_frame, rng):
    names = seq_names("lasher_train.txt", n_seq)
    for name in names:
        d = os.path.join(root, name)
        os.makedirs(os.path.join(d, "visible"), exist_ok=True)
        os.makedirs(os.path.join(d, "infrared"), exist_ok=True)
        for i in range(n_frame):
            imwrite(os.path.join(d, "visible", f"{i:06d}.jpg"), rgb_noise(rng))
            imwrite(os.path.join(d, "infrared", f"{i:06d}.jpg"), rgb_noise(rng))
        bb = boxes(n_frame, rng)
        write_txt(os.path.join(d, "visible.txt"), bb)
        write_txt(os.path.join(d, "infrared.txt"), bb)   # 测试端要读
    return names


# ---------------------------------------------------------------- DepthTrack

def gen_depthtrack(root, n_seq, n_frame, rng):
    names = seq_names("depthtrack_train.txt", n_seq)
    for name in names:
        d = os.path.join(root, name)
        os.makedirs(os.path.join(d, "color"), exist_ok=True)
        os.makedirs(os.path.join(d, "depth"), exist_ok=True)
        for i in range(n_frame):
            # 注意:帧号从 1 开始
            imwrite(os.path.join(d, "color", f"{i+1:08d}.jpg"), rgb_noise(rng))
            # 16-bit 单通道深度图,值域模拟毫米(0~10000)
            depth = rng.integers(500, 8000, (H, W), dtype=np.uint16)
            imwrite(os.path.join(d, "depth", f"{i+1:08d}.png"), depth)
        write_txt(os.path.join(d, "groundtruth.txt"), boxes(n_frame, rng))
    return names


# ---------------------------------------------------------------- VisEvent

def gen_visevent(root, n_seq, n_frame, rng):
    # VisEvent 没有官方 data_specs 清单,序列名由根目录的 trainlist.txt 决定,
    # 所以这里可以自己起名(和 LasHeR/DepthTrack 必须照抄官方清单不同)。
    names = [f"{i:05d}_fake_seq" for i in range(n_seq)]
    for name in names:
        d = os.path.join(root, name)
        os.makedirs(os.path.join(d, "vis_imgs"), exist_ok=True)
        os.makedirs(os.path.join(d, "event_imgs"), exist_ok=True)
        for i in range(n_frame):
            fn = f"frame{i:04d}.bmp"
            imwrite(os.path.join(d, "vis_imgs", fn), rgb_noise(rng))
            imwrite(os.path.join(d, "event_imgs", fn), rgb_noise(rng))
        write_txt(os.path.join(d, "groundtruth.txt"), boxes(n_frame, rng))
        with open(os.path.join(d, "absent_label.txt"), "w", encoding="utf-8") as fp:
            for _ in range(n_frame):
                fp.write("1\n")          # 1 = 可见
    # trainlist.txt 在根目录,不在序列里
    with open(os.path.join(root, "trainlist.txt"), "w", encoding="utf-8") as fp:
        for name in names:
            fp.write(name + "\n")
    return names


# ---------------------------------------------------------------- 验证

def verify(out, which):
    sys.path.insert(0, SDSTRACK)
    os.chdir(SDSTRACK)
    ok = True

    if "lasher" in which:
        from lib.train.dataset.lasher import LasHeR
        ds = LasHeR(os.path.join(out, "lasher"), dtype="rgbrgb", split="train")
        print(f"  LasHeR      : {ds.get_num_sequences()} 个序列(清单里有 881,"
              f"磁盘上只造了几个,所以取样只能落在造过的那几个上)")
        f, a, _ = ds.get_frames(0, [0, 1])
        print(f"                frame {f[0].shape} {f[0].dtype}  bbox {a['bbox'][0].tolist()}")
        ok &= f[0].shape == (256, 320, 6)

    if "depthtrack" in which:
        from lib.train.dataset.depthtrack import DepthTrack
        ds = DepthTrack(os.path.join(out, "depthtrack"), dtype="rgbcolormap", split="train")
        f, a, _ = ds.get_frames(0, [0, 1])
        print(f"  DepthTrack  : frame {f[0].shape} {f[0].dtype}  bbox {a['bbox'][0].tolist()}")
        ok &= f[0].shape == (256, 320, 6)

    if "visevent" in which:
        from lib.train.dataset.visevent import VisEvent
        ds = VisEvent(os.path.join(out, "visevent"), dtype="rgbrgb", split="train")
        f, a, _ = ds.get_frames(0, [0, 1])
        print(f"  VisEvent    : frame {f[0].shape} {f[0].dtype}  bbox {a['bbox'][0].tolist()}")
        ok &= f[0].shape == (256, 320, 6)

    return ok


def main():
    ap = argparse.ArgumentParser(description="生成三个数据集的极小伪造版本")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--datasets", default="lasher,depthtrack,visevent")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true", help="生成后实例化 dataset 验证")
    args = ap.parse_args()

    which = [s.strip() for s in args.datasets.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    print(f"输出目录: {args.out}")
    print(f"规格    : {args.seqs} 序列 × {args.frames} 帧,图 {W}×{H},框 {BOX_WH}×{BOX_WH}\n")

    if "lasher" in which:
        r = os.path.join(args.out, "lasher")
        os.makedirs(r, exist_ok=True)
        print(f"LasHeR     -> {gen_lasher(r, args.seqs, args.frames, rng)}")
    if "depthtrack" in which:
        r = os.path.join(args.out, "depthtrack")
        os.makedirs(r, exist_ok=True)
        print(f"DepthTrack -> {gen_depthtrack(r, args.seqs, args.frames, rng)}")
    if "visevent" in which:
        r = os.path.join(args.out, "visevent")
        os.makedirs(r, exist_ok=True)
        print(f"VisEvent   -> {gen_visevent(r, args.seqs, args.frames, rng)}")

    if args.verify:
        print("\n=== 实例化验证(期望每帧 (256, 320, 6) uint8)===")
        ok = verify(args.out, which)
        print("\n" + ("全部通过" if ok else "!! 有不符合预期的输出,见上"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
