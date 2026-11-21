#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LasHeR 文件名规范化工具(勘察优先,默认不改任何东西)

背景
----
SDSTrack 的 lib/train/dataset/lasher.py:89 硬编码了帧路径:

    os.path.join(seq_path, 'visible', '{:06d}.jpg'.format(frame_id))   # 从 0 开始

而同一文件第 88 行有原作者注释:
    "Note original filename is chaotic, we rename them"
说明官方下载的 LasHeR 原始文件名并非 000000.jpg,必须先规范化。

本脚本三种模式
--------------
    scan    (默认) 只勘察不改动:统计目录结构、文件名形态、帧数与标注行数是否一致
    apply   实际重命名为 000000.jpg 起的六位补零,并为每个序列写 .rename_map.json
    undo    按 .rename_map.json 回滚

安全设计
--------
  * 默认 scan,必须显式 --apply 才会动文件
  * 自然排序(数字感知):v2.jpg 排在 v10.jpg 前面,不会因字典序把帧顺序搞乱
  * 两段式重命名(先改成临时名再改成目标名),避免与已存在的目标名冲突
  * 帧数与 visible.txt 行数不一致的序列默认跳过,不猜
  * 每个序列写 .rename_map.json,可完整回滚

用法
----
    python lasher_rename.py --root D:\\...\\lasher\\trainingset
    python lasher_rename.py --root ... --sample 3        # 只看前 3 个序列的详细文件名
    python lasher_rename.py --root ... --apply
    python lasher_rename.py --root ... --undo
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

SUBDIRS = ("visible", "infrared")
ANNO = {"visible": "visible.txt", "infrared": "infrared.txt"}
MAP_NAME = ".rename_map.json"
TARGET_EXT = ".jpg"          # lasher.py 硬编码 .jpg
TARGET_FMT = "{:06d}"        # lasher.py 硬编码 {:06d},从 0 开始


# ---------------------------------------------------------------- 工具

_num_re = re.compile(r"(\d+)")


def natural_key(name):
    """数字感知排序:v2 < v10。字典序会给出 v10 < v2,那样帧顺序就毁了。"""
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(name)]


def describe_pattern(names):
    """把一组文件名归纳成形态描述,便于人眼判断。"""
    pats = Counter()
    for n in names:
        stem, ext = os.path.splitext(n)
        p = _num_re.sub("#", stem) + ext
        pats[p] += 1
    return pats


def is_already_target(names):
    """是否已经是 000000.jpg / 000001.jpg ... 的连续序列。"""
    if not names:
        return False
    want = [TARGET_FMT.format(i) + TARGET_EXT for i in range(len(names))]
    return sorted(names) == sorted(want)


def count_lines(path):
    if not os.path.exists(path):
        return None
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if line.strip():
                n += 1
    return n


def list_seqs(root):
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )


# ---------------------------------------------------------------- 勘察

def inspect_seq(root, seq):
    """返回一个序列的勘察结果。"""
    p = os.path.join(root, seq)
    info = {"seq": seq, "path": p, "problems": [], "dirs": {}}

    for sub in SUBDIRS:
        d = os.path.join(p, sub)
        if not os.path.isdir(d):
            info["problems"].append(f"缺少子目录 {sub}/")
            info["dirs"][sub] = None
            continue
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        files.sort(key=natural_key)
        exts = Counter(os.path.splitext(f)[1].lower() for f in files)
        info["dirs"][sub] = {
            "count": len(files),
            "exts": dict(exts),
            "first5": files[:5],
            "last3": files[-3:],
            "patterns": dict(describe_pattern(files).most_common(3)),
            "already": is_already_target(files),
            "files": files,
        }

    v, i = info["dirs"].get("visible"), info["dirs"].get("infrared")
    if v and i and v["count"] != i["count"]:
        info["problems"].append(
            f"visible({v['count']}) 与 infrared({i['count']}) 帧数不一致")

    for sub in SUBDIRS:
        anno = os.path.join(p, ANNO[sub])
        n = count_lines(anno)
        info.setdefault("anno", {})[sub] = n
        if n is None:
            # 训练端只读 visible.txt,infrared.txt 缺失只是警告
            lvl = "问题" if sub == "visible" else "提醒"
            info["problems"].append(f"[{lvl}] 缺少 {ANNO[sub]}")
        elif info["dirs"].get(sub) and info["dirs"][sub]["count"] != n:
            info["problems"].append(
                f"{sub}/ 有 {info['dirs'][sub]['count']} 帧,"
                f"但 {ANNO[sub]} 有 {n} 行 —— 对不上")

    non_jpg = []
    for sub in SUBDIRS:
        d = info["dirs"].get(sub)
        if d:
            non_jpg += [e for e in d["exts"] if e != TARGET_EXT]
    if non_jpg:
        info["problems"].append(
            f"存在非 .jpg 扩展名 {sorted(set(non_jpg))} —— "
            f"lasher.py 硬编码 .jpg,需要先转码或改配置")

    return info


def cmd_scan(root, sample):
    seqs = list_seqs(root)
    print(f"根目录: {root}")
    print(f"序列数: {len(seqs)}\n")
    if not seqs:
        print("!! 没找到任何序列目录,确认路径是否指向 trainingset/ 那一层")
        return 1

    agg_pat, agg_ext = Counter(), Counter()
    n_ready = n_problem = 0
    total_frames = 0
    problems = []

    for seq in seqs:
        info = inspect_seq(root, seq)
        v = info["dirs"].get("visible")
        if v:
            agg_pat.update(v["patterns"])
            agg_ext.update(v["exts"])
            total_frames += v["count"]
            if v["already"]:
                n_ready += 1
        hard = [p for p in info["problems"] if not p.startswith("[提醒]")]
        if hard:
            n_problem += 1
            problems.append((seq, hard))

    print("=" * 70)
    print(f"总帧数(visible)  : {total_frames:,}")
    print(f"已是目标命名的序列: {n_ready} / {len(seqs)}")
    print(f"有问题的序列      : {n_problem}")
    print(f"\nvisible/ 扩展名分布 : {dict(agg_ext)}")
    print(f"visible/ 文件名形态 (# 代表数字):")
    for pat, cnt in agg_pat.most_common(8):
        print(f"    {pat:<30} {cnt:,} 个文件")

    print("\n" + "=" * 70)
    print(f"前 {sample} 个序列的实际文件名:")
    for seq in seqs[:sample]:
        info = inspect_seq(root, seq)
        print(f"\n  [{seq}]")
        for sub in SUBDIRS:
            d = info["dirs"].get(sub)
            if d is None:
                print(f"    {sub}/  (不存在)")
                continue
            print(f"    {sub}/  {d['count']} 个文件, 扩展名 {dict(d['exts'])}")
            print(f"      前5: {d['first5']}")
            print(f"      末3: {d['last3']}")
            print(f"      已符合目标命名: {'是' if d['already'] else '否'}")
        print(f"    标注行数: {info.get('anno')}")
        if info["problems"]:
            for p in info["problems"]:
                print(f"    ! {p}")

    if problems:
        print("\n" + "=" * 70)
        print(f"有硬性问题的序列(前 20 个):")
        for seq, ps in problems[:20]:
            print(f"  {seq}")
            for p in ps:
                print(f"      ! {p}")
        if len(problems) > 20:
            print(f"  ... 其余 {len(problems)-20} 个略")

    print("\n" + "=" * 70)
    if n_ready == len(seqs):
        print("结论:全部序列已符合 SDSTrack 要求的命名,不需要重命名。")
    elif n_problem == 0:
        print("结论:结构一致、帧数与标注对得上,可以安全执行:")
        print(f"    python {os.path.basename(__file__)} --root \"{root}\" --apply")
    else:
        print("结论:有序列存在硬性问题(见上),先把这些问题弄清楚再 --apply。")
        print("      --apply 会自动跳过有问题的序列,不会硬改。")
    return 0


# ---------------------------------------------------------------- 执行

def rename_dir(d, files, dry=False):
    """两段式重命名,避免目标名与现有名冲突。返回 {旧名: 新名}。"""
    mapping = {}
    plan = []
    for idx, old in enumerate(files):
        new = TARGET_FMT.format(idx) + TARGET_EXT
        plan.append((old, new))
        mapping[old] = new
    if dry or all(o == n for o, n in plan):
        return mapping if not all(o == n for o, n in plan) else {}

    staged = []
    try:
        # 第一段:全部改成不可能冲突的临时名
        for old, _ in plan:
            os.rename(os.path.join(d, old), os.path.join(d, "__tmp__" + old))
            staged.append(old)
        # 第二段:临时名 -> 目标名
        while staged:
            old = staged[-1]
            new = mapping[old]
            os.rename(os.path.join(d, "__tmp__" + old), os.path.join(d, new))
            staged.pop()
    except Exception:
        # 中途失败:把还挂着 __tmp__ 前缀的文件恢复原名,避免留下半成品
        for old in staged:
            tmp = os.path.join(d, "__tmp__" + old)
            if os.path.exists(tmp):
                try:
                    os.rename(tmp, os.path.join(d, old))
                except OSError:
                    pass
        raise
    return mapping


def cmd_apply(root):
    seqs = list_seqs(root)
    done = skipped = already = 0
    for n, seq in enumerate(seqs, 1):
        info = inspect_seq(root, seq)
        hard = [p for p in info["problems"] if not p.startswith("[提醒]")]
        if hard:
            print(f"[{n}/{len(seqs)}] 跳过 {seq}: {hard[0]}")
            skipped += 1
            continue
        if all(info["dirs"][s]["already"] for s in SUBDIRS if info["dirs"].get(s)):
            already += 1
            continue

        seq_map = {}
        for sub in SUBDIRS:
            d = info["dirs"].get(sub)
            if not d:
                continue
            m = rename_dir(os.path.join(root, seq, sub), d["files"])
            if m:
                seq_map[sub] = m
        if seq_map:
            with open(os.path.join(root, seq, MAP_NAME), "w", encoding="utf-8") as fp:
                json.dump(seq_map, fp, ensure_ascii=False, indent=1)
            done += 1
            if done % 50 == 0:
                print(f"[{n}/{len(seqs)}] 已处理 {done} 个序列")

    print(f"\n完成:重命名 {done},已符合 {already},跳过 {skipped}")
    if skipped:
        print("跳过的序列有硬性问题,请先用 scan 模式查看原因。")
    return 0


def cmd_undo(root):
    seqs = list_seqs(root)
    n = 0
    for seq in seqs:
        mp = os.path.join(root, seq, MAP_NAME)
        if not os.path.exists(mp):
            continue
        with open(mp, "r", encoding="utf-8") as fp:
            seq_map = json.load(fp)
        for sub, m in seq_map.items():
            d = os.path.join(root, seq, sub)
            for old, new in m.items():
                os.rename(os.path.join(d, new), os.path.join(d, "__tmp__" + old))
            for old, new in m.items():
                os.rename(os.path.join(d, "__tmp__" + old), os.path.join(d, old))
        os.remove(mp)
        n += 1
    print(f"已回滚 {n} 个序列")
    return 0


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(description="LasHeR 文件名规范化(默认只勘察)")
    ap.add_argument("--root", required=True,
                    help=r"LasHeR 序列根目录,例如 D:\...\lasher\trainingset")
    ap.add_argument("--apply", action="store_true", help="实际执行重命名")
    ap.add_argument("--undo", action="store_true", help="按 .rename_map.json 回滚")
    ap.add_argument("--sample", type=int, default=3, help="scan 模式展示几个序列的细节")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"!! 目录不存在: {args.root}")
        return 2
    if args.undo:
        return cmd_undo(args.root)
    if args.apply:
        return cmd_apply(args.root)
    return cmd_scan(args.root, args.sample)


if __name__ == "__main__":
    sys.exit(main())
