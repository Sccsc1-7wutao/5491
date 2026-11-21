#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DepthTrack 数据集批量下载器(Zenodo 直连,无需登录/会员)

三个 record:
    5794115  训练集①  100 个序列  37.8 GB
    5837926  训练集②   52 个序列  14.6 GB
    5792146  测试集     50 个序列  18.3 GB
                              合计  70.7 GB

特性:
    - 断点续传(HTTP Range),中断后重跑自动接着下
    - 下完校验 MD5,不一致自动重下
    - 已完整下载的文件直接跳过
    - 失败自动重试,多线程并行

用法:
    python download_depthtrack.py --list                 # 只列清单,不下载
    python download_depthtrack.py                        # 下载全部到默认目录
    python download_depthtrack.py --only train           # 只下训练集(①+②)
    python download_depthtrack.py --only test            # 只下测试集
    python download_depthtrack.py --workers 2            # 并发数(默认 2,别调高)
    python download_depthtrack.py --out D:\\path\\to\\dir  # 指定输出目录

关于并发
--------
Zenodo 对未认证请求限流(约 60 次/分钟)。2026-09-10 用 4 worker 跑了 7.7 小时,
吃了 883 个 504,202 个文件里 77 个耗尽重试次数放弃。
调高并发只会让限流更狠、越重试越慢。**默认 2 是实测后定的,不要加。**
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---------------------------------------------------------------- 配置

RECORDS = {
    "train_part1": {"id": "5794115", "desc": "训练集① 100 序列 ~37.8 GB"},
    "train_part2": {"id": "5837926", "desc": "训练集② 52 序列 ~14.6 GB"},
    "test":        {"id": "5792146", "desc": "测试集   50 序列 ~18.3 GB"},
}

API = "https://zenodo.org/api/records/{}"
DEFAULT_OUT = r"D:\cvresearch\data\_zips\depthtrack"
# Zenodo 清单缓存目录。清单内容固定不变,取到一次就存下来,
# 之后 API 挂了也能照常下载。
CACHE_DIR = r"D:\cvresearch\data\_zips\_manifest_cache"

CHUNK = 1024 * 1024          # 1 MB
# 读取超时给到 300s:与百度网盘客户端抢带宽时单连接会掉到 100 KB/s 上下,
# 120s 会误判为超时,反而把还活着的连接掐掉重来,越重试越慢。
TIMEOUT = (30, 300)          # (连接超时, 读取超时)
MAX_RETRY = 12

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------- 元数据

def fetch_manifest(record_id):
    """从 Zenodo API 取文件清单。兼容新旧两种 API 返回结构。

    Zenodo 经常返回 502/504(尤其从国内访问),所以这里必须带重试。
    """
    # 清单内容是固定不变的,所以成功取到一次就落盘缓存。
    # 2026-09-10 早上 Zenodo API 持续 504(疑似限流),导致每一轮都卡在取清单、
    # 一个字节都下不动。有缓存就不再受 API 死活影响。
    cache = os.path.join(CACHE_DIR, f"manifest_{record_id}.json")

    data = None
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(API.format(record_id), timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache, "w", encoding="utf-8") as fp:
                    json.dump(data, fp)
            except Exception as ce:
                log(f"  (清单缓存写入失败,不影响本次下载: {ce})")
            break
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            log(f"  取清单第 {attempt}/{MAX_RETRY} 次失败: "
                f"{type(e).__name__}: {e} — {wait}s 后重试")
            time.sleep(wait)

    if data is None and os.path.exists(cache):
        try:
            with open(cache, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            log(f"  API 取不到,改用本地缓存清单 {os.path.basename(cache)}")
        except Exception as e:
            log(f"  缓存清单也读不了: {type(e).__name__}: {e}")

    if data is None:
        raise RuntimeError(f"取清单重试 {MAX_RETRY} 次仍失败、且无本地缓存: {last_err}")

    files = data.get("files") or []
    if isinstance(files, dict):          # 新版 API 可能包一层
        files = files.get("entries") or []
        if isinstance(files, dict):
            files = [dict(v, key=k) for k, v in files.items()]

    out = []
    for f in files:
        key = f.get("key") or f.get("filename")
        size = f.get("size") or f.get("filesize") or 0
        checksum = f.get("checksum") or ""
        if checksum.startswith("md5:"):
            checksum = checksum[4:]
        links = f.get("links") or {}
        # 两条下载路径轮换用:
        #   api_url  = /api/records/.../content   —— API 路径,限流更严
        #   web_url  = /records/.../files/...?download=1 —— 网页下载路径
        # 2026-09-10 实测:4 worker 跑 API 路径会被 Zenodo 限流,
        # 7.7 小时里吃了 883 个 504。失败时换另一条路走。
        api_url = links.get("content") or links.get("self") or links.get("download")
        web_url = f"https://zenodo.org/records/{record_id}/files/{key}?download=1"
        urls = [u for u in (api_url, web_url) if u] or [web_url]
        out.append({"key": key, "size": int(size), "md5": checksum,
                    "url": urls[0], "urls": urls})
    return out


# ---------------------------------------------------------------- 校验

def md5_of(path, expect_size=None):
    h = hashlib.md5()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_unlink(p):
    """删掉文件;Windows 上偶尔被占用,退让几次再放弃(放弃也不抛,由调用方决定)。"""
    for k in range(6):
        try:
            if os.path.exists(p):
                os.remove(p)
            return True
        except PermissionError:
            time.sleep(1.5 * (k + 1))
    return False


def already_ok(path, meta, verify_md5=True):
    """判断本地文件是否已经完整。"""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) != meta["size"]:
        return False
    if verify_md5 and meta["md5"]:
        return md5_of(path) == meta["md5"]
    return True


# ---------------------------------------------------------------- 下载

def download_one(meta, out_dir, idx, total):
    path = os.path.join(out_dir, meta["key"])
    tmp = path + ".part"

    if already_ok(path, meta):
        log(f"[{idx}/{total}] 跳过(已完整) {meta['key']}")
        return "skip", meta["size"]

    for attempt in range(1, MAX_RETRY + 1):
        try:
            done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if done > meta["size"]:      # 本地比远端还大,说明坏了,重来
                os.remove(tmp)
                done = 0

            headers = {}
            mode = "wb"
            if done > 0:
                headers["Range"] = f"bytes={done}-"
                mode = "ab"

            # 每次重试换一条下载路径,避开单条路径上的限流
            url = meta.get("urls", [meta["url"]])[(attempt - 1) % len(meta.get("urls", [1]))]

            with requests.get(url, headers=headers, stream=True,
                              timeout=TIMEOUT, allow_redirects=True) as r:
                r.raise_for_status()
                if done > 0:
                    # 续传时必须确认服务端真的从我们要的偏移开始发。
                    # 只判断 status==200 不够:某些代理/CDN 会返回 206 却从 0 开始,
                    # 于是一份完整响应被追加到已有分片后面 —— 表现为"本地比远端还大"。
                    # 2026-09-09 夜里 82 个文件就是栽在这里。
                    cr = r.headers.get("Content-Range", "")
                    if not (r.status_code == 206 and cr.startswith(f"bytes {done}-")):
                        log(f"    [{idx}/{total}] {meta['key']} 服务端未按请求续传 "
                            f"(status={r.status_code}, Content-Range={cr!r}),从头下")
                        done, mode = 0, "wb"

                start_t, start_b, last = time.time(), done, time.time()
                with open(tmp, mode) as fp:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fp.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last > 20:
                            spd = (done - start_b) / max(now - start_t, 1e-6)
                            pct = done / meta["size"] * 100 if meta["size"] else 0
                            log(f"    [{idx}/{total}] {meta['key']}  "
                                f"{pct:5.1f}%  {human(done)}/{human(meta['size'])}  "
                                f"{human(spd)}/s")
                            last = now

            # 校验不通过一律立刻删掉 .part —— 留着它下一轮会被当成可续传的起点,
            # 于是坏数据被反复沿用,重试多少次都不会好。
            if meta["size"] and os.path.getsize(tmp) != meta["size"]:
                bad = os.path.getsize(tmp)
                _safe_unlink(tmp)
                raise IOError(f"大小不符 {bad} != {meta['size']},已删除重下")

            if meta["md5"]:
                got = md5_of(tmp)
                if got != meta["md5"]:
                    _safe_unlink(tmp)
                    raise IOError(f"MD5 不符 {got} != {meta['md5']},已删除重下")

            # Windows 上文件刚写完偶尔仍被占用(WinError 32),重试几次即可
            for k in range(6):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if k == 5:
                        raise
                    time.sleep(1.5 * (k + 1))
            log(f"[{idx}/{total}] [OK] {meta['key']}  {human(meta['size'])}")
            return "ok", meta["size"]

        except Exception as e:
            # 504 是 Zenodo 侧限流,不是网络抖动 —— 退避要足够长才有意义。
            # 上限提到 300s,并加 ±25% 抖动,避免多个 worker 同时重试形成新的冲击。
            wait = min(2 ** attempt, 300) * random.uniform(0.75, 1.25)
            log(f"[{idx}/{total}] [X] {meta['key']} 第 {attempt}/{MAX_RETRY} 次失败: "
                f"{type(e).__name__}: {e} — {wait:.0f}s 后重试")
            time.sleep(wait)

    log(f"[{idx}/{total}] !! {meta['key']} 重试 {MAX_RETRY} 次仍失败,已放弃")
    return "fail", 0


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="DepthTrack (Zenodo) 批量下载器")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    ap.add_argument("--only", choices=["train", "test", "all"], default="all",
                    help="只下训练集 / 只下测试集 / 全部(默认)")
    ap.add_argument("--workers", type=int, default=2, help="并行线程数(默认 2;Zenodo 限流,别调高)")
    ap.add_argument("--list", action="store_true", help="只列清单,不下载")
    args = ap.parse_args()

    if args.only == "train":
        want = ["train_part1", "train_part2"]
    elif args.only == "test":
        want = ["test"]
    else:
        want = list(RECORDS.keys())

    all_jobs, grand_total = [], 0
    manifest_failed = []          # 取清单失败的 record —— 绝不能当成"没有文件要下"
    for name in want:
        rec = RECORDS[name]
        log(f"\n=== {name}  (record {rec['id']})  {rec['desc']} ===")
        try:
            files = fetch_manifest(rec["id"])
        except Exception as e:
            log(f"  !! 取清单失败: {type(e).__name__}: {e}")
            manifest_failed.append(name)
            continue
        size = sum(f["size"] for f in files)
        grand_total += size
        log(f"  {len(files)} 个文件,合计 {human(size)}")
        sub = os.path.join(args.out, name)
        for f in files:
            all_jobs.append((f, sub))

    log(f"\n总计 {len(all_jobs)} 个文件,{human(grand_total)}")

    if args.list:
        log("\n(--list 模式,不下载)")
        for f, sub in all_jobs[:10]:
            log(f"  {f['key']:<45} {human(f['size']):>10}  md5={f['md5'][:8] or '(无)'}")
        if len(all_jobs) > 10:
            log(f"  ... 其余 {len(all_jobs) - 10} 个略")
        return 0

    for _, sub in all_jobs:
        os.makedirs(sub, exist_ok=True)

    total = len(all_jobs)
    stats = {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, f, sub, i + 1, total): f
                for i, (f, sub) in enumerate(all_jobs)}
        for fu in as_completed(futs):
            status, _ = fu.result()
            stats[status] += 1

    dt = time.time() - t0
    log(f"\n{'='*60}")
    log(f"完成:成功 {stats['ok']}  跳过 {stats['skip']}  失败 {stats['fail']}")
    log(f"耗时 {dt/60:.1f} 分钟")

    # 只有"清单全取到了、且每个文件都成功或已完整"才算成功。
    # 2026-09-09 夜里踩过:Zenodo API 连续 504,三个 record 的清单一个都没取到,
    # all_jobs 为空 → fail 计数为 0 → 脚本返回 0 报"全部完成",守护进程信了就退出,
    # 实际只下了 66%。取不到清单 ≠ 没有文件要下。
    if manifest_failed:
        log(f"!! 以下 record 的清单没取到,其文件一个都没下:{', '.join(manifest_failed)}")
        log("   这不是完成,是失败。重跑本脚本会重新取清单。")
        return 1
    if stats["fail"]:
        log(f"有 {stats['fail']} 个文件失败,直接重跑本脚本会自动续传/重下。")
        return 1
    if total == 0:
        log("!! 清单为空,没有任何文件可下 —— 判为失败,不是完成。")
        return 1

    log(f"全部 {total} 个文件已下载并通过 MD5 校验 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
