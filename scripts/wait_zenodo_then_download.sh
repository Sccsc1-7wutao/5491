#!/usr/bin/env bash
# Zenodo 整站 504 时用:轮询到它恢复,立刻自动续传 DepthTrack。
# 2026-09-10 早上 Zenodo 全站故障(连首页都 504),写这个免得人工盯。
export PYTHONIOENCODING=utf-8   # 否则 GBK 控制台会在打印日志时抛 UnicodeEncodeError
PY="D:/Conda/envs/sdstrack/python.exe"
DL="D:/cvresearch/scripts/download_depthtrack.py"
OUT="D:/cvresearch/data/_zips/depthtrack"

check() {  # 用 python 走代理探测,curl 在这台机器上不吃代理环境变量
  "$PY" -B -c "
import sys, requests
try:
    r = requests.get('https://zenodo.org/api/records/5792146', timeout=(10, 30))
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

n=0
until check; do
  n=$((n+1))
  echo "[$(date '+%H:%M:%S')] 第 $n 次探测:Zenodo 仍不可用,5 分钟后重试"
  sleep 300
done
echo "[$(date '+%H:%M:%S')] Zenodo 恢复,开始续传"
"C:/Users/A/anaconda3/python.exe" "$DL" --workers 2 --out "$OUT"
