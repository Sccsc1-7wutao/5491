# Back up the irreplaceable cloud artifacts to the local disk (idempotent: existing files are skipped; nothing is deleted anywhere).
#  1) per-run training records from instance A: round_metrics.jsonl, run_config.json, param_group_stats.jsonl (small)
#  2) every *.pth.tar that actually exists on the shared disk /mnt/cvr_eval (~370 MB each), discovered by listing - no guessing of file names
# Usage: powershell -File cloud_backup.ps1 [-WithCkpt]
param([switch]$WithCkpt, [switch]$AllRunCkpts)
$ROOTD = 'D:\cvresearch\cloud_backup'; $LOG = "$ROOTD\backup.log"
$A = 'root@px1-jcy.matpool.com'; $PORT = '27460'
$SO = @('-o', 'ConnectTimeout=25', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=8')
New-Item -ItemType Directory -Force "$ROOTD\A_meta", "$ROOTD\ckpt" | Out-Null
function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding utf8 }
function Retry($sb) { for ($i = 0; $i -lt 6; $i++) { & $sb; if ($LASTEXITCODE -eq 0) { return $true }; Start-Sleep 30 }; return $false }

# 1) metadata of every run directory that has a round_metrics.jsonl
$ok = Retry { ssh @SO -p $PORT $A 'cd /root/cvresearch/output && tar czf /tmp/cvr_meta.tgz $(ls -d */ | tr -d / | while read d; do for f in round_metrics.jsonl run_config.json param_group_stats.jsonl; do [ -f $d/$f ] && echo $d/$f; done; done)' }
if ($ok) {
  $ok = Retry { scp -q @SO -P $PORT "${A}:/tmp/cvr_meta.tgz" "$ROOTD\A_meta.tgz" }
  if ($ok) { tar -xzf "$ROOTD\A_meta.tgz" -C "$ROOTD\A_meta"; Log "meta pulled: $((Get-ChildItem "$ROOTD\A_meta" -Recurse -File | Measure).Count) files" }
} else { Log "FAILED meta tar" }

# 2) checkpoints: list what exists, then copy each
if ($WithCkpt) {
  $files = @()
  for ($i = 0; $i -lt 8 -and $files.Count -eq 0; $i++) {
    $out = ssh @SO -p $PORT $A 'for d in /mnt/cvr_eval/ckpt_*/; do for f in $d*.pth.tar; do [ -f "$f" ] && echo "$(basename $d)/$(basename $f)"; done; done'
    if ($LASTEXITCODE -eq 0) { $files = @($out -split "`n" | Where-Object { $_ -match '^ckpt_.+/.+\.pth\.tar$' }) }
    if ($files.Count -eq 0) { Start-Sleep 30 }
  }
  if ($files.Count -eq 0) { Log "FAILED to list /mnt/cvr_eval (ssh throttled?) - no checkpoints backed up this pass" } else { Log "found $($files.Count) checkpoint files" }
  foreach ($rel in $files) {
    $name = $rel -replace '/', '__'
    $target = "$ROOTD\ckpt\$name"          # NB: PowerShell variables are case-insensitive - never reuse a name that differs only by case
    if (Test-Path $target) { continue }
    $ok = Retry { scp -q @SO -P $PORT "${A}:/mnt/cvr_eval/$rel" "$target.part" }
    if ($ok) { Move-Item "$target.part" $target -Force; Log "ckpt $rel ok" } else { Log "FAILED ckpt $rel" }
  }
}
# 3) every final (ep0120) checkpoint of every real run under A's output dir (smoke/probe/c3 runs excluded); size-verified against the cloud listing
if ($AllRunCkpts) {
  New-Item -ItemType Directory -Force "$ROOTD\run_ckpt" | Out-Null
  $items = @()
  for ($i = 0; $i -lt 8 -and $items.Count -eq 0; $i++) {
    $out = ssh @SO -p $PORT $A 'cd /root/cvresearch/output; for f in */checkpoints/*ep0120.pth.tar; do case $f in _smoke*|probe*|c3_*) continue;; esac; [ -f "$f" ] && echo "$(stat -c %s "$f") $f"; done'
    if ($LASTEXITCODE -eq 0) { $items = @($out -split "`n" | Where-Object { $_ -match '^\d+ [^ ]+/checkpoints/[^ ]+\.pth\.tar$' }) }
    if ($items.Count -eq 0) { Start-Sleep 30 }
  }
  if ($items.Count -eq 0) { Log "FAILED to list run checkpoints (ssh throttled?)" } else { Log "run checkpoints found: $($items.Count)" }
  $okn = 0; $failn = 0
  foreach ($it in $items) {
    $size = [int64]($it -split ' ')[0]; $rel = ($it -split ' ', 2)[1]
    $target = "$ROOTD\run_ckpt\" + ($rel -replace '/checkpoints/', '__' -replace '/', '__')
    if ((Test-Path $target) -and ((Get-Item $target).Length -eq $size)) { $okn++; continue }
    $ok = Retry { scp -q @SO -P $PORT "${A}:/root/cvresearch/output/$rel" "$target.part" }
    if ($ok -and ((Get-Item "$target.part").Length -eq $size)) { Move-Item "$target.part" $target -Force; $okn++ }
    else { Remove-Item "$target.part" -Force -ErrorAction SilentlyContinue; $failn++; Log "FAILED run ckpt $rel" }
  }
  Log "run checkpoints: $okn ok, $failn failed of $($items.Count)"
}
Log "backup pass done (WithCkpt=$WithCkpt AllRunCkpts=$AllRunCkpts)"
