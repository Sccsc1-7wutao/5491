# Pull everything still needed from the cloud instance to the local machine (2026-10-07). Low priority, resumable-by-rerun, size-verified.
$ErrorActionPreference = 'Continue'
$ROOT='D:\cvresearch'; $CKR="$ROOT\ViPT\output\checkpoints\train\vipt"; $DST="$ROOT\cloud_backup\round23"; $LOG="$DST\transfer.log"
New-Item -ItemType Directory -Force $DST | Out-Null
function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding utf8 }
$SSH=@('-o','ConnectTimeout=30','-o','BatchMode=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4','-p','27460'); $H='root@px1-jcy.matpool.com'
function RSize($p) { for($i=0;$i -lt 5;$i++){ $o = & ssh @SSH $H "stat -c %s $p" 2>$null; if ($LASTEXITCODE -eq 0 -and $o -match '^\d+$') { return [int64]$o }; Start-Sleep 10 }; return -1 }
Log "transfer start"
# 1) fna checkpoints -> the exact local dirs/names daemon6 expects
foreach ($run in @(@('s0_fna','s0fn'), @('s123_fna','s123fn'))) {
  foreach ($m in 'thermal','event','depth') {
    $remote = "/root/cvresearch/output/$($run[0])/checkpoints/ViPTrack_${m}_ep0120.pth.tar"
    $dir = "$CKR\$($run[1])_$m"; $dst = "$dir\ViPTrack_ep0120.pth.tar"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $want = RSize $remote
    if ((Test-Path $dst) -and ((Get-Item $dst).Length -eq $want)) { Log "SKIP (already complete) $($run[0]) $m"; continue }
    $ok = $false
    for ($i=0; $i -lt 8 -and -not $ok; $i++) {
      & scp -q -o ConnectTimeout=30 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -P 27460 "${H}:$remote" "$dst.part" 2>$null
      if ($LASTEXITCODE -eq 0 -and (Test-Path "$dst.part") -and ((Get-Item "$dst.part").Length -eq $want)) { Move-Item "$dst.part" $dst -Force; $ok = $true } else { Start-Sleep 20 }
    }
    Log "$(if($ok){'OK'}else{'FAILED'}) fna ckpt $($run[0]) $m size=$want"
  }
}
# 2) metadata of every round-1/2/3 run (no checkpoints: those are already local from the evaluations, size-checked below)
$runs = 's0_fedavg s0_modality s0_indep_thermal s0_indep_depth s0_indep_event sa_seed42 sa_seed0 sa_seed123 s123_fedavg s123_modality s123_indep_thermal s123_indep_depth s123_indep_event s0_fna s123_fna'
$cmd = "ssh -o ConnectTimeout=30 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p 27460 $H `"cd /root/cvresearch/output && tar cf - --exclude=checkpoints $runs logs`" | tar xf - -C `"$DST`""
cmd /c $cmd
Log "metadata tar exit=$LASTEXITCODE"
# 3) size check of local final checkpoints vs cloud (evaluated runs)
$pairs = @(@('s0_fedavg','s0fa'),@('s0_modality','s0bk'),@('sa_seed42','sa42'),@('sa_seed0','s0sa'),@('sa_seed123','s123sa'),@('s123_fedavg','s123fa'),@('s123_modality','s123bk'))
foreach ($p in $pairs) { foreach ($m in 'thermal','event','depth') {
  $remote = "/root/cvresearch/output/$($p[0])/checkpoints/ViPTrack_${m}_ep0120.pth.tar"; $loc = "$CKR\$($p[1])_$m\ViPTrack_ep0120.pth.tar"
  $want = RSize $remote; $have = if (Test-Path $loc) { (Get-Item $loc).Length } else { -1 }
  Log "$(if($want -eq $have){'MATCH'}else{'MISMATCH'}) $($p[0]) $m cloud=$want local=$have" } }
Log "transfer END"
