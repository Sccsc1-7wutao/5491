# daemon7 (2026-10-04): evaluate the S-A / seed-0 cloud runs locally as their final (round-120) checkpoints appear.
# thermal (LasHeR, local scoring), event (VisEvent SR), depth (DepthTrack F). One evaluation at a time via the shared GPU lock.
# Cloud: root@px1-jcy.matpool.com:27460. Round 120 only, never an earlier checkpoint. Log: output\eval_results\local_daemon7.log
param([switch]$TestThermal)
$ErrorActionPreference = 'Continue'
$ROOT = 'D:\cvresearch\ViPT'; $PY = 'D:\Conda\envs\sdstrack\python.exe'; $VOTPY = 'D:\Conda\envs\vot\python.exe'
$EV = "$ROOT\output\eval_results"; $LOG = "$EV\local_daemon7.log"; $CKR = "$ROOT\output\checkpoints\train\vipt"
$SSHA = @('-o','ConnectTimeout=30','-o','BatchMode=yes','-p','27460')
$A = 'root@px1-jcy.matpool.com'
function QEnd($p) { if (-not (Test-Path $p)) { return $false }; return [bool](Select-String -Path $p -Pattern '^END ' -Quiet) }
function WaitIdle { while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'test_rgbe_mgpus|test_rgbt_mgpus|run_offline_depthtrack|transfer_matrix' -and $_.Name -match 'python' }) { Start-Sleep 60 } }
# --- GPU LOCK BEGIN (shared by local_daemon2/3): exclusive lock file, auto-removed when the holder exits; WaitIdle still covers the lock-unaware transfer matrix
$LOCKF = "$EV\gpu_eval.lock"; $script:LockFs = $null
function LiveWants { Get-ChildItem "$EV\gpu_eval.want.*" -ErrorAction SilentlyContinue | Where-Object { $wp = [int]($_.Name -replace '.*\.', ''); if ($wp -eq $PID) { $false } elseif (Get-Process -Id $wp -ErrorAction SilentlyContinue) { $true } else { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue; $false } } }
function AcquireGpu($what) {
  $wantF = "$EV\gpu_eval.want.$PID"
  if ($script:NoWant -ne $true) { Set-Content $wantF "pid=$PID what=$what since=$(Get-Date -Format s)" -Encoding ascii }   # queue marker: the transfer matrix yields to anyone waiting
  while ($true) {
    if (Test-Path $LOCKF) {
      $txt = Get-Content $LOCKF -ErrorAction SilentlyContinue
      $lp = if ($txt -match 'pid=(\d+)') { [int]$Matches[1] } else { 0 }
      if ($lp -and -not (Get-Process -Id $lp -ErrorAction SilentlyContinue)) { Remove-Item $LOCKF -Force -ErrorAction SilentlyContinue; Log "removed stale gpu lock (pid $lp)" }
    }
    try {
      $fs = New-Object System.IO.FileStream($LOCKF, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite, 4096, [IO.FileOptions]::DeleteOnClose)
      $b = [Text.Encoding]::UTF8.GetBytes("pid=$PID what=$what since=$(Get-Date -Format s)"); $fs.Write($b, 0, $b.Length); $fs.Flush()
      $script:LockFs = $fs; Remove-Item $wantF -Force -ErrorAction SilentlyContinue; break
    } catch { Start-Sleep 10 }
  }
  WaitIdle
  Log "gpu lock acquired: $what"
}
function ReleaseGpu { if ($script:LockFs) { $script:LockFs.Dispose(); $script:LockFs = $null } }
function EventEval($tag) { AcquireGpu "event $tag"; try { EventEvalInner $tag } finally { ReleaseGpu } }
function DepthEval($tag) { AcquireGpu "depth $tag"; try { DepthEvalInner $tag } finally { ReleaseGpu } }
# --- GPU LOCK END
function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding utf8 }
function SshTest($path) { for ($i=0;$i -lt 3;$i++) { $o = & ssh @SSHA $A "test -f $path && echo yes" 2>$null; if ($o -match 'yes') { return $true } ; if ($LASTEXITCODE -eq 0 -or $o) { return $false } ; Start-Sleep 10 }; return $false }
function WaitRemote($path) { while (-not (SshTest $path)) { Start-Sleep 120 }; Start-Sleep 90 }
function Fetch($remote, $localDir) {
  New-Item -ItemType Directory -Force $localDir | Out-Null
  $dst = "$localDir\ViPTrack_ep0120.pth.tar"
  if (Test-Path $dst) { return $true }
  for ($i=0;$i -lt 10;$i++) {
    & scp -q -o ConnectTimeout=30 -o BatchMode=yes -P 27460 "${A}:$remote" "$dst.part" 2>$null
    if ($LASTEXITCODE -eq 0 -and (Test-Path "$dst.part")) { Move-Item "$dst.part" $dst -Force; return $true }
    Start-Sleep 25
  }
  Log "FETCH FAILED $remote"; return $false
}
function EventEvalInner($tag) {
  Copy-Item -Path "$ROOT\experiments\vipt\batch1_modality.yaml" -Destination "$ROOT\experiments\vipt\$tag.yaml" -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force "$ROOT\RGBE_workspace\results\VisEvent\$tag" | Out-Null
  Log "START event $tag"
  Start-Process -FilePath $PY -ArgumentList 'RGBE_workspace\test_rgbe_mgpus.py','--script_name','vipt','--dataset_name','VisEvent','--yaml_name',$tag,'--epoch','120','--mode','parallel','--threads','4' -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_$tag.log" -RedirectStandardError "$EV\daemon_$tag.err" -WindowStyle Hidden -Wait
  Log "END event $tag files=$((Get-ChildItem "$ROOT\RGBE_workspace\results\VisEvent\$tag" -File | Measure-Object).Count)"
}
function DepthEvalInner($tag) {
  $W ="$ROOT\Depthtrack_workspace_$tag"
  New-Item -ItemType Directory -Force "$W\results\$tag" | Out-Null
  Copy-Item "$ROOT\Depthtrack_workspace_batch3_bucket_fna\config.yaml" $W -Force
  (Get-Content "$ROOT\Depthtrack_workspace_batch3_bucket_fna\trackers.ini") -replace 'batch3_bucket_fna',$tag | Set-Content "$W\trackers.ini"
  if (-not (Test-Path "$W\sequences")) { New-Item -ItemType Junction -Path "$W\sequences" -Target "$ROOT\Depthtrack_workspace_batch1_fedavg\sequences" | Out-Null }
  Log "START depth $tag"
  Start-Process -FilePath $PY -ArgumentList 'Depthtrack_workspace\run_offline_depthtrack.py','rgbd-unsupervised','--checkpoint',"$CKR\$tag\ViPTrack_ep0120.pth.tar",'--output-dir',"$W\results\$tag" -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_$tag.log" -RedirectStandardError "$EV\daemon_$tag.err" -WindowStyle Hidden -Wait
  Start-Process -FilePath $VOTPY -ArgumentList '-m','vot','analysis','--workspace',$W,'--format','json','--name',"final_$tag",$tag -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_${tag}_vot.log" -RedirectStandardError "$EV\daemon_${tag}_vot.err" -WindowStyle Hidden -Wait
  # VOT writes cache paths ~195 chars below the workspace; with MAX_PATH=260 a long tag makes the analysis fail silently. Retry through a short junction, and never log END on failure.
  $hasRes = { [bool](Get-ChildItem "$W\analysis" -Recurse -Filter results.json -ErrorAction SilentlyContinue | Select-Object -First 1) }
  $vok = & $hasRes
  if (-not $vok) {
    Log "VOT analysis gave no results.json for $tag - retrying via short junction"
    $J = "$ROOT\DW_$([guid]::NewGuid().ToString('N').Substring(0,6))"; New-Item -ItemType Junction -Path $J -Target $W | Out-Null
    Start-Process -FilePath $VOTPY -ArgumentList '-m','vot','analysis','--workspace',$J,'--format','json','--name',"final_$tag",$tag -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_${tag}_vot2.log" -RedirectStandardError "$EV\daemon_${tag}_vot2.err" -WindowStyle Hidden -Wait
    cmd /c rmdir "$J"; $vok = & $hasRes
  }
  if ($vok) {
    Start-Process -FilePath $VOTPY -ArgumentList 'D:\cvresearch\scripts\batch2_depth_ci.py',$tag -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_${tag}_ci.log" -RedirectStandardError "$EV\daemon_${tag}_ci.err" -WindowStyle Hidden -Wait
    Log "END depth $tag seqs=$((Get-ChildItem "$W\results\$tag\rgbd-unsupervised" -Directory -ErrorAction SilentlyContinue | Measure-Object).Count)"
  } else { Log "FAILED depth $tag - VOT analysis produced no results.json, CI skipped" }
}
# ---- daemon7 additions ----
$CLOUD = '/root/cvresearch/output'
function ThermalEvalInner($tag) {
  Copy-Item -Path "$ROOT\experiments\vipt\batch1_modality.yaml" -Destination "$ROOT\experiments\vipt\$tag.yaml" -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force "$ROOT\RGBT_workspace\results\LasHeR\$tag" | Out-Null
  Log "START thermal $tag"
  Start-Process -FilePath $PY -ArgumentList 'RGBT_workspace\test_rgbt_mgpus.py','--script_name','vipt','--dataset_name','LasHeR','--yaml_name',$tag,'--epoch','120','--mode','parallel','--threads','4' -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_$tag.log" -RedirectStandardError "$EV\daemon_$tag.err" -WindowStyle Hidden -Wait
  $n = (Get-ChildItem "$ROOT\RGBT_workspace\results\LasHeR\$tag" -File | Measure-Object).Count
  if ($n -eq 245) {
    $o = & $PY 'D:\cvresearch\scripts\lasher_eval.py' --results "$ROOT\RGBT_workspace\results\LasHeR\$tag" --save-npz "$EV\${tag}_curves.npz" 2>&1
    ($o | Select-Object -Last 4) -join ' | ' | ForEach-Object { Log "SCORE thermal $tag : $_" }
    Log "END thermal $tag files=$n"
  } else { Log "FAILED thermal $tag files=$n (expected 245), not scored" }
}
function ThermalEval($tag) { AcquireGpu "thermal $tag"; try { ThermalEvalInner $tag } finally { ReleaseGpu } }
# ckpt on the cloud appears at the very end of a run (--export-at-end); wait for it, then +90 s so the write is complete
function WaitCkpt($path) { while (-not (SshTest $path)) { Start-Sleep 120 }; Start-Sleep 90 }
function FetchCkpt($remote, $tag) { Fetch $remote "$CKR\$tag" }
function RunMulti($run, $prefix) {   # one 12-client run, per-modality checkpoints
  Log "WAIT $run"; WaitCkpt "$CLOUD/$run/checkpoints/ViPTrack_depth_ep0120.pth.tar"
  foreach ($m in 'thermal','event','depth') { FetchCkpt "$CLOUD/$run/checkpoints/ViPTrack_${m}_ep0120.pth.tar" "${prefix}_$m" | Out-Null }
  ThermalEval "${prefix}_thermal"; EventEval "${prefix}_event"; DepthEval "${prefix}_depth"
}
function RunIndep($run, $prefix, $m) {  # one single-modality run
  Log "WAIT $run"; WaitCkpt "$CLOUD/$run/checkpoints/ViPTrack_ep0120.pth.tar"
  FetchCkpt "$CLOUD/$run/checkpoints/ViPTrack_ep0120.pth.tar" "${prefix}_$m" | Out-Null
  switch ($m) { 'thermal' { ThermalEval "${prefix}_$m" } 'event' { EventEval "${prefix}_$m" } 'depth' { DepthEval "${prefix}_$m" } }
}
Log "daemon7 start"
if ($TestThermal) {   # self-test (rule: before each eval batch, FedAvg seed 42 must reproduce AUC 46.91): re-evaluate an existing local checkpoint and compare with its earlier result
  New-Item -ItemType Directory -Force "$CKR\d6test_thermal" | Out-Null
  Copy-Item "$CKR\batch1_fedavg\ViPTrack_ep0120.pth.tar" "$CKR\d6test_thermal\ViPTrack_ep0120.pth.tar" -Force
  ThermalEval 'd6test_thermal'
  $o = & $PY 'D:\cvresearch\scripts\lasher_eval.py' --results "$ROOT\RGBT_workspace\results\LasHeR\batch1_fedavg" 2>&1
  Log "REFERENCE batch1_fedavg existing result (expected AUC 46.91) : $(($o | Select-Object -Last 4) -join ' | ')"
  Log "daemon7 TEST END"; exit
}
# daemon7 (2026-10-06): re-evaluate FedPer / Ditto thermal (previously only evaluated on the released instance B) in the LOCAL caliber.
# Uses the same GPU lock as daemon6, so it interleaves with the round-2 evals instead of colliding with them.
function RunPers($mem, $tagPre) {
  foreach ($i in '001','002','003') {
    $tag = "${tagPre}_rgbt_$i"
    FetchCkpt "$mem/ViPTrack_rgbt_${i}_ep0120.pth.tar" $tag | Out-Null
    if (-not (Test-Path "$CKR\$tag\ViPTrack_ep0120.pth.tar")) { Log "MISSING ckpt $tag"; continue }
    ThermalEval $tag
  }
}
RunPers '/mnt/cvr_eval/ckpt_fedper' 'lc_fedper'
RunPers '/mnt/cvr_eval/ckpt_batch4_ditto' 'lc_ditto'
Log "daemon7 END"
