# Transfer-matrix waiter v3: every base runs under the shared GPU lock (gpu_eval.lock); an already-running transfer_matrix process is adopted instead of restarted.
# Queueing: before taking the lock it yields to any evaluation that has announced itself (gpu_eval.want.<pid>), so it can never jump the queue.
# Stops once batch8_main_norgb is trained (its evaluation needs the GPU).
$ROOT='D:\cvresearch\ViPT'; $TM='D:\cvresearch\ViPT_transfer'; $PY='D:\Conda\envs\sdstrack\python.exe'; $EV="$ROOT\output\eval_results"; $LOG="$EV\tm_waiter2.log"
$script:NoWant = $true   # this waiter never announces itself; it only yields
function Log($m){ "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding utf8 }
function WaitIdle {}   # no-op here: every other evaluation goes through the lock; transfer_matrix processes are our own# --- GPU LOCK BEGIN (shared by local_daemon2/3): exclusive lock file, auto-removed when the holder exits; WaitIdle still covers the lock-unaware transfer matrix
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
function B8Ready { $o = & ssh -o ConnectTimeout=30 -o BatchMode=yes -p 28125 root@px2-jcy.matpool.com 'test -f /root/cvresearch/output/batch8_main_norgb/checkpoints/ViPTrack_thermal_ep0120.pth.tar && echo yes' 2>$null; return ($o -match 'yes') }
function TmDone($b) { Test-Path $b[4] }
$env:CVR_ROOT='D:\cvresearch'; $env:CVR_DATA='D:\cvresearch\data'
# entry = label, base state, round, script, done-marker csv. modality_r120 was dropped (diagonal self-check fails at r020/r060 and gets worse with convergence);
# the step-size test on the r060 base replaces it.
$MX="$EV\transfer_matrix"
$bases=@(
  @('modality_r020',"$ROOT\output\fed\batch1\modality\fed_state_r0020.pt",20,'transfer_matrix.py',"$MX\modality_r020\transfer_matrix_modality_r020.csv"),
  @('modality_r060',"$ROOT\output\fed\batch1\modality\fed_state_r0060.pt",60,'transfer_matrix.py',"$MX\modality_r060\transfer_matrix_modality_r060.csv"),
  @('fedavg_r120',"$ROOT\output\fed\batch1\fed_state_fedavg_r0120.pt",120,'transfer_matrix.py',"$MX\fedavg_r120\transfer_matrix_fedavg_r120.csv"),
  @('scale_r060',"$ROOT\output\fed\batch1\modality\fed_state_r0060.pt",60,'transfer_scale_test.py',"$MX\scale_r060\scale_test_scale_r060.csv"))
Log "waiter3 start (lock-aware, queue-aware)"
foreach($b in $bases){
  if (TmDone $b) { Log "SKIP $($b[0]) (csv exists)"; continue }
  if (B8Ready) { Log "batch8 ready - stop before $($b[0])"; break }
  $logged = $false
  while ($true) {
    while (@(LiveWants).Count -gt 0) { if (-not $logged) { Log "yielding GPU to queued eval before $($b[0]): $((@(LiveWants) | % { $_.Name }) -join ',')"; $logged = $true }; Start-Sleep 20 }
    AcquireGpu "transfer $($b[0])"
    if (@(LiveWants).Count -eq 0) { break }
    ReleaseGpu; Log "eval queued up while taking the lock - released it again before $($b[0])"
  }
  try {
    if (B8Ready) { Log "batch8 ready after lock wait - stop before $($b[0])"; break }
    $run = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match [regex]::Escape($b[3]) -and $_.CommandLine -match "--base-label $($b[0]) " } | Select-Object -First 1
    if ($run) {
      $pr = Get-Process -Id $run.ProcessId; $null = $pr.Handle
      $remain = [int](5400000 - ((Get-Date) - $pr.StartTime).TotalMilliseconds); if ($remain -lt 1000) { $remain = 1000 }
      Log "ADOPT running $($b[0]) pid $($pr.Id) started $($pr.StartTime.ToString('MM-dd HH:mm:ss')), $([int]($remain/60000)) min left"
    } else {
      Log "START $($b[0])"
      $pr = Start-Process -FilePath $PY -ArgumentList "lib\federated\$($b[3])",'--base-label',$b[0],'--base-state',$b[1],'--round-idx',"$($b[2])",'--n-eval','40','--repeats','3','--out-dir',"$EV\transfer_matrix\$($b[0])" -WorkingDirectory $TM -RedirectStandardOutput "$EV\tm2_$($b[0]).log" -RedirectStandardError "$EV\tm2_$($b[0]).err" -WindowStyle Hidden -PassThru
      $null = $pr.Handle; $remain = 5400000
    }
    if (-not $pr.WaitForExit($remain)) { Log "TIMEOUT $($b[0])"; try{$pr.Kill()}catch{} } else { Log "END $($b[0]) rc=$($pr.ExitCode)" }
  } finally { ReleaseGpu }
}
Log "waiter3 done"
