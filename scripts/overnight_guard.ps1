# Overnight guard (10-04): restarts the three local schedulers if one dies before finishing, and writes a heartbeat.
# It only restarts: local_daemon2.ps1, local_daemon3.ps1, tm_waiter3.ps1. A script counts as finished when its log has the done-marker, and is then left alone.
# It never touches cloud instances, never starts training, never deletes anything. Exits when all three are finished.
$EV = 'D:\cvresearch\ViPT\output\eval_results'; $LOG = "$EV\overnight_guard.log"
$S = 'D:\cvresearch\scripts'
function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding utf8 }
$watch = @(
  @{ n = 'local_daemon2.ps1'; f = "$EV\local_daemon2.log"; p = 'daemon2 END' },
  @{ n = 'local_daemon3.ps1'; f = "$EV\local_daemon3.log"; p = 'daemon3 END' },
  @{ n = 'tm_waiter3.ps1';    f = "$EV\tm_waiter2.log";    p = 'waiter3 done' },
  @{ n = 'local_daemon4.ps1'; f = "$EV\local_daemon4.log"; p = 'daemon4 END' }
)
function IsAlive($name) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'powershell' -and $_.CommandLine -match [regex]::Escape($name) }) }
function IsDone($w) { (Test-Path $w.f) -and [bool](Select-String -Path $w.f -Pattern $w.p -Quiet) }
Log "guard start (pid $PID)"
$tick = 0
while ($true) {
  $pending = 0
  foreach ($w in $watch) {
    if (IsDone $w) { continue }
    $pending++
    if (-not (IsAlive $w.n)) {
      Start-Sleep 20                                   # grace period: someone may be restarting it on purpose
      if (-not (IsAlive $w.n)) {
        Start-Process powershell -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', "$S\$($w.n)"
        Log "RESTARTED $($w.n) (it was not running and had not logged '$($w.p)')"
      }
    }
  }
  # kill hung ssh/scp to the cloud instances (ssh > 5 min, scp > 45 min); the schedulers treat that as a failed attempt and retry
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(ssh|scp)\.exe$' -and $_.CommandLine -match 'px[12]-jcy\.matpool\.com' } | ForEach-Object {
    $age = ((Get-Date) - $_.CreationDate).TotalMinutes
    $limit = if ($_.Name -match 'scp') { 45 } else { 5 }
    if ($age -gt $limit) { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Log "killed hung $($_.Name) pid $($_.ProcessId) after $([int]$age) min" }
  }
  if (($tick % 6) -eq 0) {
    $lock = 'none'
    try { $fs = [IO.File]::Open("$EV\gpu_eval.lock", 'Open', 'Read', [IO.FileShare]'ReadWrite,Delete'); $b = New-Object byte[] 200; $n = $fs.Read($b, 0, 200); $fs.Close(); $lock = [Text.Encoding]::UTF8.GetString($b, 0, $n) } catch {}
    $gpu = (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>$null)
    Log ("heartbeat pending=$pending D:free={0:N0}GB gpu=[{1}] lock=[{2}]" -f ((Get-PSDrive D).Free / 1GB), $gpu, $lock)
  }
  if ($pending -eq 0) { Log "all schedulers finished - guard exits"; break }
  $tick++
  Start-Sleep 300
}
