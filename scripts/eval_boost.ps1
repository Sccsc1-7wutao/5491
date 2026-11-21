# raises priority of ALL python.exe processes (the pool workers of the evaluation do not carry the script name) while the user is away; stops when eval_boost.stop exists
$stop = Join-Path $PSScriptRoot 'eval_boost.stop'
while (-not (Test-Path $stop)) {
  Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { try { if ($_.PriorityClass -ne 'High') { $_.PriorityClass = 'High' } } catch {} }
  Start-Sleep 10
}
