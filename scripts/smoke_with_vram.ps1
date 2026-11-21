# Run the ViPT smoke test while sampling GPU memory, then report the peak.
# Why a wrapper: nvidia-smi must be polled *during* the run; after the process
# exits the peak is gone. 200 ms sampling is fine for a ~20 s run.
param(
    [string]$Config  = "smoke_rgbt",
    [string]$Repo    = "D:\cvresearch\ViPT",
    [string]$Python  = "D:\Conda\envs\sdstrack\python.exe",
    [string]$VramLog = "D:\cvresearch\ViPT\output\vram_smoke.csv"
)

Set-Location $Repo
New-Item -ItemType Directory -Force (Split-Path $VramLog) | Out-Null
Remove-Item $VramLog -ErrorAction SilentlyContinue

# Baseline before the run, so we can subtract other processes' usage.
$baseline = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1)
"baseline_MiB,$baseline" | Out-File $VramLog -Encoding ascii

$poller = Start-Job -ArgumentList $VramLog -ScriptBlock {
    param($log)
    while ($true) {
        $m = (nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) | Select-Object -First 1
        "$m" | Out-File -Append $log -Encoding ascii
        Start-Sleep -Milliseconds 200
    }
}

$sw = [Diagnostics.Stopwatch]::StartNew()
& $Python lib/train/run_training.py --script vipt --config $Config --save_dir ./output --use_wandb 0 2>&1 |
    Where-Object { $_ -notmatch 'UserWarning|FutureWarning|storage = batch|torch\.stack|checkpoint = torch\.load|^\s*$' }
$sw.Stop()

Stop-Job $poller -ErrorAction SilentlyContinue
Remove-Job $poller -Force -ErrorAction SilentlyContinue

$rows = Get-Content $VramLog | Select-Object -Skip 1 | Where-Object { $_ -match ',' }
$used = $rows | ForEach-Object { [int]($_ -split ',')[0].Trim() }
$util = $rows | ForEach-Object { [int]($_ -split ',')[1].Trim() }

""
"================ VRAM / timing ================"
"wall clock        : {0:N1} s" -f $sw.Elapsed.TotalSeconds
"baseline (others) : {0} MiB" -f $baseline
"peak total        : {0} MiB" -f ($used | Measure-Object -Maximum).Maximum
"peak by training  : {0} MiB" -f (($used | Measure-Object -Maximum).Maximum - $baseline)
"GPU util mean/max : {0:N0}% / {1}%" -f (($util | Measure-Object -Average).Average), (($util | Measure-Object -Maximum).Maximum)
"samples           : {0}" -f $used.Count
