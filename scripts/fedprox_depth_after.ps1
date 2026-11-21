# Wait for the FedProx mu=0.1 DepthTrack inference (pid passed in) to exit, then run VOT analysis and the bootstrap CI.
param([int]$WaitPid)
$ErrorActionPreference = 'Continue'
$ev = 'D:\cvresearch\ViPT\output\eval_results'
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
$n = (Get-ChildItem 'D:\cvresearch\ViPT\Depthtrack_workspace_batch2_fedprox_mu01\results\batch2_fedprox_mu01\rgbd-unsupervised' -Directory -ErrorAction SilentlyContinue | Measure-Object).Count
"inference ended, sequences=$n at $(Get-Date)" | Out-File "$ev\batch2_fedprox_mu01_after.log"
if ($n -ne 50) { "NOT 50 sequences, stop" | Out-File "$ev\batch2_fedprox_mu01_after.log" -Append; exit 1 }
Set-Location 'D:\cvresearch\ViPT'
& 'D:\Conda\envs\vot\python.exe' -m vot analysis --workspace 'D:\cvresearch\ViPT\Depthtrack_workspace_batch2_fedprox_mu01' --format json --name final_batch2_fedprox_mu01 batch2_fedprox_mu01 *> "$ev\batch2_fedprox_mu01_vot_rerun.log"
"vot analysis rc=$LASTEXITCODE at $(Get-Date)" | Out-File "$ev\batch2_fedprox_mu01_after.log" -Append
& 'D:\Conda\envs\vot\python.exe' 'D:\cvresearch\scripts\batch2_depth_ci.py' batch2_fedprox_mu01 *> "$ev\batch2_fedprox_mu01_ci_rerun.log"
"ci rc=$LASTEXITCODE at $(Get-Date)" | Out-File "$ev\batch2_fedprox_mu01_after.log" -Append
