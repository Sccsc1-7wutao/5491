# Transfer-matrix experiment: run after local_daemon (daemon1) has finished, until the batch8_main_norgb evaluation needs the GPU.
$ROOT='D:\cvresearch\ViPT'; $TM='D:\cvresearch\ViPT_transfer'; $PY='D:\Conda\envs\sdstrack\python.exe'; $EV="$ROOT\output\eval_results"; $LOG="$EV\tm_now.log"
function Log($m){ "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding utf8 }
function B8Ready { $o = & ssh -o ConnectTimeout=30 -o BatchMode=yes -p 28125 root@px2-jcy.matpool.com 'test -f /root/cvresearch/output/batch8_main_norgb/checkpoints/ViPTrack_thermal_ep0120.pth.tar && echo yes' 2>$null; return ($o -match 'yes') }
Log "start now (not waiting for daemon1)"
$env:CVR_ROOT='D:\cvresearch'; $env:CVR_DATA='D:\cvresearch\data'
$bases=@(@('modality_r020',"$ROOT\output\fed\batch1\modality\fed_state_r0020.pt",20),@('modality_r060',"$ROOT\output\fed\batch1\modality\fed_state_r0060.pt",60),@('fedavg_r120',"$ROOT\output\fed\batch1\fed_state_fedavg_r0120.pt",120),@('modality_r120',"$ROOT\output\fed\batch1\fed_state_modality_r0120.pt",120))
foreach($b in $bases){
  if (B8Ready) { Log "batch8 ready - stop before $($b[0])"; break }
  if (Test-Path "$EV	ransfer_matrix\$($b[0])	ransfer_matrix_$($b[0]).csv") { Log "skip $($b[0]) (done)"; continue }
  Log "START $($b[0])"
  $pr = Start-Process -FilePath $PY -ArgumentList 'lib\federated\transfer_matrix.py','--base-label',$b[0],'--base-state',$b[1],'--round-idx',"$($b[2])",'--n-eval','40','--repeats','3','--out-dir',"$EV\transfer_matrix\$($b[0])" -WorkingDirectory $TM -RedirectStandardOutput "$EV\tmnow_$($b[0]).log" -RedirectStandardError "$EV\tmnow_$($b[0]).err" -WindowStyle Hidden -PassThru
  $null = $pr.Handle
  if (-not $pr.WaitForExit(5400000)) { Log "TIMEOUT $($b[0])"; try{$pr.Kill()}catch{} } else { Log "END $($b[0]) rc=$($pr.ExitCode)" }
}
Log "waiter2 done"
