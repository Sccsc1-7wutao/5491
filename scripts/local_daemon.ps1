# Local evaluation daemon: runs the transfer-matrix experiment, then evaluates event/depth checkpoints of new cloud runs as soon as they finish.
# One evaluation at a time (local GPU / RAM). Everything is logged to output\eval_results\local_daemon.log.
$ErrorActionPreference = 'Continue'
$ROOT = 'D:\cvresearch\ViPT'; $PY = 'D:\Conda\envs\sdstrack\python.exe'; $VOTPY = 'D:\Conda\envs\vot\python.exe'
$EV = "$ROOT\output\eval_results"; $LOG = "$EV\local_daemon.log"; $CKR = "$ROOT\output\checkpoints\train\vipt"
$SSHA = @('-o','ConnectTimeout=30','-o','BatchMode=yes','-p','28125')
$A = 'root@px2-jcy.matpool.com'
function QEnd($p) { if (-not (Test-Path $p)) { return $false }; return [bool](Select-String -Path $p -Pattern '^END ' -Quiet) }
function Log($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding utf8 }
function SshTest($path) { for ($i=0;$i -lt 3;$i++) { $o = & ssh @SSHA $A "test -f $path && echo yes" 2>$null; if ($o -match 'yes') { return $true } ; if ($LASTEXITCODE -eq 0 -or $o) { return $false } ; Start-Sleep 10 }; return $false }
function WaitRemote($path) { while (-not (SshTest $path)) { Start-Sleep 120 }; Start-Sleep 90 }
function Fetch($remote, $localDir) {
  New-Item -ItemType Directory -Force $localDir | Out-Null
  $dst = "$localDir\ViPTrack_ep0120.pth.tar"
  if (Test-Path $dst) { return $true }
  for ($i=0;$i -lt 10;$i++) {
    & scp -q -o ConnectTimeout=30 -o BatchMode=yes -P 28125 "${A}:$remote" "$dst.part" 2>$null
    if ($LASTEXITCODE -eq 0 -and (Test-Path "$dst.part")) { Move-Item "$dst.part" $dst -Force; return $true }
    Start-Sleep 25
  }
  Log "FETCH FAILED $remote"; return $false
}
function EventEval($tag) {
  Copy-Item -Path "$ROOT\experiments\vipt\batch1_modality.yaml" -Destination "$ROOT\experiments\vipt\$tag.yaml" -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force "$ROOT\RGBE_workspace\results\VisEvent\$tag" | Out-Null
  Log "START event $tag"
  Start-Process -FilePath $PY -ArgumentList 'RGBE_workspace\test_rgbe_mgpus.py','--script_name','vipt','--dataset_name','VisEvent','--yaml_name',$tag,'--epoch','120','--mode','parallel','--threads','4' -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_$tag.log" -RedirectStandardError "$EV\daemon_$tag.err" -WindowStyle Hidden -Wait
  Log "END event $tag files=$((Get-ChildItem "$ROOT\RGBE_workspace\results\VisEvent\$tag" -File | Measure-Object).Count)"
}
function DepthEval($tag) {
  $W = "$ROOT\Depthtrack_workspace_$tag"
  New-Item -ItemType Directory -Force "$W\results\$tag" | Out-Null
  Copy-Item "$ROOT\Depthtrack_workspace_batch3_bucket_fna\config.yaml" $W -Force
  (Get-Content "$ROOT\Depthtrack_workspace_batch3_bucket_fna\trackers.ini") -replace 'batch3_bucket_fna',$tag | Set-Content "$W\trackers.ini"
  if (-not (Test-Path "$W\sequences")) { New-Item -ItemType Junction -Path "$W\sequences" -Target "$ROOT\Depthtrack_workspace_batch1_fedavg\sequences" | Out-Null }
  Log "START depth $tag"
  Start-Process -FilePath $PY -ArgumentList 'Depthtrack_workspace\run_offline_depthtrack.py','rgbd-unsupervised','--checkpoint',"$CKR\$tag\ViPTrack_ep0120.pth.tar",'--output-dir',"$W\results\$tag" -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_$tag.log" -RedirectStandardError "$EV\daemon_$tag.err" -WindowStyle Hidden -Wait
  Start-Process -FilePath $VOTPY -ArgumentList '-m','vot','analysis','--workspace',$W,'--format','json','--name',"final_$tag",$tag -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_${tag}_vot.log" -RedirectStandardError "$EV\daemon_${tag}_vot.err" -WindowStyle Hidden -Wait
  Start-Process -FilePath $VOTPY -ArgumentList 'D:\cvresearch\scripts\batch2_depth_ci.py',$tag -WorkingDirectory $ROOT -RedirectStandardOutput "$EV\daemon_${tag}_ci.log" -RedirectStandardError "$EV\daemon_${tag}_ci.err" -WindowStyle Hidden -Wait
  Log "END depth $tag seqs=$((Get-ChildItem "$W\results\$tag\rgbd-unsupervised" -Directory -ErrorAction SilentlyContinue | Measure-Object).Count)"
}
function RunModalityRun($name, $tagPrefix) {
  Log "WAIT $name"; WaitRemote "/mnt/cvr_eval/ckpt_$name/DONE"
  foreach ($m in 'event','depth') { Fetch "/root/cvresearch/output/$name/checkpoints/ViPTrack_${m}_ep0120.pth.tar" "$CKR\${tagPrefix}_$m" | Out-Null }
  EventEval "${tagPrefix}_event"; DepthEval "${tagPrefix}_depth"
}

Log "daemon start"
# wait for the earlier local queues
while (-not (QEnd "$EV\queue_5\queue.log")) { Start-Sleep 60 }
Log "queue_5 finished"

# 1) transfer-matrix experiment (smoke, then full for the main method at r20/r60/r120 and FedAvg r120)
$TM = 'D:\cvresearch\ViPT_transfer'; $env:CVR_ROOT = 'D:\cvresearch'; $env:CVR_DATA = 'D:\cvresearch\data'
function RunTM($argl, $tag, $timeoutMs) {
  $pr = Start-Process -FilePath $PY -ArgumentList $argl -WorkingDirectory $TM -RedirectStandardOutput "$EV\daemon_tm_$tag.log" -RedirectStandardError "$EV\daemon_tm_$tag.err" -WindowStyle Hidden -PassThru
  $null = $pr.Handle
  if (-not $pr.WaitForExit($timeoutMs)) { Log "TIMEOUT transfer $tag - killing"; try { $pr.Kill() } catch {}; return -999 }
  $pr.WaitForExit(); return [int]$pr.ExitCode
}
$smokeRc = RunTM @('lib\federated\transfer_matrix.py','--base-label','smoke_modality_r120','--base-state',"$ROOT\output\fed\batch1\fed_state_modality_r0120.pt",'--round-idx','120','--n-eval','5','--repeats','1','--out-dir',"$EV\transfer_matrix\smoke") 'smoke' 2400000
Log "transfer smoke rc=$smokeRc"
if ($smokeRc -eq 0) {
  $bases = @(@('modality_r120',"$ROOT\output\fed\batch1\fed_state_modality_r0120.pt",120), @('modality_r060',"$ROOT\output\fed\batch1\modality\fed_state_r0060.pt",60), @('modality_r020',"$ROOT\output\fed\batch1\modality\fed_state_r0020.pt",20), @('fedavg_r120',"$ROOT\output\fed\batch1\fed_state_fedavg_r0120.pt",120))
  foreach ($b in $bases) {
    Log "START transfer $($b[0])"
    $rc = RunTM @('lib\federated\transfer_matrix.py','--base-label',$b[0],'--base-state',$b[1],'--round-idx',"$($b[2])",'--n-eval','40','--repeats','3','--out-dir',"$EV\transfer_matrix\$($b[0])") $b[0] 7200000
    Log "END transfer $($b[0]) rc=$rc"
  }
}

# 2) new-method WITHOUT pure-RGB clients (GPU0, ~08:40 10-03): event + depth here, thermal on B
RunModalityRun 'batch5_fna_norgb' 'batch5_fna_norgb'

# 3) Ditto personalized models (GPU1, ~09:00): event 3 + depth 3 here, thermal 3 on B
Log "WAIT batch4_ditto"; WaitRemote '/mnt/cvr_eval/ckpt_batch4_ditto/DONE'
foreach ($c in 'rgbe_000','rgbe_002','rgbe_003','rgbd_001','rgbd_002','rgbd_003') { Fetch "/root/cvresearch/output/batch4_ditto/checkpoints/ViPTrack_${c}_ep0120.pth.tar" "$CKR\batch4_ditto_$c" | Out-Null }
foreach ($c in 'rgbe_000','rgbe_002','rgbe_003') { EventEval "batch4_ditto_$c" }
foreach ($c in 'rgbd_001','rgbd_002','rgbd_003') { DepthEval "batch4_ditto_$c" }

# 4) independent-event WITHOUT pure-RGB client (GPU0 after item 2): normal + zero-aux, and zero-aux for the independent-event WITH RGB
Log "WAIT batch7_indep_event_norgb"; WaitRemote '/root/cvresearch/output/batch7_indep_event_norgb/checkpoints/ViPTrack_ep0120.pth.tar'
Fetch '/root/cvresearch/output/batch7_indep_event_norgb/checkpoints/ViPTrack_ep0120.pth.tar' "$CKR\batch7_indep_event_norgb" | Out-Null
EventEval 'batch7_indep_event_norgb'
foreach ($pair in @(@('batch7_indep_event_norgb','noaux_batch7_indep_event_norgb'), @('batch2_indep_event_event','noaux_batch2_indep_event'))) {
  New-Item -ItemType Directory -Force "$CKR\$($pair[1])" | Out-Null
  Copy-Item "$CKR\$($pair[0])\ViPTrack_ep0120.pth.tar" "$CKR\$($pair[1])\ViPTrack_ep0120.pth.tar" -Force
  $env:CVR_ZERO_AUX = '1'; EventEval $pair[1]; Remove-Item Env:\CVR_ZERO_AUX
}

# 5) E2 bucket arm (GPU2, ~20:40 10-03): event + depth here, thermal on B
RunModalityRun 'batch6_e2_bucket_ds' 'batch6_e2_bucket_ds'
Log "daemon END"
