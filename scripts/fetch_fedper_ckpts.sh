#!/bin/bash
# Read-only download of batch2_fedper per-client checkpoints (9 sensor clients), single connection, retries.
BASE=/d/cvresearch/ViPT/output/checkpoints/train/vipt
LOG=/d/cvresearch/ViPT/output/eval_results/fetch_fedper_ckpts.log
echo "START $(date)" > $LOG
for cid in rgbd_001 rgbd_002 rgbd_003 rgbe_000 rgbe_002 rgbe_003; do
  dst=$BASE/batch2_fedper_$cid; mkdir -p $dst
  [ -f $dst/ViPTrack_ep0120.pth.tar ] && [ -f $dst/.ok ] && continue
  ok=0
  for try in 1 2 3 4 5 6 7 8 9 10; do
    scp -q -o ConnectTimeout=30 -o BatchMode=yes -P 28125 root@px2-jcy.matpool.com:/root/cvresearch/output/batch2_fedper/checkpoints/ViPTrack_${cid}_ep0120.pth.tar $dst/ViPTrack_ep0120.pth.tar.part && mv $dst/ViPTrack_ep0120.pth.tar.part $dst/ViPTrack_ep0120.pth.tar && ok=1 && break
    echo "$cid try $try failed $(date +%T)" >> $LOG; sleep 25
  done
  [ $ok = 1 ] && echo "$cid ok $(stat -c %s $dst/ViPTrack_ep0120.pth.tar)" >> $LOG && touch $dst/.ok
done
echo "END $(date)" >> $LOG
