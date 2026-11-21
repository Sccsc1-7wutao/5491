#!/bin/bash
n=batch9_indep_thermal_norgb; f=/root/cvresearch/output/$n/checkpoints/ViPTrack_ep0120.pth.tar
while [ ! -f $f ]; do sleep 60; done
sleep 90; mkdir -p /mnt/cvr_eval/ckpt_$n; ionice -c3 cp $f /mnt/cvr_eval/ckpt_$n/ && touch /mnt/cvr_eval/ckpt_$n/DONE
echo "$(date) $n done" >> /root/cvresearch/export_watch.log
