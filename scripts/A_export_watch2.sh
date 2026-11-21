#!/bin/bash
OUTR=/root/cvresearch/output
watch() { name=$1
  while [ ! -f $OUTR/$name/checkpoints/ViPTrack_thermal_ep0120.pth.tar ]; do sleep 60; done
  sleep 90; mkdir -p /mnt/cvr_eval/ckpt_$name
  ionice -c3 cp $OUTR/$name/checkpoints/ViPTrack_thermal_ep0120.pth.tar /mnt/cvr_eval/ckpt_$name/ && touch /mnt/cvr_eval/ckpt_$name/DONE
  echo "$(date) $name done" >> /root/cvresearch/export_watch.log; }
watch batch8_main_norgb &
watch batch8_fedavg_norgb &
wait
