#!/bin/bash
# For each finished run on A: wait until its exported checkpoints exist, then copy the thermal ones to the shared disk and mark DONE.
OUTR=/root/cvresearch/output
watch() {
  name=$1; shift            # run dir name, then expected checkpoint filenames to copy for B
  expected_first=$1
  while [ ! -f $OUTR/$name/checkpoints/$expected_first ]; do sleep 60; done
  sleep 90
  mkdir -p /mnt/cvr_eval/ckpt_$name
  for f in "$@"; do ionice -c3 cp $OUTR/$name/checkpoints/$f /mnt/cvr_eval/ckpt_$name/; done
  touch /mnt/cvr_eval/ckpt_$name/DONE
  echo "$(date) $name done" >> /root/cvresearch/export_watch.log
}
watch batch5_fna_norgb ViPTrack_thermal_ep0120.pth.tar &
watch batch4_ditto ViPTrack_rgbt_003_ep0120.pth.tar ViPTrack_rgbt_001_ep0120.pth.tar ViPTrack_rgbt_002_ep0120.pth.tar ViPTrack_rgbt_003_ep0120.pth.tar &
watch batch6_e2_indep_ds ViPTrack_ep0120.pth.tar &
watch batch6_e2_bucket_ds ViPTrack_thermal_ep0120.pth.tar &
wait
