#!/bin/bash
# Independent thermal and independent depth WITHOUT the pure-RGB client, both on GPU2 after the E2 bucket arm finishes.
OUTR=/root/cvresearch/output
launch() { name=$1; clients=$2
  while pgrep -f "batch6_e2_bucket_ds" > /dev/null; do sleep 30; done
  sleep 30
  [ -e $OUTR/$name ] && { echo "$name exists" >> /root/cvresearch/launch3.err; return; }
  cd /root/cvresearch/ViPT
  export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2
  nohup /root/miniconda3/envs/sdstrack/bin/python lib/federated/server.py --variant modality --clients $clients --rounds 120 --local-steps 50 --seed 42 --workers 0 --clients-per-round 1.0 --save-every 20 --export-at-end --out-dir $OUTR/$name > $OUTR/logs/$name.log 2>&1 &
  echo "$(date) started $name" >> /root/cvresearch/launch3.log
}
launch batch9_indep_thermal_norgb rgbt_001,rgbt_002,rgbt_003 &
launch batch9_indep_depth_norgb rgbd_001,rgbd_002,rgbd_003 &
wait
