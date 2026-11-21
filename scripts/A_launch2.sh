#!/bin/bash
# Two control runs WITHOUT the pure-RGB clients: main method (aux_specific bucketed) on GPU0 after the indep-event run, FedAvg on GPU1 after the E2 independent arm.
OUTR=/root/cvresearch/output
CL=rgbt_001,rgbt_002,rgbt_003,rgbd_001,rgbd_002,rgbd_003,rgbe_000,rgbe_002,rgbe_003
launch() { name=$1; gpu=$2; variant=$3; waitpat=$4
  while pgrep -f "$waitpat" > /dev/null; do sleep 30; done
  sleep 30
  [ -e $OUTR/$name ] && { echo "$name exists" >> /root/cvresearch/launch2.err; return; }
  cd /root/cvresearch/ViPT
  export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$gpu
  nohup /root/miniconda3/envs/sdstrack/bin/python lib/federated/server.py --variant $variant --clients $CL --rounds 120 --local-steps 50 --seed 42 --workers 0 --clients-per-round 1.0 --save-every 20 --export-at-end --out-dir $OUTR/$name > $OUTR/logs/$name.log 2>&1 &
  echo "$(date) started $name" >> /root/cvresearch/launch2.log
}
launch batch8_main_norgb 0 modality "batch7_indep_event_norgb" &
launch batch8_fedavg_norgb 1 fedavg "batch6_e2_indep_ds" &
wait
