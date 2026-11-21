#!/bin/bash
export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/cvresearch/ViPT
PY=/root/miniconda3/envs/sdstrack/bin/python
CK=output/checkpoints/train/vipt
LOG=/root/cvresearch/B_chain4.log
while ! grep -q "^END" /root/cvresearch/B_chain3.log 2>/dev/null; do sleep 30; done
echo "START $(date)" > $LOG
run() { y=$1
  mkdir -p RGBT_workspace/results/LasHeR/$y; cp -n experiments/vipt/batch1_modality.yaml experiments/vipt/$y.yaml
  echo "[$(date +%T)] START $y" >> $LOG
  $PY RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name LasHeR --yaml_name $y --epoch 120 --mode parallel --threads 4 --seq_home /root/cvresearch/data/lasher/testingset > /root/cvresearch/B_eval_$y.log 2>&1
  echo "[$(date +%T)] END $y files=$(ls RGBT_workspace/results/LasHeR/$y | wc -l)" >> $LOG; }
# alpha curve + baselines (thermal), all on B
for y in B_batch2_modality_a075 B_batch1_modality_a05 B_batch2_modality_a025 B_batch1_fedprox B_batch2_fedprox_mu01 B_batch1_ours; do run $y; done
# zero-aux (missing modality) replication on B: copies of the FedAvg and modality thermal checkpoints
for v in fedavg modality; do mkdir -p $CK/B_noaux_$v; cp $CK/batch1_$v/ViPTrack_ep0120.pth.tar $CK/B_noaux_$v/; done
export CVR_ZERO_AUX=1
run B_noaux_fedavg
run B_noaux_modality
unset CVR_ZERO_AUX
echo "END $(date)" >> $LOG
