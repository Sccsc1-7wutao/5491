#!/bin/bash
export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/cvresearch/ViPT
PY=/root/miniconda3/envs/sdstrack/bin/python
CK=output/checkpoints/train/vipt
echo "START $(date)" > /root/cvresearch/B_chain.log
for y in batch1_fedavg batch2_indep_thermal batch1_modality rep_modality; do
  mkdir -p RGBT_workspace/results/LasHeR/$y
  echo "[$(date +%T)] START $y" >> /root/cvresearch/B_chain.log
  $PY RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name LasHeR --yaml_name $y --epoch 120 --mode parallel --threads 4 --seq_home /root/cvresearch/data/lasher/testingset > /root/cvresearch/B_eval_$y.log 2>&1
  echo "[$(date +%T)] END $y rc=$? files=$(ls RGBT_workspace/results/LasHeR/$y | wc -l)" >> /root/cvresearch/B_chain.log
done
echo "END $(date)" >> /root/cvresearch/B_chain.log
