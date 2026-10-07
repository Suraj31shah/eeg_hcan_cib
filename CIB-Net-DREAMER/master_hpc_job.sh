#!/bin/bash
#SBATCH --job-name=CIB_Master
#SBATCH --output=master_job_%j.log
#SBATCH --error=master_job_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=node1
#SBATCH --gres=shard:20

cd $SLURM_SUBMIT_DIR

# ==========================================
# MASTER SCRIPT: RUNS BASELINE THEN CIB-NET
# ==========================================
echo "=========================================="
echo "Starting Master Job..."
echo "Node: $HOSTNAME"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

echo "[1/2] Skipping Phase 0 Baselines (Already Done)..."
# bash baselines_hpc_job.sh

echo "=========================================="
echo "[2/2] Executing CIB-Net Training..."
# Running it as a standard bash script (ignores its internal #SBATCH headers)
bash cib_net_hpc_job.sh

echo "=========================================="
echo "Master Job Complete!"
echo "=========================================="
