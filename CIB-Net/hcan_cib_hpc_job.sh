#!/bin/bash
#SBATCH --job-name=HCAN_CIB
#SBATCH --output=hcan_cib_%j.log
#SBATCH --error=hcan_cib_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=node1
#SBATCH --gres=shard:20

cd $SLURM_SUBMIT_DIR

# ==========================================
# 1. HPC Environment Setup
# ==========================================
source $HOME/miniconda3/bin/activate
pip install pyyaml tqdm

echo "--- HPC GPU DIAGNOSTICS ---"
nvidia-smi || echo "nvidia-smi command failed or not found!"
python -c "import torch; print('PyTorch version:', torch.__version__); print('PyTorch built with CUDA:', torch.version.cuda); print('PyTorch CUDA is_available():', torch.cuda.is_available())"
echo "---------------------------"

echo "Starting HCAN-CIB Unified Pipeline Training on HPC..."
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

# ==========================================
# 2. Run Pipeline
# ==========================================
# Using -u for unbuffered output to see logs in real-time
python -u experiments/run_hcan_cib.py

echo "Job finished successfully!"
