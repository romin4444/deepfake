#!/bin/bash
# submit_job.sh — SLURM job script for HPC clusters
# Usage: sbatch submit_job.sh

#SBATCH --job-name=dfvideo_train
#SBATCH --partition=gpu             # GPU partition name (check with 'sinfo')
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --mem=32G                   # RAM
#SBATCH --time=48:00:00             # 48 hours max
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_%j.log  # log file
#SBATCH --error=logs/slurm_%j.err

# Load modules (cluster-specific; check 'module avail')
module load cuda/12.1
module load python/3.10

# Create log dir
mkdir -p logs

# Activate virtualenv or conda
# conda activate dfvideo
# or:
# source venv/bin/activate

cd /scratch/user/dfvideo  # or wherever your code is

# Run training
python -m src.train \
  --config configs/default.yaml \
  model.backbone=clip_vit_l14 \
  model.peft=lora \
  data.train_datasets=[faceforensics] \
  train.epochs=20 \
  output_dir=/scratch/user/dfvideo/outputs \
  device=cuda

echo "Job finished at $(date)"
