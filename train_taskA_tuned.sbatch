#!/bin/bash
#SBATCH --job-name=taskA-tuned
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm-%j.out

set -euo pipefail
source ~/anaconda3/bin/activate base
export HF_DATASETS_CACHE=~/.cache/hf_datasets

cd ~/MS_projects/DL_P/SemEval-2026-Task13

python task_A/scripts/train_taskA_tuned.py \
  --model-name task_A/outputs/enc_taskA/checkpoint-6250 \
  --output-dir task_A/outputs/enc_taskA_tuned \
  --train-path task_A/task_A/train.parquet \
  --dev-path task_A/task_A/dev.parquet \
  --batch-size 8 \
  --accum 2 \
  --epochs 1 \
  --lr 1.5e-5 \
  --warmup 0.1 \
  --max-length 256 \
  --weight-decay 0.01 \
  --label-smoothing 0.05
