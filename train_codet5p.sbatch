#!/bin/bash
#SBATCH --job-name=codet5p-taskA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm-codet5p-%j.out

set -euo pipefail
source ~/anaconda3/bin/activate base
export HF_DATASETS_CACHE=~/.cache/hf_datasets

cd ~/MS_projects/DL_P/SemEval-2026-Task13

python task_A/scripts/train_taskA_tuned.py \
  --model-name Salesforce/codet5p-220m \
  --tokenizer-name Salesforce/codet5p-220m \
  --output-dir task_A/outputs/codet5p_ft \
  --train-path task_A/task_A/train.parquet \
  --dev-path task_A/task_A/dev.parquet \
  --batch-size 4 \
  --accum 1 \
  --epochs 0.05 \
  --lr 1.5e-5 \
  --warmup 0.1 \
  --max-length 256 \
  --weight-decay 0.01 \
  --label-smoothing 0.05 \
  --wandb-mode online \
  --wandb-project semeval-task11 \
  --wandb-entity personal_rc \
  --wandb-run-name codet5p-ft-sbatch \
  --wandb-tags codet5p taskA
