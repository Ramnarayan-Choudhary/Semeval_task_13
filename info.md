## Task A – What we ran and results

- **Base checkpoint used for inference:** `task_A/outputs/enc_taskA/checkpoint-6250` (RobertaForSequenceClassification).
- **Eval (dev) from trainer_state:** `eval_f1`≈0.9911, `eval_acc`≈0.9911, `eval_loss`≈0.034 (from `task_A/outputs/enc_taskA/checkpoint-6250/trainer_state.json`).
- **Public test submission score (Kaggle):** 0.32196 macro F1 using the base checkpoint + sample_submission IDs.
- **Inference command (GPU):**
  ```bash
  python task_A/scripts/predict_taskA_inference.py \
    --model-path task_A/outputs/enc_taskA/checkpoint-6250 \
    --parquet-path task_A/task_A/test.parquet \
    --id-source task_A/task_A/sample_submission.csv \
    --output-csv task_A/outputs/submission_taskA_test.csv \
    --batch-size 16 \
    --max-length 512 \
    --device cuda \
    --id-column id \
    --label-column label
  ```
  Notes: `id-source` must be the official `sample_submission.csv` to match Kaggle IDs.

### Tuned training attempt (to improve from ~32 → target ~40)

- **Script:** `task_A/scripts/train_taskA_tuned.py` (class weights, label smoothing, warmup, lower LR). Uses slow tokenizer (`use_fast=False`) to avoid tokenizer parse error with the local checkpoint.
- **Planned hyperparameters:** `epochs=3`, `lr=1.5e-5`, `warmup=0.1`, `batch=8`, `accum=2`, `max_length=512`, `weight_decay=0.01`, `label_smoothing=0.05`.
- **Training command (from repo root):**
  ```bash
  python task_A/scripts/train_taskA_tuned.py \
    --model-name task_A/outputs/enc_taskA/checkpoint-6250 \
    --output-dir task_A/outputs/enc_taskA_tuned \
    --train-path task_A/task_A/train.parquet \
    --dev-path task_A/task_A/dev.parquet \
    --batch-size 8 \
    --accum 2 \
    --epochs 3 \
    --lr 1.5e-5 \
    --warmup 0.1 \
    --max-length 512 \
    --weight-decay 0.01 \
    --label-smoothing 0.05
  ```
- **Requirements:** Install/update `accelerate` (`pip install -U 'accelerate>=0.21.0'`). To reuse cached tokenization, set `export HF_DATASETS_CACHE=~/.cache/hf_datasets`.
- **Status/issues:** Previous runs were cancelled by SLURM time limits during tokenization; rerun with a longer job (`--time` ≥ 8h, `--cpus-per-task` ≥ 8). Once training finishes, run inference with `--model-path task_A/outputs/enc_taskA_tuned/best` and submit the resulting CSV.

### Live experiment tracking with Weights & Biases

All training and inference scripts now have first-class [W&B](https://wandb.ai/) support so we can inspect runs in real time (loss curves, confusion matrices, submission artifacts, etc.).

1. Install & login once per machine:
   ```bash
   pip install wandb
   wandb login
   ```
2. Enable logging when launching training or inference by passing `--wandb-mode online` (or `offline`) plus a project name. Example GPU training run:
   ```bash
   python task_A/scripts/train_taskA_tuned.py \
     --model-name task_A/outputs/enc_taskA/checkpoint-6250 \
     --output-dir task_A/outputs/enc_taskA_tuned \
     --train-path task_A/task_A/train.parquet \
     --dev-path task_A/task_A/dev.parquet \
     --batch-size 8 --accum 2 --epochs 3 \
     --lr 1.5e-5 --warmup 0.1 --max-length 512 \
     --weight-decay 0.01 --label-smoothing 0.05 \
     --wandb-mode online --wandb-project semeval-taskA \
     --wandb-run-name tuned-codebert \
     --wandb-log-predictions 200 \
     --wandb-upload-checkpoint
   ```
   The run logs config, dataset sizes, class imbalance plots, per-epoch metrics, a dev-set confusion matrix, sample predictions, and optionally uploads the best checkpoint as a `model` artifact.
3. Inference logging works the same way and can also push the CSV submission as an artifact:
   ```bash
   python task_A/scripts/predict_taskA_inference.py \
     --model-path task_A/outputs/enc_taskA_tuned/best \
     --parquet-path task_A/task_A/test.parquet \
     --id-source task_A/task_A/sample_submission.csv \
     --output-csv task_A/outputs/submission_taskA_test_tuned.csv \
     --batch-size 16 --max-length 512 --device cuda \
     --wandb-mode online --wandb-project semeval-taskA \
     --wandb-run-name tuned-inference \
     --wandb-log-predictions 50 \
     --wandb-upload-submission
   ```

Key CLI knobs (shared between scripts):

- `--wandb-project / --wandb-entity / --wandb-group / --wandb-tags / --wandb-run-name / --wandb-notes`
- `--wandb-mode {online,offline,disabled}` controls whether logging is active.
- `--wandb-log-predictions N` uploads confusion matrices plus the first `N` predictions as a preview table.
- `--wandb-upload-checkpoint` (training) and `--wandb-upload-submission` (inference) create versioned artifacts so we can download exact models/CSVs later.

### Quick “next submission” steps
1) Ensure `accelerate` is installed.  
2) Launch a longer GPU job (e.g., `srun --partition=gpu --gres=gpu:1 --time=08:00:00 --cpus-per-task=8 --mem=48G --pty bash`).  
3) Train with the command above (caching should make tokenization quick).  
4) Predict with the tuned checkpoint and submit `task_A/outputs/submission_taskA_test_tuned.csv` to Kaggle.
