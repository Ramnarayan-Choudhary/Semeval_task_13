# SemEval 2026 Task 13 – Task A (Machine-Generated Code Detection)

This folder collects data and code for the Task A binary classification problem: detect whether a code snippet is **human-written (0)** or **machine-generated (1)**.

## Layout
- `task_A_data/`: train/dev/test parquet files (`code`, `generator`, `label`, `language`), label maps, and sample submissions.
- `Approaches/code_bert_finetune/`: CodeBERT fine-tuning and inference scripts plus a sample submission CSV.
- `Approaches/prompting_qwen/`: Few-shot prompting with Qwen (LLM-only, no training) and a prediction checkpoint CSV.
- `Approaches/gpt_oss/`: LoRA fine-tuning experiments on `unsloth/gpt-oss-20b` (notebooks, adapter weights, prediction CSVs).

## Data
- Train: `task_A_data/train.parquet` (500k rows), Dev: `task_A_data/dev.parquet` (100k), Test: `task_A_data/test.parquet` (1k).
- Columns: `code` (source), `generator` (Human/AI), `label` (0=Human, 1=AI), `language`.
- Mapping files: `task_A_data/id_to_label.json`, `label_to_id.json` (identity for this binary task).
- `sample_submission.csv` holds the official ID/layout expected for submissions; use it when your test parquet lacks an `id` column.

## Environment
- Python 3.10+ recommended; GPU strongly preferred for training.
- Baseline dependencies: `torch`, `transformers`, `datasets`, `scikit-learn`, `pandas`, `pyarrow`, `accelerate`, `wandb` (optional for logging), `peft`/`bitsandbytes` for LoRA work.
- Prompting deps: `pip install -r Approaches/prompting_qwen/requirements_prompting.txt`.
- If you want to stay offline, set `HF_HUB_OFFLINE=1` or pass `--local-files-only` to the scripts; otherwise ensure the base models are cached or allow hub access.

## CodeBERT fine-tuning baseline
Training (saves best checkpoint under `--output-dir/best`):
```bash
python Approaches/code_bert_finetune/train_taskA_tuned.py \
  --model-name microsoft/codebert-base \
  --output-dir runs/codebert \
  --train-path task_A_data/train.parquet \
  --dev-path task_A_data/dev.parquet \
  --batch-size 8 --accum 2 --epochs 3 --lr 1.5e-5 --max-length 512 \
  --weight-decay 0.01 --label-smoothing 0.05 --seed 42 \
  --wandb-mode disabled   # enable + set --wandb-project to log to W&B
```

Inference to Kaggle-ready CSV (handles missing IDs via `--id-source`):
```bash
python Approaches/code_bert_finetune/predict_taskA_inference.py \
  --model-path runs/codebert/best \
  --parquet-path task_A_data/test.parquet \
  --id-source task_A_data/sample_submission.csv \
  --output-csv submission_taskA_test.csv \
  --batch-size 16 --max-length 512 --local-files-only true
```
Utility: `Approaches/code_bert_finetune/find_taskA_dataset.py` searches the HF Hub for the competition dataset ID.

## Prompting with Qwen
- Script: `Approaches/prompting_qwen/prompting_approach1.py`.
- Strategy: Jaccard-based few-shot retrieval from the train pool, short system prompt tuned for code forensics, greedy decoding (`max_new_tokens=6`, `temperature=0`).
- Paths: the script expects data under `Approaches/task_A/`; either symlink the data (`ln -s ../task_A_data Approaches/task_A`) or adjust the paths inside the script before running.
- Defaults: model `Qwen/Qwen2.5-1.5B-Instruct`, local-only loading, checkpoints to `predictions_checkpoint.csv`, final CSV to `final_predictions.csv`.
- Run example:
```bash
cd Approaches/prompting_qwen
SAMPLE_SUBMISSION=../../task_A_data/sample_submission.csv LOCAL_FILES_ONLY=1 \
python prompting_approach1.py
```

## GPT-OSS LoRA experiments
- Artifacts: LoRA adapters under `Approaches/gpt_oss/outputs/` and `Approaches/gpt_oss/finetuned_gptoss_lora_2/`, plus prediction file `Approaches/gpt_oss/q_predictions.csv`.
- Training: driven from the notebooks `Approaches/gpt_oss/peft_gpt_oss_20b.ipynb` (and copy under `gpt_oss/`); uses the Unsloth TRL stack (`unsloth/gpt-oss-20b-unsloth-bnb-4bit`) with SFT/PEFT.
- Inference: load the base model with PEFT adapters via `peft`/`transformers`, then mirror the CodeBERT prediction flow to produce a submission CSV.

## Existing outputs
- CodeBERT sample submission: `Approaches/code_bert_finetune/submission_taskA_test_f (2).csv`.
- Qwen checkpoint/preds: `Approaches/prompting_qwen/predictions_checkpoint.csv`.
- GPT-OSS preds: `Approaches/gpt_oss/q_predictions.csv`.

## Tips
- For W&B logging, pass `--wandb-project` and optionally `--wandb-group/--wandb-tags`; use `--wandb-upload-checkpoint` to push the best model as an artifact.
- If test data lacks an `id` column, always provide `--id-source task_A_data/sample_submission.csv` to avoid submission format errors.
- Upgrade `bottleneck` (`pip install -U bottleneck`) to silence the pandas warning emitted when reading the parquet files.
