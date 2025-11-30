import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


class CodeDataset(Dataset):
    """Simple dataset that stores code strings."""

    def __init__(self, codes):
        self.codes = codes

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx]


def run_inference(
    model_path: Path,
    parquet_path: Path,
    output_csv: Path,
    tokenizer_name: Path | str | None = None,
    batch_size: int = 16,
    max_length: int = 512,
    device: str | None = None,
    id_column: str = "id",
    label_column: str = "label",
    id_source: Path | None = None,
    wandb_settings: dict[str, Any] | None = None,
    local_files_only: bool = True,
) -> None:
    """
    Generate Task A predictions on a parquet file and save a Kaggle-ready CSV.

    If the test parquet has no id column, pass --id-source pointing to the official
    sample_submission.csv so ids match Kaggle. This avoids the "Solution and submission
    values for ID do not match" error.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(parquet_path)
    codes = df["code"].tolist()

    # Resolve ids: prefer id/ID from the test file, otherwise use sample_submission.
    ids = None
    if "id" in df.columns:
        ids = df["id"].tolist()
    elif "ID" in df.columns:
        ids = df["ID"].tolist()
    elif id_source is not None:
        sub_df = pd.read_csv(id_source)
        if "id" not in sub_df.columns and "ID" not in sub_df.columns:
            raise ValueError("id_source must contain an 'id' or 'ID' column")
        ref_ids = sub_df["id"] if "id" in sub_df.columns else sub_df["ID"]
        if len(ref_ids) != len(df):
            raise ValueError(
                f"id_source row count {len(ref_ids)} does not match test rows {len(df)}"
            )
        ids = ref_ids.tolist()
    else:
        raise ValueError(
            "No id column in test parquet. Pass --id-source pointing to the official "
            "sample_submission.csv so ids match Kaggle."
        )

    dataset = CodeDataset(codes)
    tokenizer_id = tokenizer_name or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=local_files_only)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=local_files_only)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    def collate_fn(batch):
        return tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    preds: list[int] = []
    wandb_settings = wandb_settings or {}
    wandb_run = _start_wandb_inference_run(
        wandb_settings,
        {
            "model_path": str(model_path),
            "parquet_path": str(parquet_path),
            "output_csv": str(output_csv),
            "batch_size": batch_size,
            "max_length": max_length,
            "device": device,
            "num_rows": len(dataset),
            "id_column": id_column,
            "label_column": label_column,
            "id_source": str(id_source) if id_source else None,
        },
    )

    try:
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(dim=-1).cpu().tolist())

        if len(ids) != len(preds):
            raise ValueError(f"Length mismatch: ids={len(ids)} preds={len(preds)}")

        out_df = pd.DataFrame({id_column: ids, label_column: preds})
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)

        if wandb_run:
            _log_submission_predictions(
                wandb_run,
                ids,
                preds,
                codes,
                label_column,
                max_rows=int(wandb_settings.get("log_predictions", 0)),
            )
            if wandb_settings.get("upload_submission"):
                _upload_submission_artifact(wandb_run, output_csv)
    finally:
        if wandb_run:
            wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task A inference and write Kaggle CSV.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to finetuned checkpoint directory")
    parser.add_argument(
        "--tokenizer-name",
        type=Path,
        default=None,
        help="Optional tokenizer path/name (defaults to model-path)",
    )
    parser.add_argument("--parquet-path", type=Path, required=True, help="Path to test parquet (e.g., task_A/task_A/test.parquet)")
    parser.add_argument("--output-csv", type=Path, required=True, help="Where to write the submission CSV")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu or cuda")
    parser.add_argument("--id-column", type=str, default="id", help="Submission id column name (Kaggle expects 'id')")
    parser.add_argument("--label-column", type=str, default="label", help="Submission label column name (Kaggle expects 'label')")
    parser.add_argument("--id-source", type=Path, default=None, help="Path to Kaggle sample_submission.csv when test parquet lacks ids")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project (required when --wandb-mode != disabled)")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/account")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group")
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Space-separated list of tags for the W&B run")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Set to 'online' for live logging or 'offline' to sync later",
    )
    parser.add_argument("--wandb-notes", type=str, default=None, help="Optional notes shown in the W&B UI")
    parser.add_argument(
        "--wandb-log-predictions",
        type=int,
        default=0,
        help="Log first N predictions + code previews to W&B (0 disables the preview table)",
    )
    parser.add_argument(
        "--wandb-upload-submission",
        action="store_true",
        help="Upload the generated CSV as a W&B artifact named after the run",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set to false to allow fetching model/tokenizer from the hub if missing locally",
    )
    return parser.parse_args()


def _start_wandb_inference_run(settings: dict[str, Any], config: dict[str, Any]):
    """Initialize a W&B run for inference logging."""
    mode = settings.get("mode", "disabled")
    if mode == "disabled":
        os.environ.setdefault("WANDB_DISABLED", "true")
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Run `pip install wandb` before enabling logging.")
    if not settings.get("project"):
        raise ValueError("Set --wandb-project when enabling W&B logging.")

    clean_config = {k: v for k, v in config.items() if v is not None}
    run = wandb.init(
        project=settings.get("project"),
        entity=settings.get("entity"),
        name=settings.get("run_name"),
        group=settings.get("group"),
        tags=settings.get("tags"),
        notes=settings.get("notes"),
        mode=mode,
        config=clean_config,
    )
    run.summary["num_rows"] = config.get("num_rows")
    return run


def _truncate_code(code: str, limit: int = 320) -> str:
    snippet = code.strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3] + "..."


def _log_submission_predictions(
    run,
    ids: list[Any],
    preds: list[int],
    codes: list[str],
    label_column: str,
    max_rows: int,
):
    """Send prediction stats, histograms, and a short preview table to W&B."""
    run.summary["submission_rows"] = len(preds)
    if preds:
        counts = np.bincount(preds, minlength=max(preds) + 1)
        table = wandb.Table(columns=["label_id", "count"])
        for idx, count in enumerate(counts.tolist()):
            table.add_data(idx, int(count))
        run.log(
            {
                "submission/pred_distribution": table,
                "submission/pred_hist": wandb.plot.bar(
                    table, "label_id", "count", title="Prediction distribution"
                ),
            }
        )

    if max_rows <= 0:
        return
    limit = min(max_rows, len(preds))
    preview = wandb.Table(columns=["row", "id", label_column, "code_preview"])
    for idx in range(limit):
        preview.add_data(
            idx,
            ids[idx],
            int(preds[idx]),
            _truncate_code(codes[idx]),
        )
    run.log({"submission/predictions_head": preview})


def _upload_submission_artifact(run, submission_path: Path):
    artifact = wandb.Artifact(
        name=f"taskA-submission-{run.id}",
        type="submission",
        description="Task A inference CSV generated by predict_taskA_inference.py",
    )
    artifact.add_file(str(submission_path))
    run.log_artifact(artifact)


if __name__ == "__main__":
    args = parse_args()
    wandb_settings = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "run_name": args.wandb_run_name,
        "group": args.wandb_group,
        "tags": args.wandb_tags,
        "mode": args.wandb_mode,
        "notes": args.wandb_notes,
        "log_predictions": args.wandb_log_predictions,
        "upload_submission": args.wandb_upload_submission,
    }
    run_inference(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        parquet_path=args.parquet_path,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        id_column=args.id_column,
        label_column=args.label_column,
        id_source=args.id_source,
        wandb_settings=wandb_settings,
        local_files_only=args.local_files_only,
    )
