import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional at runtime
    wandb = None


class WeightedTrainer(Trainer):
    """Trainer that supports class weights and label smoothing."""

    def __init__(self, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.label_smoothing)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_and_tokenize(tokenizer, train_path, dev_path, max_length):
    data_files = {"train": train_path, "dev": dev_path}
    ds = load_dataset("parquet", data_files=data_files)

    def tok(batch):
        return tokenizer(batch["code"], truncation=True, padding="max_length", max_length=max_length)

    train_ds = ds["train"].map(tok, batched=True)
    dev_ds = ds["dev"].map(tok, batched=True)

    keep_cols = {"input_ids", "attention_mask", "label"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    dev_ds = dev_ds.remove_columns([c for c in dev_ds.column_names if c not in keep_cols])

    train_ds = train_ds.rename_column("label", "labels")
    dev_ds = dev_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch")
    dev_ds.set_format(type="torch")
    return train_ds, dev_ds


def get_class_weights(train_ds):
    labels_np = np.array(train_ds["labels"])
    num_classes = int(labels_np.max()) + 1
    counts = np.bincount(labels_np, minlength=num_classes).astype(float)
    weights = counts.sum() / (counts + 1e-9)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float), counts.astype(int)


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "acc": accuracy_score(labels, preds),
    }


def _convert_args_to_config(args: argparse.Namespace) -> dict[str, Any]:
    """Drop non-serializable fields (Paths) so they can be logged to W&B config."""
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key.startswith("wandb_") or key == "no_wandb":
            continue
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def _start_wandb_run(
    args: argparse.Namespace,
    train_ds,
    dev_ds,
    class_weights: torch.Tensor,
    class_counts: np.ndarray,
) -> Any | None:
    """Initialize a W&B run and log dataset statistics."""
    wandb_enabled = args.wandb_mode != "disabled" and not args.no_wandb
    if not wandb_enabled:
        os.environ.setdefault("WANDB_DISABLED", "true")
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Run `pip install wandb` before enabling logging.")
    if not args.wandb_project:
        raise ValueError("Set --wandb-project when enabling Weights & Biases logging.")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
        config={
            **_convert_args_to_config(args),
            "train_examples": len(train_ds),
            "dev_examples": len(dev_ds),
            "num_labels": int(class_weights.numel()),
        },
    )

    table = wandb.Table(columns=["label_id", "train_count", "class_weight"])
    for idx, (count, weight) in enumerate(zip(class_counts.tolist(), class_weights.tolist())):
        table.add_data(idx, int(count), float(weight))
    run.log(
        {
            "data/class_balance": table,
            "data/train_class_hist": wandb.plot.bar(
                table, "label_id", "train_count", title="Task A train class distribution"
            ),
        }
    )
    run.summary["train_examples"] = len(train_ds)
    run.summary["dev_examples"] = len(dev_ds)
    return run


def _log_eval_predictions(
    run,
    trainer: Trainer,
    eval_ds,
    max_rows: int,
    class_counts: np.ndarray,
):
    """Log evaluation confusion matrix and sampled predictions to W&B."""
    if run is None or max_rows < 0:
        return
    predictions = trainer.predict(eval_ds)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    class_names = [f"class_{idx}" for idx in range(len(class_counts))]

    run.log(
        {
            "eval/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels.tolist(),
                preds=preds.tolist(),
                class_names=class_names,
            )
        },
        step=trainer.state.global_step,
    )

    if max_rows == 0:
        return
    limit = min(max_rows, len(labels))
    table = wandb.Table(columns=["row", "label_id", "prediction", "correct"])
    for idx in range(limit):
        table.add_data(
            idx,
            int(labels[idx]),
            int(preds[idx]),
            bool(labels[idx] == preds[idx]),
        )
    run.log({"eval/predictions_head": table}, step=trainer.state.global_step)


def _log_best_checkpoint(run, best_dir: Path, artifact_name: str | None):
    if run is None or not best_dir.exists():
        return
    run.summary["best_checkpoint"] = str(best_dir)
    if artifact_name:
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_dir(str(best_dir))
        run.log_artifact(artifact)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="microsoft/codebert-base")
    ap.add_argument("--tokenizer-name", type=str, default=None, help="Optional tokenizer path; defaults to model-name")
    ap.add_argument("--use-fast-tokenizer", action="store_true", help="Use fast tokenizer when available (auto-falls back)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--train-path", type=str, required=True)
    ap.add_argument("--dev-path", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project (required when --wandb-mode != disabled)")
    ap.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/account")
    ap.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name")
    ap.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group for multi-run studies")
    ap.add_argument("--wandb-tags", nargs="*", default=None, help="Space-separated list of W&B tags")
    ap.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Set to 'online' (default live logging) or 'offline' to sync later",
    )
    ap.add_argument("--wandb-notes", type=str, default=None, help="Optional notes field visible in the W&B UI")
    ap.add_argument("--wandb-watch", action="store_true", help="Log gradients/weights via wandb.watch()")
    ap.add_argument(
        "--wandb-log-predictions",
        type=int,
        default=0,
        help="Log first N dev predictions to W&B and always log a confusion matrix (0 disables the table)",
    )
    ap.add_argument(
        "--wandb-upload-checkpoint",
        action="store_true",
        help="Upload the best checkpoint directory as a W&B model artifact",
    )
    ap.add_argument("--no-wandb", action="store_true", help="Deprecated alias for --wandb-mode disabled")
    args = ap.parse_args()

    if args.no_wandb:
        args.wandb_mode = "disabled"

    tokenizer_name = args.tokenizer_name or args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=args.use_fast_tokenizer)
    except Exception as exc:
        # Some models (e.g., CodeT5+) may not ship a fast tokenizer; retry with slow if requested fast.
        if args.use_fast_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        else:
            raise exc
    # T5/CodeT5 models require right padding and a pad token; align both to avoid <eos> mismatch errors.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds, dev_ds = load_and_tokenize(tokenizer, args.train_path, args.dev_path, args.max_length)
    class_weights, class_counts = get_class_weights(train_ds)
    num_labels = int(class_weights.numel())
    wandb_run = _start_wandb_run(args, train_ds, dev_ds, class_weights, class_counts)
    report_to = "wandb" if wandb_run is not None else "none"
    run_name = args.wandb_run_name or (wandb_run.name if wandb_run is not None else None)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Older transformers versions use `eval_strategy` instead of `evaluation_strategy`.
    strategy_key = (
        "evaluation_strategy"
        if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames
        else "eval_strategy"
    )
    training_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup,
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_steps": 200,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "seed": args.seed,
        "report_to": report_to,
        "run_name": run_name,
    }
    training_args = TrainingArguments(**training_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    if wandb_run and args.wandb_watch:
        wandb.watch(model, log="all", log_freq=training_args.logging_steps)

    best_dir = Path(args.output_dir) / "best"
    try:
        trainer.train()
        trainer.save_model(str(best_dir))
        if wandb_run:
            final_metrics = trainer.evaluate()
            wandb_run.log(
                {f"final/{k}": v for k, v in final_metrics.items()},
                step=trainer.state.global_step,
            )
            wandb_run.summary["best_metric"] = trainer.state.best_metric
            wandb_run.summary["best_model_checkpoint"] = trainer.state.best_model_checkpoint
            _log_eval_predictions(
                wandb_run,
                trainer,
                dev_ds,
                args.wandb_log_predictions,
                class_counts,
            )
            artifact_name = None
            if args.wandb_upload_checkpoint:
                slug = run_name or best_dir.name
                artifact_name = f"{slug}-best".replace(" ", "-")
            _log_best_checkpoint(wandb_run, best_dir, artifact_name)
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
