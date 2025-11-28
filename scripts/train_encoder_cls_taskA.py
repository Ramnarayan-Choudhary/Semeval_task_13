import os, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

TASK_DIR = os.environ.get("TASK_DIR", "task_A/task_A")
OUT_DIR  = os.environ.get("OUT_DIR",  "task_A/outputs/enc_taskA")
BASE     = os.environ.get("BASE_MODEL","microsoft/codebert-base")
MAX_TRAIN= int(os.environ.get("MAX_TRAIN", 0))
MAX_DEV  = int(os.environ.get("MAX_DEV",   0))

train = load_dataset("parquet", data_files=f"{TASK_DIR}/train.parquet")["train"]
dev   = load_dataset("parquet", data_files=f"{TASK_DIR}/dev.parquet")["train"]

if MAX_TRAIN: train = train.select(range(min(MAX_TRAIN, len(train))))
if MAX_DEV:   dev   = dev.select(range(min(MAX_DEV,   len(dev))))

n_classes = int(max(train["label"])) + 1

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
def pp(b):
    e = tok(b["code"], truncation=True, padding=False, max_length=512)
    e["labels"] = b["label"]; return e
cols = train.column_names
train = train.map(pp, batched=True, remove_columns=cols)
dev   = dev.map(pp,   batched=True, remove_columns=cols)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=n_classes, use_safetensors=True
)

acc = evaluate.load("accuracy"); f1 = evaluate.load("f1")
def metrics(p):
    pr = np.argmax(p.predictions, axis=1)
    return {
        "acc": acc.compute(predictions=pr, references=p.label_ids)["accuracy"],
        "f1":  f1.compute(predictions=pr, references=p.label_ids, average="macro")["f1"],
    }

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=int(os.environ.get("BSZ",16)),
    per_device_eval_batch_size=64,
    learning_rate=float(os.environ.get("LR",2e-5)),
    num_train_epochs=int(os.environ.get("EPOCHS",1)),
    eval_strategy="epoch",              # <-- new name in transformers 4.57
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    bf16=torch.cuda.is_available(),
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                  tokenizer=tok, compute_metrics=metrics)
trainer.train()
trainer.save_model(os.path.join(OUT_DIR, "best"))
