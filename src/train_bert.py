"""Fine-tune CAMeLBERT to score عرنجية vs فصحى.

Uses the v3 setup:
  - strip diacritics, digits, honorifics and punctuation
  - fold alef variants and ى→ي, keep ة distinct
  - max_length=256, 3 epochs, effective batch size 32
  - pick the best checkpoint by lowest dev eval_loss (dev set only)

Example:
  python src/train_bert.py --data data/processed_v3 --out models/v3_camelbert --seed 42
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import full_report, load_hard, read_jsonl  # noqa: E402
from normalize import normalize  # noqa: E402


def prep(text: str) -> str:
    return normalize(text, keep_punct=False, fold_letters=False, fold_alef_ya=True)


def main():
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_v3")
    ap.add_argument("--hard", default="data/hard_test.jsonl")
    ap.add_argument("--model", default="CAMeL-Lab/bert-base-arabic-camelbert-mix")
    ap.add_argument("--out", default="models/v3_camelbert")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=0, help="subsample training rows (0 = all)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "fusha", 1: "aranjiya"},
        label2id={"fusha": 0, "aranjiya": 1},
    )

    def to_ds(rows):
        ds = Dataset.from_dict({"text": [prep(r["text"]) for r in rows], "label": [r["label"] for r in rows]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)

    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    dev_rows = read_jsonl(os.path.join(args.data, "dev.jsonl"))
    train_ds, dev_ds = to_ds(train_rows), to_ds(dev_rows)

    def metrics(ev):
        logits, labels = ev
        return {"acc": float((logits.argmax(-1) == labels).mean())}

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.bs * 2,
        warmup_steps=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )
    trainer.train()

    def probs(rows):
        out = trainer.predict(to_ds(rows)).predictions
        e = np.exp(out - out.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True))[:, 1]

    results = {
        "seed": args.seed,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_loss": round(float(trainer.state.best_metric), 4),
        "dev": full_report("dev", dev_rows, probs(dev_rows)),
    }
    for fname, label in [
        ("test.jsonl", "test"),
        ("test_holdout_haiku.jsonl", "test_holdout_haiku"),
        ("test_holdout_system.jsonl", "test_holdout_system"),
    ]:
        p = os.path.join(args.data, fname)
        if os.path.exists(p):
            rows = read_jsonl(p)
            if rows:
                results[label] = full_report(label, rows, probs(rows))
    if os.path.exists(args.hard):
        hard = load_hard(args.hard)
        hp = probs(hard)
        results["hard"] = full_report("hard test", hard, hp)
        results["hard_rows"] = [
            {"id": r["id"], "label": r["label"], "p_aranjiya": round(float(p), 3), "text": r["text"]}
            for r, p in zip(hard, hp)
        ]

    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", final)


if __name__ == "__main__":
    main()
