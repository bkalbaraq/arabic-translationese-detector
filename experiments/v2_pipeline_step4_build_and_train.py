"""v2 Pipeline Step 4: Build v2a & v2b datasets and train TF-IDF + CAMeLBERT models.

NEVER modifies v1 datasets (data/processed/) or v1 models (models/baseline/, models/camelbert/).
Strict dev-only model selection guard: best CAMeLBERT epoch is selected strictly on dev.jsonl.

Outputs:
  data/processed_v2a/      (no OPUS in training)
  data/processed_v2b/      (v2a + 15% OPUS News-Commentary & GlobalVoices ONLY in training)
  models/v2a_baseline/     (TF-IDF + LogReg)
  models/v2b_baseline/     (TF-IDF + LogReg)
  models/v2a_camelbert/    (CAMeLBERT-mix, max_len=256, fold_alef_ya=True, best on dev)
  models/v2b_camelbert/    (CAMeLBERT-mix, max_len=256, fold_alef_ya=True, best on dev)
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import joblib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from build_dataset import bucket  # noqa: E402
from evaluate import full_report  # noqa: E402
from normalize import normalize  # noqa: E402
from spans import token_labels  # noqa: E402
from train_baseline import Baseline  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_datasets():
    rng = random.Random(11)
    pair_files = [
        os.path.join(ROOT, "data", "raw", "pairs_v2_existing_clean.jsonl"),
        os.path.join(ROOT, "data", "raw", "pairs_v2_stepD_clean.jsonl"),
    ]
    all_pairs = []
    seen_pids = set()
    for pf in pair_files:
        for r in read_jsonl(pf):
            if r["pair_id"] in seen_pids:
                continue
            seen_pids.add(r["pair_id"])
            all_pairs.append(r)

    splits_a = {"train": [], "dev": [], "test": []}
    used_fusha = set()
    for r in all_pairs:
        sp = bucket(r.get("author") or r["pair_id"], 0.1, 0.1)
        sys_name = f"{r['forward']}>{r['backward']}"
        used_fusha.add(r["pair_id"])
        splits_a[sp].append({
            "text": r["fusha"],
            "label": 0,
            "source": "openiti",
            "pair_id": r["pair_id"],
            "group": r.get("author"),
            "genre": r.get("genre"),
            "origin": r.get("origin", "unknown"),
        })
        toks, labs = token_labels(r["fusha"], r["aranjiya"])
        splits_a[sp].append({
            "text": r["aranjiya"],
            "label": 1,
            "source": f"bt:{sys_name}",
            "pair_id": r["pair_id"],
            "group": r.get("author"),
            "genre": r.get("genre"),
            "origin": r.get("origin", "unknown"),
            "tokens": toks,
            "token_labels": labs,
        })

    # Haiku held-out test set (strictly test-split authors, never in training)
    haiku_pairs = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v2_haiku_holdout.jsonl"))
    holdout_rows = []
    for r in haiku_pairs:
        sp = bucket(r.get("author") or r["pair_id"], 0.1, 0.1)
        assert sp == "test", f"Non-test author leaked into Haiku holdout: {r.get('author')}"
        holdout_rows.append({
            "text": r["fusha"], "label": 0, "source": "openiti", "pair_id": r["pair_id"],
            "group": r.get("author"), "genre": r.get("genre"),
        })
        holdout_rows.append({
            "text": r["aranjiya"], "label": 1, "source": "bt:gemini>claude_haiku", "pair_id": r["pair_id"],
            "group": r.get("author"), "genre": r.get("genre"),
        })

    # Save v2a (no OPUS)
    dir_a = os.path.join(ROOT, "data", "processed_v2a")
    os.makedirs(dir_a, exist_ok=True)
    rep_a = {}
    for sp, rows in splits_a.items():
        rows_copy = list(rows)
        rng.shuffle(rows_copy)
        for i, r in enumerate(rows_copy):
            r["id"] = f"{sp}-{i}"
        write_jsonl(os.path.join(dir_a, f"{sp}.jsonl"), rows_copy)
        rep_a[sp] = {
            "rows": len(rows_copy),
            "labels": dict(Counter(r["label"] for r in rows_copy)),
            "sources": dict(Counter(r["source"] for r in rows_copy)),
            "origins": dict(Counter(r.get("origin") for r in rows_copy)),
            "authors": len({r["group"] for r in rows_copy}),
        }
    for i, r in enumerate(holdout_rows):
        r["id"] = f"haiku-ho-{i}"
    write_jsonl(os.path.join(dir_a, "test_holdout_haiku.jsonl"), holdout_rows)
    rep_a["test_holdout_haiku"] = {"rows": len(holdout_rows), "pairs": len(holdout_rows) // 2}
    with open(os.path.join(dir_a, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(rep_a, f, ensure_ascii=False, indent=2)

    # Build v2b = v2a + OPUS News-Commentary & GlobalVoices ONLY in train, capped at ~15% of عرنجية side
    # Keep dev.jsonl, test.jsonl, and test_holdout_haiku.jsonl IDENTICAL to v2a!
    dir_b = os.path.join(ROOT, "data", "processed_v2b")
    os.makedirs(dir_b, exist_ok=True)
    n_bt_train = sum(1 for r in splits_a["train"] if r["label"] == 1)
    opus_cap = int(round(n_bt_train * 0.15 / 0.85))  # so opus / (n_bt + opus) ~= 15%
    per_corpus = opus_cap // 2

    human_all = read_jsonl(os.path.join(ROOT, "data", "raw", "human_translated.jsonl"))
    nc_train = [h for h in human_all if h["source"] == "opus:News-Commentary" and bucket(h["id"], 0.1, 0.1) == "train"]
    gv_train = [h for h in human_all if h["source"] == "opus:GlobalVoices" and bucket(h["id"], 0.1, 0.1) == "train"]
    rng.shuffle(nc_train)
    rng.shuffle(gv_train)
    chosen_opus = nc_train[:per_corpus] + gv_train[: (opus_cap - per_corpus)]

    # Balance with equal number of unpaired train-split fusha rows from fusha.jsonl (unused ids, no stray filenames)
    unpaired_fusha = []
    with open(os.path.join(ROOT, "data", "raw", "fusha.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in used_fusha:
                continue
            if bucket(r["author"], 0.1, 0.1) != "train":
                continue
            if re.search(r"[A-Za-z]{3,}", r["text"]):
                continue
            unpaired_fusha.append(r)
    rng.shuffle(unpaired_fusha)
    chosen_fusha = unpaired_fusha[: len(chosen_opus)]

    train_b = list(splits_a["train"])
    for h in chosen_opus:
        train_b.append({
            "text": h["text"], "label": 1, "source": h["source"], "pair_id": None,
            "group": h["source"], "genre": h.get("genre"), "origin": "opus_15pct",
        })
    for r in chosen_fusha:
        train_b.append({
            "text": r["text"], "label": 0, "source": "openiti", "pair_id": None,
            "group": r["author"], "genre": r.get("genre"), "origin": "openiti_balance",
        })
    rng.shuffle(train_b)
    for i, r in enumerate(train_b):
        r["id"] = f"train-{i}"

    write_jsonl(os.path.join(dir_b, "train.jsonl"), train_b)
    # Copy identical dev, test, holdout to v2b
    for fname in ["dev.jsonl", "test.jsonl", "test_holdout_haiku.jsonl"]:
        write_jsonl(os.path.join(dir_b, fname), read_jsonl(os.path.join(dir_a, fname)))

    rep_b = dict(rep_a)
    rep_b["train"] = {
        "rows": len(train_b),
        "labels": dict(Counter(r["label"] for r in train_b)),
        "sources": dict(Counter(r["source"] for r in train_b)),
        "origins": dict(Counter(r.get("origin") for r in train_b)),
        "opus_share_of_aranjiya_pct": round(len(chosen_opus) / max(1, sum(1 for r in train_b if r["label"] == 1)) * 100.0, 2),
        "authors": len({r["group"] for r in train_b}),
    }
    with open(os.path.join(dir_b, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(rep_b, f, ensure_ascii=False, indent=2)

    print("=== BUILT v2a & v2b DATASETS ===")
    print("v2a:", json.dumps(rep_a, ensure_ascii=False, indent=2))
    print("v2b train:", json.dumps(rep_b["train"], ensure_ascii=False, indent=2))


def train_tfidf(variant: str):
    data_dir = os.path.join(ROOT, "data", f"processed_{variant}")
    out_dir = os.path.join(ROOT, "models", f"{variant}_baseline")
    os.makedirs(out_dir, exist_ok=True)
    train = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    dev = read_jsonl(os.path.join(data_dir, "dev.jsonl"))

    model = Baseline(C=4.0).fit([r["text"] for r in train], [r["label"] for r in train])
    dev_res = full_report(f"{variant}_tfidf dev", dev, model.predict_proba([r["text"] for r in dev]))
    feats = model.top_word_features(k=30)
    joblib.dump(model, os.path.join(out_dir, "baseline.joblib"))
    with open(os.path.join(out_dir, "dev_results.json"), "w", encoding="utf-8") as f:
        json.dump({"dev": dev_res, "top_features": feats}, f, ensure_ascii=False, indent=2)
    print(f"Saved {variant} TF-IDF -> {out_dir}")


def train_camelbert(variant: str, epochs: int = 3, max_len: int = 256):
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    data_dir = os.path.join(ROOT, "data", f"processed_{variant}")
    out_dir = os.path.join(ROOT, "models", f"{variant}_camelbert")
    os.makedirs(out_dir, exist_ok=True)

    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "fusha", 1: "aranjiya"},
        label2id={"fusha": 0, "aranjiya": 1},
    )

    def prep_v2(t: str) -> str:
        # v2 CAMeLBERT normalization: strip punct & digits, fold alef + ى, keep ة distinct
        return normalize(t, keep_punct=False, fold_letters=False, fold_alef_ya=True)

    def to_ds(rows):
        ds = Dataset.from_dict({"text": [prep_v2(r["text"]) for r in rows], "label": [r["label"] for r in rows]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True)

    train_rows = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    dev_rows = read_jsonl(os.path.join(data_dir, "dev.jsonl"))
    train_ds, dev_ds = to_ds(train_rows), to_ds(dev_rows)

    def metrics(ev):
        logits, labels = ev
        return {"acc": float((logits.argmax(-1) == labels).mean())}

    # Strict dev-only selection guard: eval_dataset=dev_ds ONLY!
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,   # effective bs=32, safe for 8 GB VRAM at max_len=256
        per_device_eval_batch_size=32,
        warmup_steps=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="acc",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        seed=42,
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

    dev_preds = trainer.predict(dev_ds).predictions
    e = np.exp(dev_preds - dev_preds.max(-1, keepdims=True))
    dev_probs = (e / e.sum(-1, keepdims=True))[:, 1]
    dev_res = full_report(f"{variant}_camelbert dev", dev_rows, dev_probs)

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    meta = {
        "variant": variant,
        "base_model": model_id,
        "max_len": max_len,
        "epochs": epochs,
        "fold_letters": False,
        "fold_alef_ya": True,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_acc": trainer.state.best_metric,
        "dev_report": dev_res,
        "epoch_history": [h for h in trainer.state.log_history if "eval_acc" in h],
    }
    with open(os.path.join(out_dir, "v2_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved {variant} CAMeLBERT -> {final_dir} | best_dev_acc={trainer.state.best_metric}")

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--only", choices=["all", "build", "tfidf", "v2a_bert", "v2b_bert"], default="all")
    args = ap.parse_args()

    if args.only in ("all", "build") and not args.skip_build:
        build_datasets()
    if args.only in ("all", "tfidf"):
        train_tfidf("v2a")
        train_tfidf("v2b")
    if args.only in ("all", "v2a_bert"):
        train_camelbert("v2a", epochs=3, max_len=256)
    if args.only in ("all", "v2b_bert"):
        train_camelbert("v2b", epochs=3, max_len=256)


if __name__ == "__main__":
    main()
