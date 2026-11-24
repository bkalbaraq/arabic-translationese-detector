"""v3 Final Pre-Packaging Checks (all 4 requested checks, zero new data, zero API calls):
  1. Epoch confound: retrain v2a CAMeLBERT for 1 epoch (seed=42, exact v2a settings) and evaluate.
  2. Hard test diff: every item that changed between v2a_bert and v3_bert in BOTH directions
     (fixed vs newly wrong).
  3. Haiku "recalled original" check: character 3-gram Jaccard overlap between twin & classical source
     for Haiku held-out vs OpusMT & NLLB-1.3B training twins; top 5 Haiku examples; Haiku metrics
     on ALL twins vs NON-recalled twins for v1, v2a, v2a_1ep, v3.
  4. Seed stability: train v3 CAMeLBERT with seed=101 and seed=202 (same settings, dev-only selection),
     evaluate alongside seed=42, and report mean & min-max across all 3 runs.

COPYRIGHT GUARD: Zero text from مابعد الأيام or الأيام is ever written to results JSON!
Outputs -> results/v3_final_checks.json
"""
import json
import os
import sys

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from evaluate import load_hard  # noqa: E402
from normalize import normalize  # noqa: E402
from shortcut_check import load_modern_chunks  # noqa: E402
from v1_diagnostics import map_nazarat_to_epub_chapters  # noqa: E402
from v2_pipeline_step5_eval_sectionE import (  # noqa: E402
    BertModelWrapper,
    eval_calque,
    eval_paired_set,
    read_jsonl,
    summarize_binary,
)
from v3_step5_eval import load_already_clean_chunks  # noqa: E402


# --------------------------------------------------------------------------- Training helper
def train_camelbert_run(data_variant: str, out_dir: str, epochs: int, seed: int, max_len: int = 256):
    final_dir = os.path.join(out_dir, "final")
    meta_path = os.path.join(out_dir, "run_meta.json")
    if os.path.exists(final_dir) and os.path.exists(meta_path):
        print(f"[SKIP] Already trained -> {final_dir}", file=sys.stderr)
        return json.load(open(meta_path, encoding="utf-8"))

    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(ROOT, "data", f"processed_{data_variant}")
    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "fusha", 1: "aranjiya"},
        label2id={"fusha": 0, "aranjiya": 1},
    )

    def prep(t: str) -> str:
        return normalize(t, keep_punct=False, fold_letters=False, fold_alef_ya=True)

    def to_ds(rows):
        ds = Dataset.from_dict({"text": [prep(r["text"]) for r in rows], "label": [r["label"] for r in rows]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True)

    train_rows = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    dev_rows = read_jsonl(os.path.join(data_dir, "dev.jsonl"))
    train_ds, dev_ds = to_ds(train_rows), to_ds(dev_rows)

    def metrics(ev):
        logits, labels = ev
        return {"acc": float((logits.argmax(-1) == labels).mean())}

    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        warmup_steps=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="acc",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        seed=seed,
        data_seed=seed,
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
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    hist = [
        {"epoch": h["epoch"], "eval_acc": round(float(h["eval_acc"]), 4), "eval_loss": round(float(h["eval_loss"]), 4)}
        for h in trainer.state.log_history
        if "eval_acc" in h
    ]
    best_ep = max(hist, key=lambda x: x["eval_acc"])["epoch"] if hist else epochs
    meta = {
        "data_variant": data_variant,
        "epochs": epochs,
        "seed": seed,
        "best_dev_epoch": best_ep,
        "best_dev_acc": round(float(trainer.state.best_metric), 4),
        "epoch_history": hist,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Finished {out_dir}: best_dev_epoch={best_ep}, best_dev_acc={meta['best_dev_acc']}", file=sys.stderr)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# --------------------------------------------------------------------------- Char 3-gram overlap
def char_ngrams(text: str, n: int = 3):
    s = normalize(text, keep_punct=False, fold_letters=True)
    if len(s) < n:
        return set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char3_jaccard(src: str, twin: str) -> float:
    a = char_ngrams(src, 3)
    b = char_ngrams(twin, 3)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pct_summary(arr):
    a = np.asarray(arr, dtype=float)
    return {
        "n": int(len(a)),
        "mean": round(float(a.mean()), 4),
        "min": round(float(a.min()), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.percentile(a, 50)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "p99": round(float(np.percentile(a, 99)), 4),
        "max": round(float(a.max()), 4),
        "pct_ge_040": round(float((a >= 0.40).mean() * 100.0), 2),
        "pct_ge_045": round(float((a >= 0.45).mean() * 100.0), 2),
        "pct_ge_050": round(float((a >= 0.50).mean() * 100.0), 2),
        "pct_ge_055": round(float((a >= 0.55).mean() * 100.0), 2),
        "pct_ge_060": round(float((a >= 0.60).mean() * 100.0), 2),
    }


def eval_bert_suite(model_dir: str, fold_alef_ya: bool, datasets: dict):
    bw = BertModelWrapper(model_dir, max_len=256, fold_alef_ya=fold_alef_ya)
    p_tv2, m_tv2 = bw.score_all([r["text"] for r in datasets["test_v2"]])
    p_ho, m_ho = bw.score_all([r["text"] for r in datasets["holdout_haiku"]])
    p_hd, m_hd = bw.score_all([r["text"] for r in datasets["hard_rows"]])
    p_ess, _ = bw.score_all(datasets["essays"])
    p_adp, _ = bw.score_all(datasets["adapted"])
    p_hay, _ = bw.score_all(datasets["hayati"])
    p_taj, _ = bw.score_all(datasets["taj"])
    p_zay, _ = bw.score_all(datasets["zayyat"])
    calque = eval_calque(datasets["cps"], bw.score_all, is_bert=True)
    bw.close()

    return {
        "test_v2": eval_paired_set(datasets["test_v2"], p_tv2, m_tv2),
        "haiku_all": eval_paired_set(datasets["holdout_haiku"], p_ho, m_ho),
        "hard_test": eval_paired_set(datasets["hard_rows"], p_hd, m_hd),
        "modern_books": {
            "nazarat_essays": summarize_binary(p_ess),
            "nazarat_adapted": summarize_binary(p_adp),
            "hayati": summarize_binary(p_hay),
            "sabil_altaj": summarize_binary(p_taj),
            "maba3d_alayyam": summarize_binary(p_zay),
        },
        "calque_pairs": {
            "contexts_correct": calque["contexts"]["overall"]["correct"],
            "contexts_pct": calque["contexts"]["overall"]["pct"],
            "contexts_mean_logit_diff": calque["contexts"]["overall"]["mean_diff"],
            "contexts_by_dir": calque["contexts"]["by_direction"],
            "contexts_wrong": [w["id"] for w in calque["contexts"]["wrong_pairs"]],
            "sentences_correct": calque["sentences"]["overall"]["correct"],
            "sentences_pct": calque["sentences"]["overall"]["pct"],
            "sentences_mean_logit_diff": calque["sentences"]["overall"]["mean_diff"],
            "sentences_wrong": [w["id"] for w in calque["sentences"]["wrong_pairs"]],
        },
        "_haiku_probs": p_ho,
        "_haiku_margins": m_ho,
        "_hard_probs": p_hd,
    }


def main():
    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")

    # ---- Step 1: Train v2a 1-epoch (seed=42) and v3 seeds 101 & 202 ----
    meta_v2a_1ep = train_camelbert_run(
        "v2a", os.path.join(ROOT, "models", "v2a_camelbert_1ep"), epochs=1, seed=42
    )
    meta_v3_s101 = train_camelbert_run(
        "v3", os.path.join(ROOT, "models", "v3_camelbert_s101"), epochs=3, seed=101
    )
    meta_v3_s202 = train_camelbert_run(
        "v3", os.path.join(ROOT, "models", "v3_camelbert_s202"), epochs=3, seed=202
    )
    meta_v3_s42 = json.load(open(os.path.join(ROOT, "models", "v3_camelbert", "v2_train_meta.json"), encoding="utf-8"))

    # ---- Load shared evaluation datasets ----
    test_v2 = read_jsonl(os.path.join(ROOT, "data", "processed_v2a", "test.jsonl"))
    holdout_haiku = read_jsonl(os.path.join(ROOT, "data", "processed_v2a", "test_holdout_haiku.jsonl"))
    haiku_pairs = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v2_haiku_holdout.jsonl"))
    hard_rows = load_hard(os.path.join(ROOT, "data", "hard_test.jsonl"))
    cps = read_jsonl(os.path.join(ROOT, "data", "calque_pairs.jsonl"))
    adapted, essays, _, _, _ = map_nazarat_to_epub_chapters(
        os.path.join(dl, "manfaluti_nazarat_clean.txt"),
        os.path.join(dl, "94061468.epub"),
    )
    hayati = load_modern_chunks(os.path.join(dl, "ahmad_amin_hayati_clean.txt"))
    taj = load_modern_chunks(os.path.join(dl, "manfaluti_sabil_altaj_clean.txt"))
    zayyat = load_already_clean_chunks(os.path.join(dl, "mhzayyat_maba3d_alayyam_TESTONLY.txt"))

    datasets = {
        "test_v2": test_v2,
        "holdout_haiku": holdout_haiku,
        "hard_rows": hard_rows,
        "cps": cps,
        "essays": essays,
        "adapted": adapted,
        "hayati": hayati,
        "taj": taj,
        "zayyat": zayyat,
    }

    # ---- Check 3: Char 3-gram Jaccard overlap (Haiku held-out vs OpusMT & NLLB-1.3B train twins) ----
    haiku_jacc = []
    haiku_jacc_by_pid = {}
    for r in haiku_pairs:
        j = char3_jaccard(r["fusha"], r["aranjiya"])
        haiku_jacc.append((j, r))
        haiku_jacc_by_pid[r["pair_id"]] = j

    # Sanity check: OpusMT & NLLB-1.3B classical training twins (exclude copyrighted Taha Hussein!)
    train_pairs_classic = (
        read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v2_existing_clean.jsonl"))
        + read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v2_stepD_clean.jsonl"))
    )
    opus_jacc = [char3_jaccard(r["fusha"], r["aranjiya"]) for r in train_pairs_classic if r["backward"] == "opus_mt"]
    nllb_jacc = [char3_jaccard(r["fusha"], r["aranjiya"]) for r in train_pairs_classic if r["backward"] == "nllb_13b"]

    haiku_jacc_sorted = sorted(haiku_jacc, key=lambda x: x[0], reverse=True)
    top5_haiku_overlap = [
        {
            "pair_id": r["pair_id"],
            "book": r.get("book"),
            "char3_jaccard": round(float(j), 4),
            "fusha": r["fusha"],
            "pivot": r["pivot"],
            "haiku_aranjiya": r["aranjiya"],
        }
        for j, r in haiku_jacc_sorted[:5]
    ]

    # Choose recalled threshold based on OpusMT/NLLB p95 (~0.40-0.45): evaluate at 0.45 AND 0.50!
    RECALL_THR = 0.45
    non_recalled_pids_45 = {pid for pid, j in haiku_jacc_by_pid.items() if j < 0.45}
    non_recalled_pids_50 = {pid for pid, j in haiku_jacc_by_pid.items() if j < 0.50}

    # ---- Evaluate all relevant BERT models ----
    models_to_eval = [
        ("v1_bert", os.path.join(ROOT, "models", "camelbert", "final"), False),
        ("v2a_bert_3ep", os.path.join(ROOT, "models", "v2a_camelbert", "final"), True),
        ("v2a_bert_1ep", os.path.join(ROOT, "models", "v2a_camelbert_1ep", "final"), True),
        ("v3_bert_s42", os.path.join(ROOT, "models", "v3_camelbert", "final"), True),
        ("v3_bert_s101", os.path.join(ROOT, "models", "v3_camelbert_s101", "final"), True),
        ("v3_bert_s202", os.path.join(ROOT, "models", "v3_camelbert_s202", "final"), True),
    ]

    evals = {}
    haiku_filtered = {}
    hard_probs_map = {}

    for mkey, mdir, fay in models_to_eval:
        print(f"Evaluating {mkey}...", file=sys.stderr)
        res = eval_bert_suite(mdir, fay, datasets)
        p_ho = res.pop("_haiku_probs")
        m_ho = res.pop("_haiku_margins")
        p_hd = res.pop("_hard_probs")
        hard_probs_map[mkey] = p_hd
        evals[mkey] = res

        def subset_haiku(allowed_pids):
            idx = [i for i, r in enumerate(holdout_haiku) if r["pair_id"] in allowed_pids]
            sub_rows = [holdout_haiku[i] for i in idx]
            return eval_paired_set(sub_rows, p_ho[idx], m_ho[idx])

        haiku_filtered[mkey] = {
            "all_436": res["haiku_all"],
            "non_recalled_lt_045": subset_haiku(non_recalled_pids_45),
            "non_recalled_lt_050": subset_haiku(non_recalled_pids_50),
        }

    # ---- Check 2: Hard test diff between v2a_bert and v3_bert (BOTH directions) ----
    p_v2a_hd = hard_probs_map["v2a_bert_3ep"]
    p_v3_hd = hard_probs_map["v3_bert_s42"]
    fixed_by_v3 = []
    newly_wrong_in_v3 = []
    for i, r in enumerate(hard_rows):
        pv2, pv3 = float(p_v2a_hd[i]), float(p_v3_hd[i])
        ok2 = (pv2 >= 0.5) == (r["label"] == 1)
        ok3 = (pv3 >= 0.5) == (r["label"] == 1)
        if (not ok2) and ok3:
            fixed_by_v3.append({
                "id": r["id"], "label": r["label"], "v2a_p": round(pv2, 4), "v3_p": round(pv3, 4), "text": r["text"]
            })
        elif ok2 and (not ok3):
            newly_wrong_in_v3.append({
                "id": r["id"], "label": r["label"], "v2a_p": round(pv2, 4), "v3_p": round(pv3, 4), "text": r["text"]
            })

    # ---- Check 4: Seed stability aggregation across seeds 42, 101, 202 ----
    seed_keys = ["v3_bert_s42", "v3_bert_s101", "v3_bert_s202"]
    def agg_seeds(getter):
        vals = [float(getter(evals[k])) for k in seed_keys]
        return {
            "per_seed": {k: round(v, 2) for k, v in zip(seed_keys, vals)},
            "mean": round(float(np.mean(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
            "spread": round(float(np.max(vals) - np.min(vals)), 2),
        }

    seed_stability = {
        "train_metas": {
            "s42": {
                "best_dev_epoch": max(meta_v3_s42["epoch_history"], key=lambda x: x["eval_acc"])["epoch"],
                "best_dev_acc": round(float(meta_v3_s42["best_dev_acc"]), 4),
            },
            "s101": {"best_dev_epoch": meta_v3_s101["best_dev_epoch"], "best_dev_acc": meta_v3_s101["best_dev_acc"]},
            "s202": {"best_dev_epoch": meta_v3_s202["best_dev_epoch"], "best_dev_acc": meta_v3_s202["best_dev_acc"]},
        },
        "metrics": {
            "hayati_pct_fusha": agg_seeds(lambda e: e["modern_books"]["hayati"]["pct_fusha"]),
            "nazarat_essays_pct_fusha": agg_seeds(lambda e: e["modern_books"]["nazarat_essays"]["pct_fusha"]),
            "nazarat_adapted_pct_fusha": agg_seeds(lambda e: e["modern_books"]["nazarat_adapted"]["pct_fusha"]),
            "sabil_altaj_pct_fusha": agg_seeds(lambda e: e["modern_books"]["sabil_altaj"]["pct_fusha"]),
            "maba3d_alayyam_pct_fusha": agg_seeds(lambda e: e["modern_books"]["maba3d_alayyam"]["pct_fusha"]),
            "calque_contexts_correct": agg_seeds(lambda e: e["calque_pairs"]["contexts_correct"]),
            "calque_sentences_correct": agg_seeds(lambda e: e["calque_pairs"]["sentences_correct"]),
            "haiku_pairwise_acc": agg_seeds(lambda e: e["haiku_all"]["pairwise_acc"]),
            "clean_test_acc": agg_seeds(lambda e: e["test_v2"]["acc"]),
            "hard_test_acc": agg_seeds(lambda e: e["hard_test"]["acc"]),
            "hard_test_pairwise_acc": agg_seeds(lambda e: e["hard_test"]["pairwise_acc"]),
        },
    }

    out = {
        "check1_epoch_confound": {
            "v2a_1ep_meta": meta_v2a_1ep,
            "v2a_bert_3ep": evals["v2a_bert_3ep"],
            "v2a_bert_1ep": evals["v2a_bert_1ep"],
            "v3_bert_s42": evals["v3_bert_s42"],
        },
        "check2_hard_test_diff_v2a_vs_v3": {
            "v2a_acc": evals["v2a_bert_3ep"]["hard_test"]["acc"],
            "v3_acc": evals["v3_bert_s42"]["hard_test"]["acc"],
            "fixed_by_v3_count": len(fixed_by_v3),
            "newly_wrong_in_v3_count": len(newly_wrong_in_v3),
            "fixed_by_v3": fixed_by_v3,
            "newly_wrong_in_v3": newly_wrong_in_v3,
        },
        "check3_haiku_recalled_original": {
            "overlap_distributions_char3_jaccard": {
                "haiku_heldout_436": pct_summary([j for j, _ in haiku_jacc]),
                "opus_mt_train_clean": pct_summary(opus_jacc),
                "nllb_13b_train_clean": pct_summary(nllb_jacc),
            },
            "top5_haiku_highest_overlap": top5_haiku_overlap,
            "flagged_counts": {
                "ge_045": len(haiku_jacc_by_pid) - len(non_recalled_pids_45),
                "lt_045": len(non_recalled_pids_45),
                "ge_050": len(haiku_jacc_by_pid) - len(non_recalled_pids_50),
                "lt_050": len(non_recalled_pids_50),
            },
            "haiku_eval_all_vs_non_recalled": haiku_filtered,
        },
        "check4_seed_stability": seed_stability,
    }

    out_path = os.path.join(ROOT, "results", "v3_final_checks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nSaved ->", out_path)
    print("\n=== CHECK 1: EPOCH CONFOUND ===")
    for k in ["v2a_bert_3ep", "v2a_bert_1ep", "v3_bert_s42"]:
        e = evals[k]
        mb = e["modern_books"]
        print(
            f"{k:14s} | test={e['test_v2']['acc']:5.1f}% | haiku_pw={e['haiku_all']['pairwise_acc']:5.1f}% "
            f"| calque={e['calque_pairs']['contexts_correct']}/38 ({e['calque_pairs']['sentences_correct']}/38 snt) "
            f"| naz_ess={mb['nazarat_essays']['pct_fusha']:5.1f}% | naz_adp={mb['nazarat_adapted']['pct_fusha']:5.1f}% "
            f"| hayati={mb['hayati']['pct_fusha']:5.1f}% | taj={mb['sabil_altaj']['pct_fusha']:5.1f}% "
            f"| maba3d={mb['maba3d_alayyam']['pct_fusha']:5.1f}%"
        )

    print("\n=== CHECK 2: HARD TEST DIFFS (v2a vs v3) ===")
    print(f"Fixed by v3: {len(fixed_by_v3)} | Newly wrong in v3: {len(newly_wrong_in_v3)}")

    print("\n=== CHECK 3: CHAR 3-GRAM JACCARD OVERLAP ===")
    for k, v in out["check3_haiku_recalled_original"]["overlap_distributions_char3_jaccard"].items():
        print(f"  {k:22s}: median={v['median']} p75={v['p75']} p90={v['p90']} p95={v['p95']} max={v['max']} | >=0.45: {v['pct_ge_045']}% | >=0.50: {v['pct_ge_050']}%")
    for mk in ["v1_bert", "v2a_bert_3ep", "v2a_bert_1ep", "v3_bert_s42"]:
        a = haiku_filtered[mk]["all_436"]
        n45 = haiku_filtered[mk]["non_recalled_lt_045"]
        n50 = haiku_filtered[mk]["non_recalled_lt_050"]
        print(f"  {mk:14s} | ALL: acc={a['acc']} rec={a['twin_pct_aranjiya']} pw={a['pairwise_acc']} | <0.45({n45['n_pairs']}): acc={n45['acc']} rec={n45['twin_pct_aranjiya']} pw={n45['pairwise_acc']} | <0.50({n50['n_pairs']}): acc={n50['acc']} rec={n50['twin_pct_aranjiya']} pw={n50['pairwise_acc']}")

    print("\n=== CHECK 4: SEED STABILITY (3 runs: 42, 101, 202) ===")
    print("Metas:", seed_stability["train_metas"])
    for mname, mv in seed_stability["metrics"].items():
        print(f"  {mname:28s}: mean={mv['mean']} [min={mv['min']}, max={mv['max']}] | {mv['per_seed']}")


if __name__ == "__main__":
    main()
