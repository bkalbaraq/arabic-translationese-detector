"""Finalize v3 CAMeLBERT before packaging:
  1. Fixed checkpoint rule: select best epoch by LOWEST dev eval_loss for seeds 42, 101, 202.
     - Seed 42 already chose epoch 1 (eval_loss=0.1330) -> reuse models/v3_camelbert/final.
     - Seeds 101 & 202 retrained with metric_for_best_model="eval_loss", greater_is_better=False
       -> models/v3_camelbert_s101_devloss/final & models/v3_camelbert_s202_devloss/final.
  2. Seed ensemble: average p(aranjiya) across the 3 dev-loss-selected seeds.
  3. Primary Haiku metric: non-recalled subset (char3 Jaccard < 0.45, 236 pairs); all 436 secondary.
  4. Freeze results/v3_final_metrics.json with single-model (seed 42, plus 101 & 202),
     3-seed mean & min-max, and 3-seed ensemble across EVERY test set (plus v3_tfidf companion).
  5. Sample 20 random non-recalled Haiku twins predicted فصحى and 10 predicted عرنجية by
     the v3 ensemble -> results/v3_haiku_manual_review_sample.json (all public-domain OpenITI).

COPYRIGHT GUARD: ZERO text from مابعد الأيام or الأيام is ever written to any output file!
"""
import json
import os
import random
import sys

import joblib
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

from build_dataset import bucket  # noqa: E402
from evaluate import load_hard  # noqa: E402
from normalize import normalize  # noqa: E402
from shortcut_check import load_modern_chunks, mask_classical_surface  # noqa: E402
from train_baseline import Baseline  # noqa: F401, E402
from v1_diagnostics import map_nazarat_to_epub_chapters  # noqa: E402
from v2_pipeline_step5_eval_sectionE import (  # noqa: E402
    CONV_TRIGGERS,
    BertModelWrapper,
    eval_calque,
    eval_paired_set,
    read_jsonl,
    summarize_binary,
)
from v3_final_checks import char3_jaccard  # noqa: E402
from v3_step5_eval import load_already_clean_chunks  # noqa: E402


def train_camelbert_devloss(out_dir: str, seed: int, epochs: int = 3, max_len: int = 256):
    final_dir = os.path.join(out_dir, "final")
    meta_path = os.path.join(out_dir, "run_meta.json")
    if os.path.exists(final_dir) and os.path.exists(meta_path):
        print(f"[SKIP] Already trained -> {final_dir}", file=sys.stderr)
        return json.load(open(meta_path, encoding="utf-8"))

    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(ROOT, "data", "processed_v3")
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        if "eval_loss" in h
    ]
    best_row = min(hist, key=lambda x: x["eval_loss"])
    meta = {
        "data_variant": "v3",
        "selection_metric": "eval_loss",
        "epochs": epochs,
        "seed": seed,
        "best_dev_epoch": best_row["epoch"],
        "best_dev_loss": best_row["eval_loss"],
        "best_dev_acc_at_best_loss": best_row["eval_acc"],
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "epoch_history": hist,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"Finished {out_dir}: best_dev_epoch={meta['best_dev_epoch']}, "
        f"best_dev_loss={meta['best_dev_loss']}, acc={meta['best_dev_acc_at_best_loss']}",
        file=sys.stderr,
    )

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def pack_calque_from_scores(cps, sc_ctx, sn_ctx, pc_ctx, pn_ctx, sc_snt, sn_snt, pc_snt, pn_snt):
    def pack(sc, sn, pc, pn):
        def agg(mask=None):
            idx = range(len(cps)) if mask is None else [i for i, m in enumerate(mask) if m]
            diffs = [float(sc[i] - sn[i]) for i in idx]
            ok = sum(1 for d in diffs if d > 0)
            return {
                "n": len(idx),
                "correct": ok,
                "pct": round(ok / max(1, len(idx)) * 100.0, 2),
                "mean_diff": round(float(np.mean(diffs)), 4),
            }

        wrongs = [
            {
                "id": r["id"],
                "book": r["book"],
                "direction": r["direction"],
                "diff": round(float(sc[i] - sn[i]), 4),
                "p_calque": round(float(pc[i]), 4),
                "p_native": round(float(pn[i]), 4),
            }
            for i, r in enumerate(cps)
            if sc[i] <= sn[i]
        ]
        return {
            "overall": agg(),
            "by_direction": {d: agg([r["direction"] == d for r in cps]) for d in sorted({r["direction"] for r in cps})},
            "by_book": {b: agg([r["book"] == b for r in cps]) for b in sorted({r["book"] for r in cps})},
            "wrong_ids": [w["id"] for w in wrongs],
            "wrong_pairs": wrongs,
        }

    return {"contexts": pack(sc_ctx, sn_ctx, pc_ctx, pn_ctx), "sentences": pack(sc_snt, sn_snt, pc_snt, pn_snt)}


def eval_from_cached_preds(preds: dict, ds: dict, use_margins: bool):
    """Compute the full standardized metric bundle from a model's predicted probabilities (and optional margins)."""
    def get_pair_metrics(rows, key):
        p = preds[key]["p"]
        m = preds[key]["m"] if use_margins else None
        return eval_paired_set(rows, p, m)

    # Haiku primary (< 0.45) & secondary (all 436)
    p_ho = preds["haiku"]["p"]
    m_ho = preds["haiku"]["m"] if use_margins else None
    idx_nr = ds["haiku_nr_idx"]
    haiku_nr_rows = [ds["holdout_haiku"][i] for i in idx_nr]
    haiku_nr_res = eval_paired_set(
        haiku_nr_rows, p_ho[idx_nr], m_ho[idx_nr] if m_ho is not None else None
    )
    haiku_nr_twin_p = np.array([float(p_ho[i]) for i in idx_nr if ds["holdout_haiku"][i]["label"] == 1])
    haiku_nr_res["twin_p_quartiles"] = {
        k: round(float(v), 4)
        for k, v in zip(["q1", "median", "q3"], np.percentile(haiku_nr_twin_p, [25, 50, 75]))
    }

    haiku_all_res = eval_paired_set(ds["holdout_haiku"], p_ho, m_ho)
    haiku_all_twin_p = np.array([float(p) for r, p in zip(ds["holdout_haiku"], p_ho) if r["label"] == 1])
    haiku_all_res["twin_p_quartiles"] = {
        k: round(float(v), 4)
        for k, v in zip(["q1", "median", "q3"], np.percentile(haiku_all_twin_p, [25, 50, 75]))
    }

    # Calque pairs (logit margins for single BERT seeds; averaged p for ensemble & TF-IDF)
    def ck(name):
        return (preds[name]["m"] if use_margins else preds[name]["p"]), preds[name]["p"]

    sc_ctx, pc_ctx = ck("calque_cc")
    sn_ctx, pn_ctx = ck("calque_nc")
    sc_snt, pc_snt = ck("calque_cs")
    sn_snt, pn_snt = ck("calque_ns")
    calque_res = pack_calque_from_scores(ds["cps"], sc_ctx, sn_ctx, pc_ctx, pn_ctx, sc_snt, sn_snt, pc_snt, pn_snt)

    # Masking
    pf_o, pf_m = preds["mask_f_orig"]["p"], preds["mask_f_mask"]["p"]
    pt_o, pt_m = preds["mask_t_orig"]["p"], preds["mask_t_mask"]["p"]
    f_flips = int(((pf_o < 0.5) & (pf_m >= 0.5)).sum())
    t_flips = int(((pt_o >= 0.5) & (pt_m < 0.5)).sum())

    # Voice
    p_ev, p_env = preds["essays_voice"]["p"], preds["essays_novoice"]["p"]

    return {
        "clean_test": get_pair_metrics(ds["test_v2"], "test_v2"),
        "haiku_non_recalled_lt_045": haiku_nr_res,
        "haiku_all_436": haiku_all_res,
        "hard_test": get_pair_metrics(ds["hard_rows"], "hard_test"),
        "calque_pairs": calque_res,
        "modern_books": {
            "nazarat_essays": summarize_binary(preds["nazarat_essays"]["p"]),
            "nazarat_adapted": summarize_binary(preds["nazarat_adapted"]["p"]),
            "hayati": summarize_binary(preds["hayati"]["p"]),
            "sabil_altaj": summarize_binary(preds["sabil_altaj"]["p"]),
            "maba3d_alayyam": summarize_binary(preds["maba3d_alayyam"]["p"]),
        },
        "opus_test_per_corpus": {
            corp: summarize_binary(preds[f"opus::{corp}"]["p"]) for corp in ds["opus_corpora"]
        },
        "regression_masking": {
            "classical_pct_fusha_orig": round(float((pf_o < 0.5).mean() * 100.0), 2),
            "classical_pct_fusha_masked": round(float((pf_m < 0.5).mean() * 100.0), 2),
            "classical_flips_f2a": f_flips,
            "classical_flip_rate_pct": round(f_flips / max(1, len(pf_o)) * 100.0, 2),
            "twins_pct_aranjiya_orig": round(float((pt_o >= 0.5).mean() * 100.0), 2),
            "twins_pct_aranjiya_masked": round(float((pt_m >= 0.5).mean() * 100.0), 2),
            "twins_flips_a2f": t_flips,
            "twins_flip_rate_pct": round(t_flips / max(1, len(pt_o)) * 100.0, 2),
        },
        "regression_voice_on_nazarat_essays": {
            "with_voice_words": summarize_binary(p_ev),
            "without_voice_words": summarize_binary(p_env),
            "pct_fusha_gap": round(float((p_env < 0.5).mean() * 100.0 - (p_ev < 0.5).mean() * 100.0), 2),
        },
    }


def summarize_3seeds(seed_evals: dict):
    """Recursively compute {mean, min, max, spread, per_seed} for every numeric leaf across the 3 seeds."""
    keys = ["seed_42", "seed_101", "seed_202"]

    def rec(objs):
        sample = objs[0]
        if isinstance(sample, (int, float)) and not isinstance(sample, bool):
            vals = [float(o) for o in objs]
            return {
                "mean": round(float(np.mean(vals)), 2),
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2),
                "spread": round(float(np.max(vals) - np.min(vals)), 2),
                "per_seed": {k: round(float(v), 4) for k, v in zip(keys, vals)},
            }
        if isinstance(sample, dict):
            out = {}
            for k in sample:
                if k in ("wrong_ids", "wrong_pairs"):
                    continue
                out[k] = rec([o[k] for o in objs])
            return out
        return sample

    return rec([seed_evals[k] for k in keys])


def main():
    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")

    # ---- Step 1: Ensure dev-loss-selected checkpoints exist for seeds 42, 101, 202 ----
    meta_s42_raw = json.load(open(os.path.join(ROOT, "models", "v3_camelbert", "v2_train_meta.json"), encoding="utf-8"))
    hist_s42 = [
        {"epoch": h["epoch"], "eval_acc": round(float(h["eval_acc"]), 4), "eval_loss": round(float(h["eval_loss"]), 4)}
        for h in meta_s42_raw["epoch_history"]
    ]
    best_s42 = min(hist_s42, key=lambda x: x["eval_loss"])
    assert best_s42["epoch"] == 1.0, "Seed 42 lowest dev eval_loss should be epoch 1!"
    meta_s42 = {
        "data_variant": "v3",
        "selection_metric": "eval_loss",
        "epochs": 3,
        "seed": 42,
        "best_dev_epoch": best_s42["epoch"],
        "best_dev_loss": best_s42["eval_loss"],
        "best_dev_acc_at_best_loss": best_s42["eval_acc"],
        "best_model_checkpoint": meta_s42_raw["best_model_checkpoint"],
        "reused_existing_checkpoint": True,
        "epoch_history": hist_s42,
    }

    meta_s101 = train_camelbert_devloss(os.path.join(ROOT, "models", "v3_camelbert_s101_devloss"), seed=101)
    meta_s202 = train_camelbert_devloss(os.path.join(ROOT, "models", "v3_camelbert_s202_devloss"), seed=202)

    # ---- Step 2: Load all evaluation datasets ----
    test_v2 = read_jsonl(os.path.join(ROOT, "data", "processed_v2a", "test.jsonl"))
    holdout_haiku = read_jsonl(os.path.join(ROOT, "data", "processed_v2a", "test_holdout_haiku.jsonl"))
    haiku_pairs = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v2_haiku_holdout.jsonl"))
    hard_rows = load_hard(os.path.join(ROOT, "data", "hard_test.jsonl"))
    cps = read_jsonl(os.path.join(ROOT, "data", "calque_pairs.jsonl"))

    haiku_jacc_by_pid = {r["pair_id"]: char3_jaccard(r["fusha"], r["aranjiya"]) for r in haiku_pairs}
    non_recalled_pids = {pid for pid, j in haiku_jacc_by_pid.items() if j < 0.45}
    haiku_nr_idx = [i for i, r in enumerate(holdout_haiku) if r["pair_id"] in non_recalled_pids]

    human_all = read_jsonl(os.path.join(ROOT, "data", "raw", "human_translated.jsonl"))
    rng_opus = random.Random(42)
    opus_corpora = ["opus:OpenSubtitles", "opus:TED2020", "opus:News-Commentary", "opus:GlobalVoices"]
    opus_test = {}
    for corpus in opus_corpora:
        pool = [h["text"] for h in human_all if h["source"] == corpus and bucket(h["id"], 0.1, 0.1) == "test"]
        rng_opus.shuffle(pool)
        opus_test[corpus] = pool[:1000]

    adapted_chunks, essay_chunks, _, _, _ = map_nazarat_to_epub_chapters(
        os.path.join(dl, "manfaluti_nazarat_clean.txt"),
        os.path.join(dl, "94061468.epub"),
    )
    hayati_chunks = load_modern_chunks(os.path.join(dl, "ahmad_amin_hayati_clean.txt"))
    taj_chunks = load_modern_chunks(os.path.join(dl, "manfaluti_sabil_altaj_clean.txt"))
    # Copyrighted test-only file: NEVER write any text excerpts to output files!
    zayyat_chunks = load_already_clean_chunks(os.path.join(dl, "mhzayyat_maba3d_alayyam_TESTONLY.txt"))

    essays_with_voice, essays_no_voice = [], []
    for ch in essay_chunks:
        ws = set(normalize(ch, keep_punct=False, fold_letters=True).split())
        (essays_with_voice if (ws & CONV_TRIGGERS) else essays_no_voice).append(ch)

    test_fusha_orig = [r["text"] for r in test_v2 if r["label"] == 0]
    test_fusha_mask = [mask_classical_surface(t) for t in test_fusha_orig]
    test_twin_orig = [r["text"] for r in test_v2 if r["label"] == 1]
    test_twin_mask = [mask_classical_surface(t) for t in test_twin_orig]

    text_buckets = {
        "test_v2": [r["text"] for r in test_v2],
        "haiku": [r["text"] for r in holdout_haiku],
        "hard_test": [r["text"] for r in hard_rows],
        "calque_cc": [r["calque_context"] for r in cps],
        "calque_nc": [r["native_context"] for r in cps],
        "calque_cs": [r["calque_sentence"] for r in cps],
        "calque_ns": [r["native_sentence"] for r in cps],
        "nazarat_essays": essay_chunks,
        "nazarat_adapted": adapted_chunks,
        "hayati": hayati_chunks,
        "sabil_altaj": taj_chunks,
        "maba3d_alayyam": zayyat_chunks,
        "essays_voice": essays_with_voice,
        "essays_novoice": essays_no_voice,
        "mask_f_orig": test_fusha_orig,
        "mask_f_mask": test_fusha_mask,
        "mask_t_orig": test_twin_orig,
        "mask_t_mask": test_twin_mask,
    }
    for corp in opus_corpora:
        text_buckets[f"opus::{corp}"] = opus_test[corp]

    ds = {
        "test_v2": test_v2,
        "holdout_haiku": holdout_haiku,
        "haiku_nr_idx": haiku_nr_idx,
        "hard_rows": hard_rows,
        "cps": cps,
        "opus_corpora": opus_corpora,
    }

    # ---- Step 3: Score all text buckets with each of the 3 dev-loss BERT seeds ----
    seed_dirs = {
        "seed_42": os.path.join(ROOT, "models", "v3_camelbert", "final"),
        "seed_101": os.path.join(ROOT, "models", "v3_camelbert_s101_devloss", "final"),
        "seed_202": os.path.join(ROOT, "models", "v3_camelbert_s202_devloss", "final"),
    }

    seed_preds = {}
    seed_evals = {}
    for skey, sdir in seed_dirs.items():
        print(f"Scoring all sets with {skey} ({sdir})...", file=sys.stderr)
        bw = BertModelWrapper(sdir, max_len=256, fold_alef_ya=True)
        preds = {}
        for bname, txts in text_buckets.items():
            p, m = bw.score_all(txts)
            preds[bname] = {"p": p, "m": m}
        bw.close()
        seed_preds[skey] = preds
        seed_evals[skey] = eval_from_cached_preds(preds, ds, use_margins=True)

    # ---- Step 4: Build 3-seed ensemble by averaging p(aranjiya) ----
    ens_preds = {}
    for bname in text_buckets:
        p_avg = (
            seed_preds["seed_42"][bname]["p"]
            + seed_preds["seed_101"][bname]["p"]
            + seed_preds["seed_202"][bname]["p"]
        ) / 3.0
        m_avg = (
            seed_preds["seed_42"][bname]["m"]
            + seed_preds["seed_101"][bname]["m"]
            + seed_preds["seed_202"][bname]["m"]
        ) / 3.0
        ens_preds[bname] = {"p": p_avg, "m": m_avg}

    # Evaluate ensemble using averaged p(aranjiya) directly (as requested), and verify if mean_logit differs anywhere
    ens_eval = eval_from_cached_preds(ens_preds, ds, use_margins=False)
    ens_eval_logit_check = eval_from_cached_preds(ens_preds, ds, use_margins=True)
    logit_check_note = {
        "calque_contexts_p_avg": ens_eval["calque_pairs"]["contexts"]["overall"]["correct"],
        "calque_contexts_logit_avg": ens_eval_logit_check["calque_pairs"]["contexts"]["overall"]["correct"],
        "calque_sentences_p_avg": ens_eval["calque_pairs"]["sentences"]["overall"]["correct"],
        "calque_sentences_logit_avg": ens_eval_logit_check["calque_pairs"]["sentences"]["overall"]["correct"],
        "clean_test_pw_p_avg": ens_eval["clean_test"]["pairwise_acc"],
        "clean_test_pw_logit_avg": ens_eval_logit_check["clean_test"]["pairwise_acc"],
        "haiku_nr_pw_p_avg": ens_eval["haiku_non_recalled_lt_045"]["pairwise_acc"],
        "haiku_nr_pw_logit_avg": ens_eval_logit_check["haiku_non_recalled_lt_045"]["pairwise_acc"],
        "hard_test_pw_p_avg": ens_eval["hard_test"]["pairwise_acc"],
        "hard_test_pw_logit_avg": ens_eval_logit_check["hard_test"]["pairwise_acc"],
    }

    # Also evaluate v3_tfidf companion with identical standardized layout
    tfidf_mdl = joblib.load(os.path.join(ROOT, "models", "v3_baseline", "baseline.joblib"))
    tfidf_preds = {bname: {"p": tfidf_mdl.predict_proba(txts), "m": None} for bname, txts in text_buckets.items()}
    tfidf_eval = eval_from_cached_preds(tfidf_preds, ds, use_margins=False)

    # ---- Step 5: Freeze results/v3_final_metrics.json ----
    frozen = {
        "schema_note": (
            "Final frozen v3 evaluation metrics. Checkpoint selection rule: lowest dev eval_loss "
            "(selected after observing epoch 1 prevented overfitting on modern books). "
            "Primary Haiku benchmark: non-recalled subset (char3 Jaccard < 0.45, 236 pairs); "
            "all 436 pairs reported as secondary. Zero copyrighted text stored."
        ),
        "checkpoint_rule": "lowest_dev_eval_loss",
        "dataset_sizes": {
            "clean_test_pairs": len(test_v2) // 2,
            "haiku_non_recalled_pairs": len(haiku_nr_idx) // 2,
            "haiku_all_pairs": len(holdout_haiku) // 2,
            "hard_test_items": len(hard_rows),
            "hard_test_pairs": 36,
            "calque_pairs": len(cps),
            "nazarat_essays": len(essay_chunks),
            "nazarat_adapted": len(adapted_chunks),
            "hayati": len(hayati_chunks),
            "sabil_altaj": len(taj_chunks),
            "maba3d_alayyam": len(zayyat_chunks),
            "opus_per_corpus": 1000,
        },
        "train_metas": {
            "seed_42": meta_s42,
            "seed_101": meta_s101,
            "seed_202": meta_s202,
        },
        "single_model_seed_42": seed_evals["seed_42"],
        "all_seeds": seed_evals,
        "three_seed_summary": summarize_3seeds(seed_evals),
        "ensemble_3seed_avg_p": ens_eval,
        "ensemble_p_vs_logit_pairwise_check": logit_check_note,
        "companion_v3_tfidf": tfidf_eval,
    }

    frozen_path = os.path.join(ROOT, "results", "v3_final_metrics.json")
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    print(f"\nSaved frozen metrics -> {frozen_path}", file=sys.stderr)

    # ---- Step 6: Sample 20 non-recalled Haiku twins predicted فصحى and 10 predicted عرنجية by ensemble ----
    p_ens_ho = ens_preds["haiku"]["p"]
    p_by_pid = {}
    for r, p in zip(holdout_haiku, p_ens_ho):
        p_by_pid.setdefault(r["pair_id"], {})[r["label"]] = float(p)

    nr_pred_fusha = []
    nr_pred_aranjiya = []
    for r in haiku_pairs:
        pid = r["pair_id"]
        j = haiku_jacc_by_pid[pid]
        if j >= 0.45:
            continue
        p_src = p_by_pid[pid][0]
        p_hku = p_by_pid[pid][1]
        item = {
            "pair_id": pid,
            "char3_jaccard": round(float(j), 4),
            "p_ens_source": round(p_src, 4),
            "p_ens_haiku": round(p_hku, 4),
            "fusha_source": r["fusha"],
            "haiku_output": r["aranjiya"],
        }
        if p_hku < 0.5:
            nr_pred_fusha.append(item)
        else:
            nr_pred_aranjiya.append(item)

    rng_samp = random.Random(2026)
    samp_fusha = rng_samp.sample(nr_pred_fusha, min(20, len(nr_pred_fusha)))
    samp_aranjiya = rng_samp.sample(nr_pred_aranjiya, min(10, len(nr_pred_aranjiya)))

    sample_out = {
        "subset": "haiku_non_recalled_lt_045",
        "total_non_recalled": len(nr_pred_fusha) + len(nr_pred_aranjiya),
        "total_pred_fusha": len(nr_pred_fusha),
        "total_pred_aranjiya": len(nr_pred_aranjiya),
        "sample_seed": 2026,
        "predicted_fusha_20": samp_fusha,
        "predicted_aranjiya_10": samp_aranjiya,
    }
    samp_path = os.path.join(ROOT, "results", "v3_haiku_manual_review_sample.json")
    with open(samp_path, "w", encoding="utf-8") as f:
        json.dump(sample_out, f, ensure_ascii=False, indent=2)
    print(f"Saved manual review sample -> {samp_path}", file=sys.stderr)

    # ---- Print compact summary to stdout ----
    print("\n=== CHECKPOINT SELECTION (LOWEST DEV EVAL_LOSS) ===")
    for k, m in frozen["train_metas"].items():
        print(f"  {k:8s}: best_ep={m['best_dev_epoch']} | best_dev_loss={m['best_dev_loss']} | acc@best={m['best_dev_acc_at_best_loss']} | hist={m['epoch_history']}")

    print("\n=== ENSEMBLE p_avg vs logit_avg PAIRWISE CHECK ===")
    print(json.dumps(logit_check_note, indent=2))

    def row_line(name, e):
        mb = e["modern_books"]
        cp = e["calque_pairs"]
        hn = e["haiku_non_recalled_lt_045"]
        ha = e["haiku_all_436"]
        return (
            f"{name:12s} | test={e['clean_test']['acc']:5.2f}%(pw {e['clean_test']['pairwise_acc']:5.2f}%) "
            f"| haiku<0.45: pw={hn['pairwise_acc']:5.2f}% rec={hn['twin_pct_aranjiya']:5.2f}% "
            f"| haikuALL: pw={ha['pairwise_acc']:5.2f}% rec={ha['twin_pct_aranjiya']:5.2f}% "
            f"| hard={e['hard_test']['acc']:5.2f}%(pw {e['hard_test']['pairwise_acc']:5.2f}%) "
            f"| calque={cp['contexts']['overall']['correct']}/38({cp['sentences']['overall']['correct']}/38) "
            f"| naz_ess={mb['nazarat_essays']['pct_fusha']:5.2f}% "
            f"| naz_adp={mb['nazarat_adapted']['pct_fusha']:5.2f}% "
            f"| hayati={mb['hayati']['pct_fusha']:5.2f}% "
            f"| taj={mb['sabil_altaj']['pct_fusha']:5.2f}% "
            f"| maba3d={mb['maba3d_alayyam']['pct_fusha']:5.2f}%"
        )

    print("\n=== MODEL COMPARISON ===")
    for k in ["seed_42", "seed_101", "seed_202"]:
        print(row_line(k, seed_evals[k]))
    print(row_line("ENSEMBLE", ens_eval))
    print(row_line("v3_tfidf", tfidf_eval))


if __name__ == "__main__":
    main()
