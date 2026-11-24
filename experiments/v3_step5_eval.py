"""v3 Step 5 — Complete Side-by-Side Evaluation: v1 vs v2a vs v3 (TF-IDF + CAMeLBERT).

Evaluates all 6 models (v1_tfidf, v2a_tfidf, v3_tfidf, v1_bert, v2a_bert, v3_bert) on the EXACT SAME test sets:
  1. Clean unseen-author test pairs (data/processed_v2a/test.jsonl, 461 pairs)
  2. Haiku held-out translator (data/processed_v2a/test_holdout_haiku.jsonl, 436 pairs)
  3. OPUS test per corpus (OpenSubtitles, TED2020, News-Commentary, GlobalVoices; 1,000 test-split rows each)
  4. Hard test (data/hard_test.jsonl, all 83 rows / 36 pairs) + list of items changed between v1, v2a, v3
  5. Modern books table (ALL TEST-ONLY):
       - النظرات essays (3,075 chunks)
       - النظرات adapted chapters (376 chunks)
       - حياتي (1,550 chunks)
       - في سبيل التاج (513 chunks)
       - ما بعد الأيام (mhzayyat_maba3d_alayyam_TESTONLY.txt, 1,639 chunks — copyrighted, aggregate stats ONLY!)
  6. calque_pairs.jsonl (contexts primary + sentences, by direction & book; CAMeLBERT by LOGITS; every wrong pair)
  7. Regression checks: masking test + voice gap on النظرات essays + voice rates/10k in v3 train
  8. Top 25 TF-IDF features per side for v1, v2a, v3

Saves aggregate metrics -> results/v3_eval.json (contains ZERO text from ما بعد الأيام).
"""
import json
import os
import random
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_dataset import bucket  # noqa: E402
from evaluate import load_hard  # noqa: E402
from normalize import normalize  # noqa: E402
from openiti_extract import chunk_paragraph  # noqa: E402
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
    voice_rates_per_10k,
)


def load_already_clean_chunks(path: str):
    """Chunk an already-cleaned one-paragraph-per-line file (15-60 words) without re-cleaning."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            for ch in chunk_paragraph(line, min_words=15, max_words=60):
                if len(ch.split()) >= 15:
                    chunks.append(ch)
    return chunks


def main():
    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")

    test_v2 = read_jsonl(os.path.join(ROOT, "data", "processed_v2a", "test.jsonl"))
    holdout_haiku = read_jsonl(os.path.join(ROOT, "data", "processed_v2a", "test_holdout_haiku.jsonl"))
    hard_rows = load_hard(os.path.join(ROOT, "data", "hard_test.jsonl"))
    cps = read_jsonl(os.path.join(ROOT, "data", "calque_pairs.jsonl"))

    human_all = read_jsonl(os.path.join(ROOT, "data", "raw", "human_translated.jsonl"))
    rng = random.Random(42)
    opus_test = {}
    for corpus in ["opus:OpenSubtitles", "opus:TED2020", "opus:News-Commentary", "opus:GlobalVoices"]:
        pool = [h["text"] for h in human_all if h["source"] == corpus and bucket(h["id"], 0.1, 0.1) == "test"]
        rng.shuffle(pool)
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

    results = {
        "dataset_sizes": {
            "test_v2_rows": len(test_v2),
            "holdout_haiku_rows": len(holdout_haiku),
            "hard_test_rows": len(hard_rows),
            "calque_pairs": len(cps),
            "nazarat_essays": len(essay_chunks),
            "nazarat_adapted": len(adapted_chunks),
            "hayati": len(hayati_chunks),
            "sabil_altaj": len(taj_chunks),
            "maba3d_alayyam": len(zayyat_chunks),
        },
        "models": {},
        "top25_tfidf_features": {},
        "voice_rates_v3_train": {},
    }

    tr_v3 = read_jsonl(os.path.join(ROOT, "data", "processed_v3", "train.jsonl"))
    results["voice_rates_v3_train"]["v3_fusha"] = voice_rates_per_10k([r["text"] for r in tr_v3 if r["label"] == 0])
    results["voice_rates_v3_train"]["v3_aranjiya"] = voice_rates_per_10k([r["text"] for r in tr_v3 if r["label"] == 1])

    hard_preds = {}

    def run_eval(mkey: str, score_fn, is_bert: bool):
        print(f"Evaluating {mkey}...", file=sys.stderr)
        def get_pm(texts):
            if is_bert:
                return score_fn(texts)
            return score_fn(texts), None

        p_tv2, m_tv2 = get_pm([r["text"] for r in test_v2])
        p_ho, m_ho = get_pm([r["text"] for r in holdout_haiku])
        p_hd, m_hd = get_pm([r["text"] for r in hard_rows])
        hard_preds[mkey] = [
            {"id": r["id"], "label": r["label"], "p": round(float(p), 4), "pred": int(p >= 0.5)}
            for r, p in zip(hard_rows, p_hd)
        ]

        opus_res = {corp: summarize_binary(get_pm(txts)[0]) for corp, txts in opus_test.items()}

        p_ess, _ = get_pm(essay_chunks)
        p_adp, _ = get_pm(adapted_chunks)
        p_hay, _ = get_pm(hayati_chunks)
        p_taj, _ = get_pm(taj_chunks)
        p_zay, _ = get_pm(zayyat_chunks)

        p_ev, _ = get_pm(essays_with_voice)
        p_env, _ = get_pm(essays_no_voice)

        pf_o, _ = get_pm(test_fusha_orig)
        pf_m, _ = get_pm(test_fusha_mask)
        pt_o, _ = get_pm(test_twin_orig)
        pt_m, _ = get_pm(test_twin_mask)
        f_flips = int(((pf_o < 0.5) & (pf_m >= 0.5)).sum())
        t_flips = int(((pt_o >= 0.5) & (pt_m < 0.5)).sum())

        # Also track Haiku twin p(aranjiya) quartiles for v1/v2a/v3
        haiku_twin_p = np.array([float(p) for r, p in zip(holdout_haiku, p_ho) if r["label"] == 1])

        results["models"][mkey] = {
            "test_v2_pairs": eval_paired_set(test_v2, p_tv2, m_tv2),
            "holdout_haiku": eval_paired_set(holdout_haiku, p_ho, m_ho),
            "holdout_haiku_twin_p_quartiles": {
                k: round(float(v), 4)
                for k, v in zip(["q1", "median", "q3"], np.percentile(haiku_twin_p, [25, 50, 75]))
            },
            "opus_test_per_corpus": opus_res,
            "hard_test": eval_paired_set(hard_rows, p_hd, m_hd),
            "modern_books": {
                "nazarat_essays": summarize_binary(p_ess),
                "nazarat_adapted": summarize_binary(p_adp),
                "hayati": summarize_binary(p_hay),
                "sabil_altaj": summarize_binary(p_taj),
                "maba3d_alayyam": summarize_binary(p_zay),
            },
            "calque_pairs": eval_calque(cps, score_fn, is_bert),
            "regression_masking": {
                "classical_pct_fusha_orig": round(float((pf_o < 0.5).mean() * 100.0), 2),
                "classical_pct_fusha_masked": round(float((pf_m < 0.5).mean() * 100.0), 2),
                "classical_flips_f2a": f_flips,
                "classical_flip_rate_pct": round(f_flips / max(1, len(pf_o)) * 100.0, 2),
                "twins_pct_aranjiya_orig": round(float((pt_o >= 0.5).mean() * 100.0), 2),
                "twins_pct_aranjiya_masked": round(float((pt_m >= 0.5).mean() * 100.0), 2),
                "twins_flips_a2f": t_flips,
            },
            "regression_voice_on_nazarat_essays": {
                "with_voice_words": summarize_binary(p_ev),
                "without_voice_words": summarize_binary(p_env),
                "pct_fusha_gap": round(float((p_env < 0.5).mean() * 100.0 - (p_ev < 0.5).mean() * 100.0), 2),
            },
        }

    for mkey, mpath in [
        ("v1_tfidf", os.path.join(ROOT, "models", "baseline", "baseline.joblib")),
        ("v2a_tfidf", os.path.join(ROOT, "models", "v2a_baseline", "baseline.joblib")),
        ("v3_tfidf", os.path.join(ROOT, "models", "v3_baseline", "baseline.joblib")),
    ]:
        mdl = joblib.load(mpath)
        run_eval(mkey, mdl.predict_proba, is_bert=False)
        results["top25_tfidf_features"][mkey] = mdl.top_word_features(k=25)

    for mkey, mdir, mlen, fay in [
        ("v1_bert", os.path.join(ROOT, "models", "camelbert", "final"), 256, False),
        ("v2a_bert", os.path.join(ROOT, "models", "v2a_camelbert", "final"), 256, True),
        ("v3_bert", os.path.join(ROOT, "models", "v3_camelbert", "final"), 256, True),
    ]:
        bw = BertModelWrapper(mdir, max_len=mlen, fold_alef_ya=fay)
        run_eval(mkey, bw.score_all, is_bert=True)
        bw.close()

    def diffs(keys):
        out = []
        for idx, r in enumerate(hard_rows):
            ps = {k: hard_preds[k][idx]["pred"] for k in keys}
            if len(set(ps.values())) > 1:
                out.append({
                    "id": r["id"], "label": r["label"], "text": r["text"],
                    "scores": {k: hard_preds[k][idx]["p"] for k in keys},
                    "preds": ps,
                })
        return out

    results["hard_test_diffs"] = {
        "tfidf_v1_vs_v2a_vs_v3": diffs(["v1_tfidf", "v2a_tfidf", "v3_tfidf"]),
        "bert_v1_vs_v2a_vs_v3": diffs(["v1_bert", "v2a_bert", "v3_bert"]),
    }

    out_path = os.path.join(ROOT, "results", "v3_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")

    print("\n=== v1 vs v2a vs v3 SUMMARY ===")
    for mk, d in results["models"].items():
        mb = d["modern_books"]
        print(
            f"{mk:10s} | test={d['test_v2_pairs']['acc']:5.1f}% (pw {d['test_v2_pairs']['pairwise_acc']:5.1f}%) "
            f"| haiku={d['holdout_haiku']['acc']:5.1f}% (pw {d['holdout_haiku']['pairwise_acc']:5.1f}%) "
            f"| hard={d['hard_test']['acc']:5.1f}% (pw {d['hard_test']['pairwise_acc']:5.1f}%) "
            f"| calque={d['calque_pairs']['contexts']['overall']['correct']}/38 "
            f"| naz_ess={mb['nazarat_essays']['pct_fusha']:5.1f}% "
            f"| naz_adp={mb['nazarat_adapted']['pct_fusha']:5.1f}% "
            f"| hayati={mb['hayati']['pct_fusha']:5.1f}% "
            f"| taj={mb['sabil_altaj']['pct_fusha']:5.1f}% "
            f"| maba3d={mb['maba3d_alayyam']['pct_fusha']:5.1f}%"
        )


if __name__ == "__main__":
    main()
