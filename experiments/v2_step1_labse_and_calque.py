"""Step 1A: Compute LaBSE cross-lingual scores for pairs_v1.jsonl + evaluate calque_pairs.jsonl.

Read-only with respect to v1 models. Caches LaBSE scores to results/pairs_v1_labse_scores.json
and saves calque_pairs evaluation to results/v1_calque_eval.json.
"""
import json
import os
import re
import sys
from collections import Counter

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from normalize import normalize, strip_diacritics  # noqa: E402
from shortcut_check import BertScorer  # noqa: E402
from train_baseline import Baseline  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def eval_calque_pairs(cps, model_name, scorer):
    # Score both contexts (primary) and sentences
    c_ctx = [r["calque_context"] for r in cps]
    n_ctx = [r["native_context"] for r in cps]
    c_snt = [r["calque_sentence"] for r in cps]
    n_snt = [r["native_sentence"] for r in cps]

    pc_ctx = np.asarray(scorer(c_ctx), dtype=float)
    pn_ctx = np.asarray(scorer(n_ctx), dtype=float)
    pc_snt = np.asarray(scorer(c_snt), dtype=float)
    pn_snt = np.asarray(scorer(n_snt), dtype=float)

    def agg(pc, pn, mask=None):
        idx = range(len(cps)) if mask is None else [i for i, m in enumerate(mask) if m]
        diffs = [float(pc[i] - pn[i]) for i in idx]
        ok = [d > 0 for d in diffs]
        return {
            "n": len(idx),
            "correct": int(sum(ok)),
            "pct_correct": round(sum(ok) / max(1, len(idx)) * 100.0, 2),
            "mean_diff": round(float(np.mean(diffs)), 4) if diffs else 0.0,
            "mean_p_calque": round(float(np.mean([pc[i] for i in idx])), 4) if idx else 0.0,
            "mean_p_native": round(float(np.mean([pn[i] for i in idx])), 4) if idx else 0.0,
        }

    def wrongs(pc, pn, c_texts, n_texts):
        out = []
        for i, r in enumerate(cps):
            if pc[i] <= pn[i]:
                out.append({
                    "id": r["id"],
                    "book": r["book"],
                    "direction": r["direction"],
                    "calque_types": r.get("calque_types"),
                    "p_calque": round(float(pc[i]), 4),
                    "p_native": round(float(pn[i]), 4),
                    "diff": round(float(pc[i] - pn[i]), 4),
                    "calque_text": c_texts[i],
                    "native_text": n_texts[i],
                    "calque_sentence": r["calque_sentence"],
                    "native_sentence": r["native_sentence"],
                })
        return out

    dirs = sorted({r["direction"] for r in cps})
    books = sorted({r["book"] for r in cps})

    return {
        "model": model_name,
        "contexts": {
            "overall": agg(pc_ctx, pn_ctx),
            "by_direction": {d: agg(pc_ctx, pn_ctx, [r["direction"] == d for r in cps]) for d in dirs},
            "by_book": {b: agg(pc_ctx, pn_ctx, [r["book"] == b for r in cps]) for b in books},
            "wrong_pairs": wrongs(pc_ctx, pn_ctx, c_ctx, n_ctx),
        },
        "sentences": {
            "overall": agg(pc_snt, pn_snt),
            "by_direction": {d: agg(pc_snt, pn_snt, [r["direction"] == d for r in cps]) for d in dirs},
            "by_book": {b: agg(pc_snt, pn_snt, [r["book"] == b for r in cps]) for b in books},
            "wrong_pairs": wrongs(pc_snt, pn_snt, c_snt, n_snt),
        },
    }


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    # 1. Evaluate calque_pairs.jsonl on v1 TF-IDF, v1 CAMeLBERT (max_len=512), and no-OPUS TF-IDF
    cps = read_jsonl(os.path.join(ROOT, "data", "calque_pairs.jsonl"))
    v1_tfidf = joblib.load(os.path.join(ROOT, "models", "baseline", "baseline.joblib"))
    v1_bert = BertScorer(os.path.join(ROOT, "models", "camelbert", "final"), max_len=512, bs=32)

    train_rows = read_jsonl(os.path.join(ROOT, "data", "processed", "train.jsonl"))
    paired_train = [r for r in train_rows if r.get("pair_id") is not None]
    no_opus_tfidf = Baseline(C=4.0).fit([r["text"] for r in paired_train], [r["label"] for r in paired_train])

    calque_res = {
        "v1_tfidf": eval_calque_pairs(cps, "v1_tfidf", v1_tfidf.predict_proba),
        "no_opus_tfidf": eval_calque_pairs(cps, "no_opus_tfidf", no_opus_tfidf.predict_proba),
        "v1_camelbert": eval_calque_pairs(cps, "v1_camelbert", v1_bert.predict_proba),
    }
    calque_path = os.path.join(ROOT, "results", "v1_calque_eval.json")
    with open(calque_path, "w", encoding="utf-8") as f:
        json.dump(calque_res, f, ensure_ascii=False, indent=2)
    print("Saved calque eval ->", calque_path)

    # Free BERT VRAM before loading LaBSE
    del v1_bert
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Compute LaBSE similarity between English pivot and Arabic output for all pairs_v1.jsonl
    from sentence_transformers import SentenceTransformer

    labse = SentenceTransformer("sentence-transformers/LaBSE", device="cuda" if torch.cuda.is_available() else "cpu")
    pairs = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v1.jsonl"))
    pivots = [p["pivot"] for p in pairs]
    arabs = [strip_diacritics(p["aranjiya"]) for p in pairs]

    emb_en = labse.encode(pivots, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    emb_ar = labse.encode(arabs, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    sims = (emb_en * emb_ar).sum(axis=1).astype(float)

    scores_out = []
    for p, s in zip(pairs, sims):
        scores_out.append({
            "pair_id": p["pair_id"],
            "backward": p["backward"],
            "labse": round(float(s), 4),
        })
    labse_path = os.path.join(ROOT, "results", "pairs_v1_labse_scores.json")
    with open(labse_path, "w", encoding="utf-8") as f:
        json.dump(scores_out, f, ensure_ascii=False)
    print("Saved LaBSE scores ->", labse_path)

    # Print distribution & sample borderline pairs so we can pick the threshold cleanly
    for bwd in ["nllb", "opus_mt"]:
        sub = np.array([s for p, s in zip(pairs, sims) if p["backward"] == bwd])
        pcts = np.percentile(sub, [1, 5, 10, 15, 20, 25, 50, 75, 90])
        print(f"\nLaBSE {bwd:8s} (n={len(sub)}): mean={sub.mean():.4f} | p1/5/10/15/20/25/50/75/90 = " +
              " ".join(f"{x:.3f}" for x in pcts))
        for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            print(f"  < {thr:.2f}: {int((sub < thr).sum()):4d} ({(sub < thr).mean()*100:5.2f}%)")

    # Show 2 NLLB examples in each 0.05 LaBSE band from 0.50 to 0.80
    print("\n--- Borderline LaBSE NLLB examples ---")
    for lo in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        hi = lo + 0.05
        band = [(p, float(s)) for p, s in zip(pairs, sims) if p["backward"] == "nllb" and lo <= s < hi]
        print(f"\nBand [{lo:.2f}, {hi:.2f}) (n={len(band)}):")
        for p, s in band[:2]:
            print(f"  sim={s:.3f} | EN: {p['pivot'][:120]}")
            print(f"            | AR: {p['aranjiya'][:120]}")

    # Print calque summary
    print("\n=== CALQUE PAIRS SUMMARY ===")
    for m in ["v1_tfidf", "no_opus_tfidf", "v1_camelbert"]:
        c = calque_res[m]["contexts"]
        s = calque_res[m]["sentences"]
        print(f"\n{m}:")
        print(f"  CONTEXTS : {c['overall']['correct']}/{c['overall']['n']} ({c['overall']['pct_correct']}%) mean_diff={c['overall']['mean_diff']:+.4f}")
        for d, v in c["by_direction"].items():
            print(f"    dir={d:18s}: {v['correct']}/{v['n']} ({v['pct_correct']}%) mean_diff={v['mean_diff']:+.4f}")
        for b, v in c["by_book"].items():
            print(f"    book={b:17s}: {v['correct']}/{v['n']} ({v['pct_correct']}%) mean_diff={v['mean_diff']:+.4f}")
        print(f"  SENTENCES: {s['overall']['correct']}/{s['overall']['n']} ({s['overall']['pct_correct']}%) mean_diff={s['overall']['mean_diff']:+.4f}")
        for d, v in s["by_direction"].items():
            print(f"    dir={d:18s}: {v['correct']}/{v['n']} ({v['pct_correct']}%) mean_diff={v['mean_diff']:+.4f}")


if __name__ == "__main__":
    main()
