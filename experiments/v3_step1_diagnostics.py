"""v3 Pre-Training Diagnostics on v2a_bert (read-only, no training):
  (a) Hard test item count verification (83 total = 72 paired / 36 pairs + 11 unpaired)
  (b) Single-word occlusion check on v2a_bert:
      - 10 random of the 523 flipped النظرات essays (v1_bert < 0.2, v2a_bert > 0.9)
      - 10 random حياتي chunks with v2a_bert p(عرنجية) > 0.9
      For each chunk: remove one word at a time, rank by drop in logit(عرنجية) (monotonic with p,
      tie-free under saturation), and report top 8 words + Δlogit + Δp + full chunk text.
  (c) Haiku held-out p(عرنجية) distribution on v2a_bert (median, Q1, Q3, etc.) + 5 Haiku twins
      most confidently called فصحى with their classical sources.

Saves -> results/v3_diagnostics.json
"""
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from evaluate import load_hard  # noqa: E402
from normalize import normalize  # noqa: E402
from shortcut_check import load_modern_chunks  # noqa: E402
from v1_diagnostics import map_nazarat_to_epub_chapters  # noqa: E402
from v2_pipeline_step5_eval_sectionE import BertModelWrapper  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def occlude_chunk(bert: BertModelWrapper, raw_text: str):
    """Leave-one-word-out occlusion on the normalized word sequence seen by v2a_bert."""
    norm_text = normalize(raw_text, keep_punct=False, fold_letters=False, fold_alef_ya=True)
    words = norm_text.split()
    variants = [norm_text] + [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
    # Score already-normalized texts directly (normalize is idempotent)
    probs, logits = bert.score_all(variants, bs=64)
    base_p, base_l = float(probs[0]), float(logits[0])
    items = []
    for i, w in enumerate(words):
        p_wo, l_wo = float(probs[i + 1]), float(logits[i + 1])
        items.append({
            "pos": i,
            "word": w,
            "drop_logit": round(base_l - l_wo, 4),
            "drop_p": round(base_p - p_wo, 5),
            "p_without": round(p_wo, 4),
            "logit_without": round(l_wo, 3),
        })
    # Positive drop_logit = removing this word LOWERS p(عرنجية)
    items.sort(key=lambda x: x["drop_logit"], reverse=True)
    return {
        "text": raw_text,
        "base_p": round(base_p, 5),
        "base_logit": round(base_l, 3),
        "n_words": len(words),
        "top8_lowering_words": items[:8],
    }


def main():
    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")

    # (a) Hard test count check
    hard = load_hard(os.path.join(ROOT, "data", "hard_test.jsonl"))
    paired = [r for r in hard if r.get("pair_id") is not None]
    unpaired = [r for r in hard if r.get("pair_id") is None]
    se = json.load(open(os.path.join(ROOT, "results", "v2_sectionE_eval.json"), encoding="utf-8"))
    diag_a = {
        "hard_test_total_rows": len(hard),
        "paired_rows": len(paired),
        "n_pairs": len(paired) // 2,
        "unpaired_rows": len(unpaired),
        "unpaired_ids": [r["id"] for r in unpaired],
        "sectionE_json_hard_n": se["dataset_sizes"]["hard_test_rows"],
        "v2a_bert_hard_acc": se["models"]["v2a_bert"]["hard_test"]["acc"],
        "explanation": (
            "hard_test.jsonl still has all 83 items (72 paired rows = 36 pairs + 11 unpaired عرنجية rows: "
            "old1..old10 and reductio1). Section E evaluated all 83 rows (e.g. v2a_bert 69/83 = 83.13%, "
            "v1_bert 68/83 = 81.93%). The '72 items / 36 pairs' text in the Section E table header was "
            "a prose typo counting only the paired subset."
        ),
    }

    # Load texts for (b) and (c)
    _, essays, _, _, _ = map_nazarat_to_epub_chapters(
        os.path.join(dl, "manfaluti_nazarat_clean.txt"),
        os.path.join(dl, "94061468.epub"),
    )
    hayati = load_modern_chunks(os.path.join(dl, "ahmad_amin_hayati_clean.txt"))
    haiku_pairs = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v2_haiku_holdout.jsonl"))

    # Score essays with v1_bert to get exact 523 flipped essays (v1 < 0.2, v2a > 0.9)
    b1 = BertModelWrapper(os.path.join(ROOT, "models", "camelbert", "final"), 256, False)
    p1_ess, _ = b1.score_all(essays)
    b1.close()

    # Load v2a_bert
    b2 = BertModelWrapper(os.path.join(ROOT, "models", "v2a_camelbert", "final"), 256, True)
    p2_ess, _ = b2.score_all(essays)
    p2_hay, _ = b2.score_all(hayati)
    p2_hku, l2_hku = b2.score_all([r["aranjiya"] for r in haiku_pairs])

    rng = random.Random(42)

    # (b1) 10 random of the 523 flipped النظرات essays
    flipped_idx = [i for i in range(len(essays)) if p1_ess[i] < 0.2 and p2_ess[i] > 0.9]
    rng.shuffle(flipped_idx)
    naz_occ = []
    for idx in flipped_idx[:10]:
        res = occlude_chunk(b2, essays[idx])
        res["v1_p"] = round(float(p1_ess[idx]), 4)
        naz_occ.append(res)

    # (b2) 10 random حياتي chunks with v2a_bert p(عرنجية) > 0.9
    hay_hi_idx = [i for i in range(len(hayati)) if p2_hay[i] > 0.9]
    rng.shuffle(hay_hi_idx)
    hay_occ = [occlude_chunk(b2, hayati[idx]) for idx in hay_hi_idx[:10]]

    b2.close()

    # (c) Haiku held-out p(عرنجية) distribution + 5 most confidently called فصحى
    p2_arr = np.asarray(p2_hku, dtype=float)
    l2_arr = np.asarray(l2_hku, dtype=float)
    order_fusha = np.argsort(l2_arr)
    top5_fusha_haiku = []
    for idx in order_fusha[:5]:
        r = haiku_pairs[int(idx)]
        top5_fusha_haiku.append({
            "pair_id": r["pair_id"],
            "author": r.get("author"),
            "book": r.get("book"),
            "p_aranjiya": round(float(p2_arr[idx]), 6),
            "logit_margin": round(float(l2_arr[idx]), 3),
            "fusha": r["fusha"],
            "pivot": r["pivot"],
            "haiku_aranjiya": r["aranjiya"],
        })

    diag_c = {
        "n_haiku_twins": int(len(p2_arr)),
        "pct_aranjiya_ge_05": round(float((p2_arr >= 0.5).mean() * 100.0), 2),
        "mean_p": round(float(p2_arr.mean()), 4),
        "percentiles_p": {
            k: round(float(v), 4)
            for k, v in zip(
                ["min", "p10", "q1_p25", "median_p50", "q3_p75", "p90", "max"],
                np.percentile(p2_arr, [0, 10, 25, 50, 75, 90, 100]),
            )
        },
        "percentiles_logit": {
            k: round(float(v), 3)
            for k, v in zip(
                ["q1_p25", "median_p50", "q3_p75"],
                np.percentile(l2_arr, [25, 50, 75]),
            )
        },
        "top5_most_confident_fusha_haiku": top5_fusha_haiku,
    }

    out = {
        "diag_a_hard_test_count": diag_a,
        "diag_b_occlusion": {
            "total_flipped_nazarat_essays": len(flipped_idx),
            "total_hayati_gt_09": len(hay_hi_idx),
            "nazarat_10_flipped": naz_occ,
            "hayati_10_high_p": hay_occ,
        },
        "diag_c_haiku_heldout": diag_c,
    }
    out_path = os.path.join(ROOT, "results", "v3_diagnostics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved ->", out_path)
    print("Diag (a):", diag_a["hard_test_total_rows"], "rows (", diag_a["paired_rows"], "paired +", diag_a["unpaired_rows"], "unpaired)")
    print("Diag (b): Nazarat flipped =", len(flipped_idx), "| Hayati p>0.9 =", len(hay_hi_idx))
    print("Diag (c) Haiku p percentiles:", diag_c["percentiles_p"], "| % >= 0.5:", diag_c["pct_aranjiya_ge_05"])


if __name__ == "__main__":
    main()
