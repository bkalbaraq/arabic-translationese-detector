"""Step 1B: Apply strict 6-rule quality filter to pairs_v1.jsonl + benchmark NLLB-1.3B vs 600M on 200 pivots.

Read-only with respect to v1 models and dataset. Saves complete audit to results/v2_step1_audit.json.
"""
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from normalize import normalize, strip_diacritics  # noqa: E402

# Diacritic patterns
_ANY_DIAC = re.compile(r"[\u064b-\u0652\u0670]")
# Short vowel harakat + sukun + tanween damm/kasr (excludes everyday modern tanween-fath ً & shadda ّ)
_VOWEL_HARAKAT = re.compile(r"[\u064c-\u0650\u0652\u0670]")
_QUOTES = re.compile(r'["«»“”]')

LABSE_THRESHOLD = 0.75


def has_repetition_loop(text: str) -> bool:
    words = normalize(text).split()
    n = len(words)
    if n < 6:
        return False
    # 1) Same word >= 3 times in a row or within any 5-word window
    for i in range(n):
        win = words[i : i + 5]
        if Counter(win).most_common(1)[0][1] >= 3:
            return True
    # 2) Same 2-gram >= 3 times within any 10-bigram window
    bigrams = list(zip(words, words[1:]))
    for i in range(len(bigrams)):
        win = bigrams[i : i + 10]
        if Counter(win).most_common(1)[0][1] >= 3:
            return True
    # 3) Any 3-gram occurring >= 3 times anywhere
    if n >= 9:
        trigrams = Counter(zip(words, words[1:], words[2:]))
        if trigrams and trigrams.most_common(1)[0][1] >= 3:
            return True
    # 4) High repeated-n-gram ratio
    if len(bigrams) >= 10 and len(set(bigrams)) / len(bigrams) < 0.72:
        return True
    if n >= 12 and len(set(words)) / n < 0.50:
        return True
    return False


def check_twin(fusha: str, aranjiya: str, labse_sim: float, labse_thr: float = LABSE_THRESHOLD):
    """Return list of triggered rule names (empty = KEPT)."""
    flags = []
    wf = normalize(fusha).split()
    wa = normalize(aranjiya).split()

    # 1) «أي» excess (>= 2 more than source)
    ay_f = sum(1 for w in wf if w == "اي")
    ay_a = sum(1 for w in wa if w == "اي")
    if ay_a - ay_f >= 2:
        flags.append("ay_excess")

    # 2) Repetition loops / high repeated-ngram ratio
    if has_repetition_loop(aranjiya):
        flags.append("repetition_loop")

    # 3) Many quotation marks not in source (>= 4 extra quote chars)
    qf = len(_QUOTES.findall(fusha))
    qa = len(_QUOTES.findall(aranjiya))
    if qa - qf >= 4:
        flags.append("extra_quotes")

    # 4) Length ratio outside [0.6, 1.6]
    ratio = len(wa) / max(1, len(wf))
    if not (0.6 <= ratio <= 1.6):
        flags.append("length_ratio")

    # 5) Vowel harakat in raw output (>= 2 non-tanween-fath/shadda harakat)
    if len(_VOWEL_HARAKAT.findall(aranjiya)) >= 2:
        flags.append("raw_diacritics")

    # 6) LaBSE cross-lingual similarity below threshold
    if labse_sim < labse_thr:
        flags.append("labse_low_sim")

    return flags


def summarize_filter(records):
    rule_counts = Counter()
    kept, removed = [], []
    any_diac_count = 0
    for rec in records:
        if _ANY_DIAC.search(rec["aranjiya"]):
            any_diac_count += 1
        flags = check_twin(rec["fusha"], rec["aranjiya"], rec["labse"])
        rec_out = dict(rec, flags=flags)
        if flags:
            removed.append(rec_out)
            for fl in flags:
                rule_counts[fl] += 1
        else:
            kept.append(rec_out)
    n = len(records)
    return {
        "total": n,
        "kept": len(kept),
        "kept_pct": round(len(kept) / max(1, n) * 100.0, 2),
        "removed": len(removed),
        "removed_pct": round(len(removed) / max(1, n) * 100.0, 2),
        "mean_labse_all": round(float(np.mean([r["labse"] for r in records])), 4),
        "mean_labse_kept": round(float(np.mean([r["labse"] for r in kept])), 4) if kept else 0.0,
        "rule_triggers": dict(rule_counts),
        "note_any_diacritic_incl_tanween_fath": any_diac_count,
    }, kept, removed


def main():
    pairs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", "pairs_v1.jsonl"), encoding="utf-8")]
    labse_rows = json.load(open(os.path.join(ROOT, "results", "pairs_v1_labse_scores.json"), encoding="utf-8"))
    assert len(pairs) == len(labse_rows)

    enriched = []
    for p, lr in zip(pairs, labse_rows):
        enriched.append({
            "pair_id": p["pair_id"],
            "forward": p["forward"],
            "backward": p["backward"],
            "book": p.get("book"),
            "author": p.get("author"),
            "genre": p.get("genre"),
            "fusha": p["fusha"],
            "pivot": p["pivot"],
            "aranjiya": p["aranjiya"],
            "labse": lr["labse"],
        })

    nllb_recs = [r for r in enriched if r["backward"] == "nllb"]
    opus_recs = [r for r in enriched if r["backward"] == "opus_mt"]

    nllb_stats, nllb_kept, nllb_rem = summarize_filter(nllb_recs)
    opus_stats, opus_kept, opus_rem = summarize_filter(opus_recs)
    all_stats, all_kept, all_rem = summarize_filter(enriched)

    # Sample 20 random removed & 20 random kept (stratified: 10 NLLB + 10 OpusMT each for fair view!)
    rng = random.Random(42)
    sample_removed = rng.sample(nllb_rem, min(10, len(nllb_rem))) + rng.sample(opus_rem, min(10, len(opus_rem)))
    sample_kept = rng.sample(nllb_kept, min(10, len(nllb_kept))) + rng.sample(opus_kept, min(10, len(opus_kept)))

    # ----------------------------------------------------------------------- Part 4: NLLB-1.3B vs 600M on 200 pivots
    sample_200 = random.Random(42).sample(nllb_recs, 200)
    pivots_200 = [r["pivot"] for r in sample_200]

    print("Loading facebook/nllb-200-distilled-1.3B ...", file=sys.stderr)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "facebook/nllb-200-distilled-1.3B"
    tok = AutoTokenizer.from_pretrained(model_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()

    tok.src_lang = "eng_Latn"
    tgt_id = tok.convert_tokens_to_ids("arb_Arab")
    bs = 8
    outs_13b = []
    for i in range(0, len(pivots_200), bs):
        batch = pivots_200[i : i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=tgt_id,
                max_length=None,
                max_new_tokens=300,
                num_beams=4,
                no_repeat_ngram_size=4,
            )
        outs_13b.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"\rNLLB-1.3B en->ar: {min(i + bs, len(pivots_200))}/{len(pivots_200)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Score the 200 NLLB-1.3B outputs with LaBSE
    from sentence_transformers import SentenceTransformer

    labse = SentenceTransformer("sentence-transformers/LaBSE", device=dev)
    emb_en = labse.encode(pivots_200, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb_ar_13b = labse.encode([strip_diacritics(t) for t in outs_13b], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sims_13b = (emb_en * emb_ar_13b).sum(axis=1).astype(float)

    recs_600m_200 = sample_200
    recs_13b_200 = []
    for r, ar13, s13 in zip(sample_200, outs_13b, sims_13b):
        nr = dict(r)
        nr["backward"] = "nllb-1.3B"
        nr["aranjiya"] = ar13
        nr["labse"] = round(float(s13), 4)
        recs_13b_200.append(nr)

    stats_600m_200, _, _ = summarize_filter(recs_600m_200)
    stats_13b_200, _, _ = summarize_filter(recs_13b_200)

    # Side-by-side 5 examples where 600M failed and 1.3B passed
    improved_examples = []
    for r6, r13 in zip(recs_600m_200, recs_13b_200):
        f6 = check_twin(r6["fusha"], r6["aranjiya"], r6["labse"])
        f13 = check_twin(r13["fusha"], r13["aranjiya"], r13["labse"])
        if f6 and not f13:
            improved_examples.append({
                "pair_id": r6["pair_id"],
                "fusha": r6["fusha"],
                "pivot": r6["pivot"],
                "nllb_600m": r6["aranjiya"],
                "nllb_600m_flags": f6,
                "nllb_600m_labse": r6["labse"],
                "nllb_13b": r13["aranjiya"],
                "nllb_13b_labse": r13["labse"],
            })

    calque_eval = json.load(open(os.path.join(ROOT, "results", "v1_calque_eval.json"), encoding="utf-8"))

    audit = {
        "filter_config": {
            "ay_excess": "out_ay - src_ay >= 2",
            "repetition_loop": "word>=3 in 5-win OR 2-gram>=3 in 10-win OR 3-gram>=3 OR uniq_bigram<0.72 OR uniq_word<0.50",
            "extra_quotes": "out_quote_chars - src_quote_chars >= 4",
            "length_ratio": "outside [0.6, 1.6]",
            "raw_diacritics": ">= 2 vowel harakat [ٌٍَُِْٰ] in raw output (allows normal modern tanween-fath ً and shadda ّ)",
            "labse_threshold": LABSE_THRESHOLD,
        },
        "filter_summary": {
            "gemini>nllb": nllb_stats,
            "gemini>opus_mt": opus_stats,
            "combined": all_stats,
        },
        "sample_20_removed": sample_removed,
        "sample_20_kept": sample_kept,
        "nllb_13b_vs_600m_200_pivots": {
            "nllb_600m": stats_600m_200,
            "nllb_13b": stats_13b_200,
            "n_600m_failed_and_13b_passed": len(improved_examples),
            "sample_improved": improved_examples[:5],
        },
        "calque_pairs_before": calque_eval,
    }

    out_path = os.path.join(ROOT, "results", "v2_step1_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print("Saved ->", out_path)
    print("\n=== FILTER SUMMARY (pairs_v1.jsonl) ===")
    for k, s in audit["filter_summary"].items():
        print(f"{k:16s} | total={s['total']:4d} | kept={s['kept']:4d} ({s['kept_pct']:5.2f}%) | removed={s['removed']:4d} ({s['removed_pct']:5.2f}%) | meanLaBSE={s['mean_labse_all']:.4f}->{s['mean_labse_kept']:.4f}")
        print("   rules:", s["rule_triggers"])

    print("\n=== NLLB-600M vs NLLB-1.3B (SAME 200 PIVOTS) ===")
    for k, s in [("nllb_600m", stats_600m_200), ("nllb_13b", stats_13b_200)]:
        print(f"{k:12s} | kept={s['kept']:3d}/200 ({s['kept_pct']:5.2f}%) | failed={s['removed']:3d}/200 ({s['removed_pct']:5.2f}%) | meanLaBSE={s['mean_labse_all']:.4f}")
        print("   rules:", s["rule_triggers"])
    print("600M failed -> 1.3B passed:", len(improved_examples))


if __name__ == "__main__":
    main()
