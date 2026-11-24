"""Regenerate all 3,780 NLLB-1.3B twins from cached Gemini English pivots (length-sorted GPU batching),
score them with LaBSE, and compute:
  - Punctuation rates per 100 words across all sources
  - CAMeLBERT max_len=128 overflow rates + calque_pairs LOGIT evaluation
  - Length ratio (ar_out / en_pivot) and wordfreq unknown-word distributions for threshold selection

Outputs:
  data/raw/pairs_nllb_13b.jsonl
  results/v2_sectionB_stats.json
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
from wordfreq import word_frequency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from backtranslate import Cache  # noqa: E402
from normalize import normalize, strip_diacritics  # noqa: E402
from shortcut_check import load_modern_chunks  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------- 1. NLLB-1.3B regeneration
def run_nllb_13b_all():
    cache_path = os.path.join(ROOT, "data", "cache", "translations.jsonl")
    cache = Cache(cache_path)

    rows = read_jsonl(os.path.join(ROOT, "data", "raw", "fusha.jsonl"))
    random.Random(1).shuffle(rows)
    rows = rows[:5000]

    items = []
    for r in rows:
        piv = cache.get(Cache.key("gemini", "ar", "en", r["text"]))
        if piv and piv.strip():
            items.append((r, piv))
    print(f"Found {len(items)} cached Gemini ar->en pivots for seed=1", file=sys.stderr)

    # Check cache for nllb_13b en->ar
    outs = [None] * len(items)
    todo_idx = []
    for i, (_, piv) in enumerate(items):
        v = cache.get(Cache.key("nllb_13b", "en", "ar", piv))
        if v is not None:
            outs[i] = v
        else:
            todo_idx.append(i)
    print(f"nllb_13b cache hits: {len(items) - len(todo_idx)}/{len(items)}, todo: {len(todo_idx)}", file=sys.stderr)

    if todo_idx:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = "facebook/nllb-200-distilled-1.3B"
        tok = AutoTokenizer.from_pretrained(model_id)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()
        tok.src_lang = "eng_Latn"
        tgt_id = tok.convert_tokens_to_ids("arb_Arab")

        # Sort todo by pivot character length so batches have minimal padding (2-3x faster, identical output)
        todo_sorted = sorted(todo_idx, key=lambda i: len(items[i][1]))
        done = 0
        t0 = time.time()
        pos = 0
        while pos < len(todo_sorted):
            # Adaptive batch size to avoid WDDM shared-VRAM paging on 8 GB RTX 4060
            max_chars = len(items[todo_sorted[min(pos + 7, len(todo_sorted) - 1)]][1])
            bs = 8 if max_chars < 450 else 4
            batch_idx = todo_sorted[pos : pos + bs]
            pos += bs
            batch_texts = [items[i][1] for i in batch_idx]
            enc = tok(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    forced_bos_token_id=tgt_id,
                    max_length=None,
                    max_new_tokens=300,
                    num_beams=4,
                    no_repeat_ngram_size=4,
                )
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            del enc, gen
            if bs <= 4 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            for idx, tr in zip(batch_idx, dec):
                outs[idx] = tr
                if tr:
                    cache.put(Cache.key("nllb_13b", "en", "ar", items[idx][1]), tr)
            done += len(batch_idx)
            elapsed = time.time() - t0
            rate = done / max(1e-6, elapsed)
            eta = (len(todo_sorted) - done) / max(1e-6, rate)
            print(f"\rnllb_13b en->ar: {done}/{len(todo_sorted)} ({rate:.1f}/s, ETA {eta/60:.1f}m)", end="", file=sys.stderr)
        print(file=sys.stderr)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Score all NLLB-1.3B outputs with LaBSE
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labse = SentenceTransformer("sentence-transformers/LaBSE", device=dev)
    pivots = [piv for _, piv in items]
    arabs = [strip_diacritics(o or "") for o in outs]
    emb_en = labse.encode(pivots, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb_ar = labse.encode(arabs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sims = (emb_en * emb_ar).sum(axis=1).astype(float)
    del labse
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path = os.path.join(ROOT, "data", "raw", "pairs_nllb_13b.jsonl")
    records = []
    with open(out_path, "w", encoding="utf-8") as f:
        for (r, piv), ar, sim in zip(items, outs, sims):
            if not ar or not re.search(r"[ء-ي]", ar):
                continue
            rec = {
                "pair_id": r["id"],
                "fusha": r["text"],
                "aranjiya": ar,
                "pivot": piv,
                "pivot_lang": "en",
                "forward": "gemini",
                "backward": "nllb_13b",
                "labse": round(float(sim), 4),
                "book": r.get("book"),
                "author": r.get("author"),
                "genre": r.get("genre"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
    print(f"Wrote {len(records)} NLLB-1.3B pairs -> {out_path}", file=sys.stderr)
    return records


# --------------------------------------------------------------------------- 2. Punctuation rates
PUNCT_PATTERNS = {
    "commas": re.compile(r"[,،]"),
    "periods": re.compile(r"\."),
    "colons": re.compile(r":"),
    "semicolons": re.compile(r"[;؛]"),
    "questions": re.compile(r"[?؟]"),
    "quotes": re.compile(r'["\'«»“”‘’]'),
    "brackets": re.compile(r"[\(\)\[\]\{\}]"),
    "dashes": re.compile(r"[-–—]"),
}


def punct_rates_per_100_words(texts):
    total_words = sum(len(t.split()) for t in texts)
    counts = {k: sum(len(pat.findall(t)) for t in texts) for k, pat in PUNCT_PATTERNS.items()}
    total_p = sum(counts.values())
    rates = {k: round(v / max(1, total_words) * 100.0, 2) for k, v in counts.items()}
    rates["total_punct"] = round(total_p / max(1, total_words) * 100.0, 2)
    return {"chunks": len(texts), "total_words": total_words, "per_100_words": rates}


# --------------------------------------------------------------------------- 3. Unknown-word helper
_STRIP_PUNCT_NUM = re.compile(r"[^\u0621-\u064a\s]")


def unknown_word_stats(raw_ar: str):
    """Run wordfreq on RAW output with ONLY diacritics & punctuation removed (NO hamza/alef/ى folding)."""
    s = strip_diacritics(raw_ar)
    s = _STRIP_PUNCT_NUM.sub(" ", s)
    words = [w for w in s.split() if len(w) >= 2]
    if not words:
        return 1.0, []
    unk = [w for w in words if word_frequency(w, "ar") == 0.0]
    return len(unk) / len(words), unk


# --------------------------------------------------------------------------- 4. CAMeLBERT max_len & logit calque
def bert_lengths_and_logits():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = os.path.join(ROOT, "models", "camelbert", "final")
    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()

    def len_stats(texts):
        lens = [len(tok.encode(normalize(t, fold_letters=False))) for t in texts]
        arr = np.array(lens)
        return {
            "n": int(len(arr)),
            "median": int(np.median(arr)),
            "p95": int(np.percentile(arr, 95)),
            "max": int(arr.max()),
            "pct_gt_128": round(float((arr > 128).mean() * 100.0), 2),
            "pct_gt_256": round(float((arr > 256).mean() * 100.0), 2),
        }

    train_rows = read_jsonl(os.path.join(ROOT, "data", "processed", "train.jsonl"))
    test_rows = read_jsonl(os.path.join(ROOT, "data", "processed", "test.jsonl"))
    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")
    naz_chunks = load_modern_chunks(os.path.join(dl, "manfaluti_nazarat_clean.txt"))
    cps = read_jsonl(os.path.join(ROOT, "data", "calque_pairs.jsonl"))

    lengths = {
        "v1_train_max_len_setting": 128,
        "train_jsonl": len_stats([r["text"] for r in train_rows]),
        "test_jsonl": len_stats([r["text"] for r in test_rows]),
        "nazarat_chunks": len_stats(naz_chunks),
        "calque_contexts": len_stats([r["calque_context"] for r in cps]),
        "calque_sentences": len_stats([r["calque_sentence"] for r in cps]),
    }

    def score_logits(texts, max_len=512):
        prepped = [normalize(t, fold_letters=False) for t in texts]
        margins = []
        probs = []
        with torch.no_grad():
            for i in range(0, len(prepped), 32):
                enc = tok(prepped[i : i + 32], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)
                lg = model(**enc).logits.float().cpu().numpy()
                # logit margin = logit(aranjiya) - logit(fusha)
                margins.extend((lg[:, 1] - lg[:, 0]).tolist())
                e = np.exp(lg - lg.max(-1, keepdims=True))
                probs.extend((e / e.sum(-1, keepdims=True))[:, 1].tolist())
        return np.array(margins, dtype=float), np.array(probs, dtype=float)

    lc_ctx, pc_ctx = score_logits([r["calque_context"] for r in cps])
    ln_ctx, pn_ctx = score_logits([r["native_context"] for r in cps])
    lc_snt, pc_snt = score_logits([r["calque_sentence"] for r in cps])
    ln_snt, pn_snt = score_logits([r["native_sentence"] for r in cps])

    def summarize_logits(lc, ln, pc, pn, is_ctx=True):
        def agg(mask=None):
            idx = range(len(cps)) if mask is None else [i for i, m in enumerate(mask) if m]
            ldiffs = [float(lc[i] - ln[i]) for i in idx]
            pdiffs = [float(pc[i] - pn[i]) for i in idx]
            ok = [d > 0 for d in ldiffs]
            return {
                "n": len(idx),
                "correct_by_logit": int(sum(ok)),
                "pct_correct_by_logit": round(sum(ok) / max(1, len(idx)) * 100.0, 2),
                "mean_logit_diff": round(float(np.mean(ldiffs)), 4),
                "mean_prob_diff": round(float(np.mean(pdiffs)), 4),
            }

        wrongs = []
        for i, r in enumerate(cps):
            if lc[i] <= ln[i]:
                wrongs.append({
                    "id": r["id"],
                    "book": r["book"],
                    "direction": r["direction"],
                    "calque_types": r.get("calque_types"),
                    "logit_calque": round(float(lc[i]), 4),
                    "logit_native": round(float(ln[i]), 4),
                    "logit_diff": round(float(lc[i] - ln[i]), 4),
                    "p_calque": round(float(pc[i]), 4),
                    "p_native": round(float(pn[i]), 4),
                    "calque_sentence": r["calque_sentence"],
                    "native_sentence": r["native_sentence"],
                })
        dirs = sorted({r["direction"] for r in cps})
        books = sorted({r["book"] for r in cps})
        return {
            "overall": agg(),
            "by_direction": {d: agg([r["direction"] == d for r in cps]) for d in dirs},
            "by_book": {b: agg([r["book"] == b for r in cps]) for b in books},
            "wrong_pairs": wrongs,
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "token_lengths": lengths,
        "calque_v1_camelbert_logits": {
            "contexts": summarize_logits(lc_ctx, ln_ctx, pc_ctx, pn_ctx, True),
            "sentences": summarize_logits(lc_snt, ln_snt, pc_snt, pn_snt, False),
        },
    }


def main():
    # Step 1: Regenerate all 3,780 NLLB-1.3B twins + LaBSE
    nllb13_recs = run_nllb_13b_all()

    # Load OpusMT twins + LaBSE from Step 1A
    pairs_v1 = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v1.jsonl"))
    labse_v1 = json.load(open(os.path.join(ROOT, "results", "pairs_v1_labse_scores.json"), encoding="utf-8"))
    opusmt_recs = []
    for p, lr in zip(pairs_v1, labse_v1):
        if p["backward"] == "opus_mt":
            rec = dict(p)
            rec["labse"] = lr["labse"]
            opusmt_recs.append(rec)

    # Step 2: Punctuation table across all requested sources
    train_rows = read_jsonl(os.path.join(ROOT, "data", "processed", "train.jsonl"))
    opus_all = read_jsonl(os.path.join(ROOT, "data", "raw", "human_translated.jsonl"))
    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")
    naz_chunks = load_modern_chunks(os.path.join(dl, "manfaluti_nazarat_clean.txt"))
    hay_chunks = load_modern_chunks(os.path.join(dl, "ahmad_amin_hayati_clean.txt"))

    punct_sources = {
        "classical_fusha_train": [r["text"] for r in train_rows if r["source"] == "openiti"],
        "opus_mt_twins": [r["aranjiya"] for r in opusmt_recs],
        "nllb_13b_twins": [r["aranjiya"] for r in nllb13_recs],
        "manfaluti_nazarat": naz_chunks,
        "ahmad_amin_hayati": hay_chunks,
        "opus:OpenSubtitles": [r["text"] for r in opus_all if r["source"] == "opus:OpenSubtitles"],
        "opus:TED2020": [r["text"] for r in opus_all if r["source"] == "opus:TED2020"],
        "opus:News-Commentary": [r["text"] for r in opus_all if r["source"] == "opus:News-Commentary"],
        "opus:GlobalVoices": [r["text"] for r in opus_all if r["source"] == "opus:GlobalVoices"],
    }
    punct_table = {k: punct_rates_per_100_words(v) for k, v in punct_sources.items()}

    # Step 3: Length ratio (ar_out / en_pivot) & wordfreq unknown-word distributions
    dists = {}
    for name, recs in [("opus_mt", opusmt_recs), ("nllb_13b", nllb13_recs)]:
        lrs = []
        unks = []
        unk_samples = []
        for r in recs:
            en_w = len(r["pivot"].split())
            ar_w = len(normalize(r["aranjiya"]).split())
            lrs.append(ar_w / max(1, en_w))
            u_ratio, u_words = unknown_word_stats(r["aranjiya"])
            unks.append(u_ratio)
            unk_samples.append((u_ratio, u_words[:8], r["aranjiya"][:140]))
        lrs_arr = np.array(lrs)
        unks_arr = np.array(unks)
        dists[name] = {
            "n": len(recs),
            "ar_over_en_len_pcts": {
                k: round(float(v), 3)
                for k, v in zip(["p1", "p2", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p98", "p99"],
                                np.percentile(lrs_arr, [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]))
            },
            "unknown_word_pcts": {
                k: round(float(v * 100.0), 2)
                for k, v in zip(["p50", "p75", "p80", "p85", "p90", "p95", "p98", "p99"],
                                np.percentile(unks_arr, [50, 75, 80, 85, 90, 95, 98, 99]))
            },
            "unknown_bands_examples": {},
        }
        for lo in [0.10, 0.15, 0.20, 0.25, 0.30]:
            hi = lo + 0.05
            band = [x for x in unk_samples if lo <= x[0] < hi]
            dists[name]["unknown_bands_examples"][f"{int(lo*100)}-{int(hi*100)}%"] = {
                "count": len(band),
                "examples": [{"unk_pct": round(x[0] * 100, 1), "unk_words": x[1], "text": x[2]} for x in band[:3]],
            }

    # Step 4: CAMeLBERT max_len + calque logits
    bert_info = bert_lengths_and_logits()

    out = {
        "punctuation_per_100_words": punct_table,
        "distributions_for_thresholds": dists,
        "bert_max_len_and_calque_logits": bert_info,
    }
    out_path = os.path.join(ROOT, "results", "v2_sectionB_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nSaved ->", out_path)
    print("\n=== PUNCTUATION PER 100 WORDS ===")
    cols = ["commas", "periods", "colons", "semicolons", "questions", "quotes", "brackets", "dashes", "total_punct"]
    print(f"{'source':24s} | " + " | ".join(f"{c[:6]:>6s}" for c in cols))
    for k, v in punct_table.items():
        r = v["per_100_words"]
        print(f"{k:24s} | " + " | ".join(f"{r[c]:6.2f}" for c in cols))

    print("\n=== DISTRIBUTIONS ===")
    for name, d in dists.items():
        print(name, "ar/en len:", d["ar_over_en_len_pcts"])
        print(name, "unk %   :", d["unknown_word_pcts"])

    print("\n=== CAMELBERT MAX_LEN & LOGIT CALQUE ===")
    print("Lengths:", json.dumps(bert_info["token_lengths"], indent=1))
    for scope in ["contexts", "sentences"]:
        ov = bert_info["calque_v1_camelbert_logits"][scope]["overall"]
        print(f"calque_logits {scope}: {ov}")


if __name__ == "__main__":
    main()
