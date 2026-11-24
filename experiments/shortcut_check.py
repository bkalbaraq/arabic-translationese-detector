"""v1 Shortcut Diagnostic (measurement only — does NOT modify training data or models).

Tests whether the v1 models (TF-IDF Baseline and CAMeLBERT-mix) rely on genuine
Arab style vs. surface signs of classical books (kunya/nasab names, isnad formulas).

Outputs:
  results/v1_shortcut_check.json
"""
import json
import os
import re
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from normalize import light_clean, normalize, strip_diacritics  # noqa: E402
from openiti_extract import chunk_paragraph  # noqa: E402
from train_baseline import Baseline  # noqa: F401, E402  (needed for joblib unpickling)

# --------------------------------------------------------------------------- masking (Test 1)
_BOUNDARY_L = r"(?:^|(?<=[\s،؛:.,!?؟«»\"'()\-–—]))"
_BOUNDARY_R = r"(?=$|[\s،؛:.,!?؟«»\"'()\-–—])"

# Kunya / nasab: أبو/أبي/أبا/ابن/بن/بنت/أم + following word (including compound عبد X)
# Preserve optional proclitic prefix (و/ف/ب/ل/ك) so surrounding grammar stays intact.
_KUNYA_NASAB = re.compile(
    _BOUNDARY_L
    + r"([وفبلك])?(?:أبو|أبي|أبا|ابو|ابي|ابا|ابن|بن|بنت|أم|ام)\s+(?:عبد\s+\S+|\S+)"
)

# Isnad formulas specified in Test 1 (with optional و/ف prefix)
_ISNAD_FORMULA = re.compile(
    _BOUNDARY_L + r"[وف]?(?:حدثنا|حدثني|أخبرنا|أخبرني|أنبأنا)" + _BOUNDARY_R
)

# "عن" sitting between two names in an isnad chain (after kunya/nasab -> فلان)
_AN_BETWEEN_NAMES = re.compile(
    r"(فلان(?:\s+\S+)?)\s+[وف]?عن\s+(فلان)"
)
_AN_CHAIN = re.compile(
    _BOUNDARY_L + r"[وف]?عن\s+(\S+(?:\s+فلان)?)(?=\s+[وف]?عن\s+)"
)
_LEADING_AN_FULAN = re.compile(r"^\s*[وف]?عن\s+(فلان)")


def mask_classical_surface(text: str) -> str:
    s = text
    # Iteratively collapse chained kunya/nasab (e.g. أبو بكر بن أبي شيبة -> فلان بن فلان -> فلان)
    for _ in range(4):
        new_s = _KUNYA_NASAB.sub(lambda m: (m.group(1) or "") + "فلان", s)
        if new_s == s:
            break
        s = new_s

    # Remove isnad formulas
    s = _ISNAD_FORMULA.sub(" ", s)

    # Remove "عن" sitting between names in a narration chain
    for _ in range(4):
        new_s = _AN_BETWEEN_NAMES.sub(r"\1 \2", s)
        new_s = _AN_CHAIN.sub(r"\1", new_s)
        if new_s == s:
            break
        s = new_s
    s = _LEADING_AN_FULAN.sub(r"\1", s)

    # Collapse repeated adjacent "فلان فلان" from multi-part nasabs into one "فلان"
    s = re.sub(r"(?:^|(?<=\s))فلان(?:\s+فلان)+", "فلان", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------------- modern text chunking
def load_modern_chunks(path: str, min_words: int = 15, max_words: int = 60):
    """Chunk modern text identically to training: strip tashkeel + light_clean + chunk_paragraph."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Explicitly remove ALL tashkeel first, then apply light_clean (same as openiti_extract)
            clean = light_clean(strip_diacritics(line))
            if not clean:
                continue
            chunks.extend(chunk_paragraph(clean, min_words, max_words))
    return chunks


# --------------------------------------------------------------------------- scoring helpers
def score_baseline(model, texts):
    return model.predict_proba(texts).astype(float)


def explain_baseline_ngrams(model, text, top_k=8):
    """Exact logistic-regression contributions (tfidf * coef) for word and char n-grams."""
    norm = normalize(text)
    xc = model.char.transform([norm])
    xw = model.word.transform([norm])
    n_char = len(model.char.vocabulary_)
    coefs = model.clf.coef_[0]

    word_names = model.word.get_feature_names_out()
    char_names = model.char.get_feature_names_out()

    word_contribs = []
    for idx, val in zip(xw.indices, xw.data):
        c = float(val * coefs[n_char + idx])
        if c > 0:
            word_contribs.append((str(word_names[idx]), round(c, 4)))
    word_contribs.sort(key=lambda x: x[1], reverse=True)

    char_contribs = []
    for idx, val in zip(xc.indices, xc.data):
        c = float(val * coefs[idx])
        if c > 0:
            char_contribs.append((str(char_names[idx]), round(c, 4)))
    char_contribs.sort(key=lambda x: x[1], reverse=True)

    return {
        "top_word_ngrams": word_contribs[:top_k],
        "top_char_ngrams": char_contribs[:top_k],
    }


class BertScorer:
    def __init__(self, model_dir: str, max_len: int = 128, bs: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.dev).eval()
        self.max_len = max_len
        self.bs = bs

    def predict_proba(self, texts):
        torch = self.torch
        prepped = [normalize(t, fold_letters=False) for t in texts]
        probs = []
        with torch.no_grad():
            for i in range(0, len(prepped), self.bs):
                batch = prepped[i : i + self.bs]
                enc = self.tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                ).to(self.dev)
                logits = self.model(**enc).logits.float().cpu().numpy()
                e = np.exp(logits - logits.max(-1, keepdims=True))
                p = (e / e.sum(-1, keepdims=True))[:, 1]
                probs.extend(p.tolist())
        return np.array(probs, dtype=float)


def summarize(probs):
    probs = np.asarray(probs, dtype=float)
    pct_fusha = float((probs < 0.5).mean() * 100.0) if len(probs) else 0.0
    mean_p = float(probs.mean()) if len(probs) else 0.0
    return {
        "n": int(len(probs)),
        "pct_fusha": round(pct_fusha, 2),
        "pct_aranjiya": round(100.0 - pct_fusha, 2),
        "mean_p_aranjiya": round(mean_p, 4),
    }


def flip_stats(orig_texts, masked_texts, p_orig, p_masked, max_examples=10):
    f2a_idx = [i for i in range(len(p_orig)) if p_orig[i] < 0.5 and p_masked[i] >= 0.5]
    a2f_idx = [i for i in range(len(p_orig)) if p_orig[i] >= 0.5 and p_masked[i] < 0.5]
    # Sort flipped examples by largest probability shift
    f2a_sorted = sorted(f2a_idx, key=lambda i: p_masked[i] - p_orig[i], reverse=True)
    examples = [
        {
            "index": int(i),
            "p_orig": round(float(p_orig[i]), 4),
            "p_masked": round(float(p_masked[i]), 4),
            "original": orig_texts[i],
            "masked": masked_texts[i],
        }
        for i in f2a_sorted[:max_examples]
    ]
    changed = sum(1 for a, b in zip(orig_texts, masked_texts) if a != b)
    return {
        "chunks_modified_by_mask": int(changed),
        "flipped_fusha_to_aranjiya": int(len(f2a_idx)),
        "flipped_fusha_to_aranjiya_pct_of_all": round(len(f2a_idx) / max(1, len(p_orig)) * 100.0, 2),
        "flipped_aranjiya_to_fusha": int(len(a2f_idx)),
        "flipped_aranjiya_to_fusha_pct_of_all": round(len(a2f_idx) / max(1, len(p_orig)) * 100.0, 2),
        "examples_fusha_to_aranjiya": examples,
    }


def main():
    baseline = joblib.load(os.path.join(ROOT, "models", "baseline", "baseline.joblib"))
    bert = BertScorer(os.path.join(ROOT, "models", "camelbert", "final"))

    test_rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "processed", "test.jsonl"), encoding="utf-8")]

    # Test 1 sets:
    # (a) All classical fusha chunks in test.jsonl (615)
    classical_all = [r for r in test_rows if r["label"] == 0]
    # (b) Exact paired subset: classical fusha chunks with a twin + their عرنجية twins (431 each)
    pair_map_f = {r["pair_id"]: r for r in test_rows if r["label"] == 0 and r.get("pair_id")}
    pair_map_a = {r["pair_id"]: r for r in test_rows if r["label"] == 1 and r.get("pair_id")}
    common_pids = sorted(set(pair_map_f) & set(pair_map_a))
    classical_paired = [pair_map_f[pid] for pid in common_pids]
    aranjiya_twins = [pair_map_a[pid] for pid in common_pids]

    def run_mask_group(name, rows):
        orig = [r["text"] for r in rows]
        masked = [mask_classical_surface(t) for t in orig]
        out = {}
        for mname, scorer in [("tfidf", lambda xs: score_baseline(baseline, xs)), ("camelbert", bert.predict_proba)]:
            po = scorer(orig)
            pm = scorer(masked)
            out[mname] = {
                "original": summarize(po),
                "masked": summarize(pm),
                "flips": flip_stats(orig, masked, po, pm, max_examples=10),
            }
        return out

    test1 = {
        "classical_all_test": run_mask_group("classical_all_test", classical_all),
        "classical_paired_test": run_mask_group("classical_paired_test", classical_paired),
        "aranjiya_twins_test": run_mask_group("aranjiya_twins_test", aranjiya_twins),
    }

    # Test 2 + Extra modern texts
    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")
    modern_files = [
        ("manfaluti_nazarat", os.path.join(dl, "manfaluti_nazarat_clean.txt"), True),
        ("mansour_sabil_albayan", os.path.join(dl, "mansour_sabil_albayan.txt"), False),
        ("nahw_reading_advice", os.path.join(dl, "nahw_reading_advice.txt"), False),
        ("manfaluti_sabil_altaj", os.path.join(dl, "manfaluti_sabil_altaj_clean.txt"), False),
        ("ahmad_amin_hayati", os.path.join(dl, "ahmad_amin_hayati_clean.txt"), False),
    ]

    modern_results = {}
    for key, path, is_main in modern_files:
        chunks = load_modern_chunks(path)
        p_tf = score_baseline(baseline, chunks)
        p_bt = bert.predict_proba(chunks)
        entry = {
            "file": os.path.basename(path),
            "tfidf": summarize(p_tf),
            "camelbert": summarize(p_bt),
        }
        if is_main:
            # Top 10 most confidently predicted عرنجية for BOTH models (with TF-IDF n-gram explanations)
            top_tf_idx = np.argsort(p_tf)[::-1][:10]
            top_bt_idx = np.argsort(p_bt)[::-1][:10]
            entry["top10_aranjiya_tfidf"] = [
                {
                    "rank": r + 1,
                    "p_tfidf": round(float(p_tf[i]), 4),
                    "p_camelbert": round(float(p_bt[i]), 4),
                    "text": chunks[i],
                    "tfidf_drivers": explain_baseline_ngrams(baseline, chunks[i], top_k=8),
                }
                for r, i in enumerate(top_tf_idx)
            ]
            entry["top10_aranjiya_camelbert"] = [
                {
                    "rank": r + 1,
                    "p_camelbert": round(float(p_bt[i]), 4),
                    "p_tfidf": round(float(p_tf[i]), 4),
                    "text": chunks[i],
                    "tfidf_drivers": explain_baseline_ngrams(baseline, chunks[i], top_k=8),
                }
                for r, i in enumerate(top_bt_idx)
            ]
        modern_results[key] = entry

    results = {
        "test1_masking": test1,
        "test2_main_nazarat": modern_results["manfaluti_nazarat"],
        "extra_texts": {k: modern_results[k] for k in [
            "mansour_sabil_albayan", "nahw_reading_advice", "manfaluti_sabil_altaj", "ahmad_amin_hayati"
        ]},
    }

    out_path = os.path.join(ROOT, "results", "v1_shortcut_check.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved ->", out_path)
    # Print compact summary table
    print("\n=== SUMMARY ===")
    for gname, g in test1.items():
        for m in ["tfidf", "camelbert"]:
            o, mk, fl = g[m]["original"], g[m]["masked"], g[m]["flips"]
            print(
                f"{gname:22s} | {m:9s} | n={o['n']:4d} | orig %fusha={o['pct_fusha']:6.2f} (meanP={o['mean_p_aranjiya']:.4f}) "
                f"| masked %fusha={mk['pct_fusha']:6.2f} (meanP={mk['mean_p_aranjiya']:.4f}) "
                f"| mod={fl['chunks_modified_by_mask']:3d} | f->a={fl['flipped_fusha_to_aranjiya']:3d} ({fl['flipped_fusha_to_aranjiya_pct_of_all']:.2f}%) "
                f"| a->f={fl['flipped_aranjiya_to_fusha']:3d} ({fl['flipped_aranjiya_to_fusha_pct_of_all']:.2f}%)"
            )
    print()
    for key, path, _ in modern_files:
        e = modern_results[key]
        for m in ["tfidf", "camelbert"]:
            s = e[m]
            print(f"{key:22s} | {m:9s} | n={s['n']:4d} | %fusha={s['pct_fusha']:6.2f} | meanP={s['mean_p_aranjiya']:.4f}")


if __name__ == "__main__":
    main()
