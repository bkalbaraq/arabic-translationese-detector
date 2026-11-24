"""v2 Pipeline Step 5 — Complete Section E Side-by-Side Evaluation: v1 vs v2a vs v2b.

Evaluates ALL 6 models (v1_tfidf, v2a_tfidf, v2b_tfidf, v1_bert, v2a_bert, v2b_bert) on the EXACT SAME test sets:
  1. Classical test + عرنجية twins (data/processed_v2a/test.jsonl): accuracy, % فصحى, % عرنجية, pairwise accuracy
  2. Haiku held-out translator (data/processed_v2a/test_holdout_haiku.jsonl): accuracy, % فصحى, % عرنجية, pairwise
  3. OPUS test per corpus (OpenSubtitles, TED2020, News-Commentary, GlobalVoices; strictly test-split rows!)
  4. Hard test (data/hard_test.jsonl): accuracy, pairwise, plus exact list of items that changed between models
  5. Modern Arabic test sets: النظرات essays, النظرات adapted chapters, في سبيل التاج, حياتي
  6. calque_pairs.jsonl (contexts primary + sentences, by direction and by book; CAMeLBERT by LOGITS; every wrong pair)
  7. Regression checks: masking test (on v2a/v2b/v1) + voice check (rates/10k words in v2a/v2b train & voice gap on النظرات)
  8. Top 25 TF-IDF features per side for v1, v2a, v2b

Saves full JSON -> results/v2_sectionE_eval.json
"""
import json
import os
import random
import re
import sys
from collections import Counter

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_dataset import bucket  # noqa: E402
from evaluate import load_hard  # noqa: E402
from normalize import normalize  # noqa: E402
from shortcut_check import load_modern_chunks, mask_classical_surface  # noqa: E402
from train_baseline import Baseline  # noqa: F401, E402
from v1_diagnostics import map_nazarat_to_epub_chapters  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class BertModelWrapper:
    def __init__(self, model_dir: str, max_len: int, fold_alef_ya: bool):
        self.model_dir = model_dir
        self.max_len = max_len
        self.fold_alef_ya = fold_alef_ya
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.dev).eval()

    def _prep(self, texts):
        return [normalize(t, keep_punct=False, fold_letters=False, fold_alef_ya=self.fold_alef_ya) for t in texts]

    def score_all(self, texts, bs=64):
        prepped = self._prep(texts)
        margins = []
        probs = []
        with torch.no_grad():
            for i in range(0, len(prepped), bs):
                enc = self.tok(
                    prepped[i : i + bs],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                ).to(self.dev)
                lg = self.model(**enc).logits.float().cpu().numpy()
                margins.extend((lg[:, 1] - lg[:, 0]).tolist())
                e = np.exp(lg - lg.max(-1, keepdims=True))
                probs.extend((e / e.sum(-1, keepdims=True))[:, 1].tolist())
        return np.array(probs, dtype=float), np.array(margins, dtype=float)

    def predict_proba(self, texts):
        p, _ = self.score_all(texts)
        return p

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize_binary(probs):
    probs = np.asarray(probs, dtype=float)
    return {
        "n": int(len(probs)),
        "pct_fusha": round(float((probs < 0.5).mean() * 100.0), 2),
        "pct_aranjiya": round(float((probs >= 0.5).mean() * 100.0), 2),
        "mean_p_aranjiya": round(float(probs.mean()), 4),
    }


def eval_paired_set(rows, probs, margins=None):
    probs = np.asarray(probs, dtype=float)
    labels = np.array([r["label"] for r in rows])
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == labels).mean() * 100.0)
    f_mask = labels == 0
    a_mask = labels == 1
    f_acc = float((preds[f_mask] == 0).mean() * 100.0)
    a_acc = float((preds[a_mask] == 1).mean() * 100.0)

    # Pairwise accuracy (use logit margin when available to avoid saturation ties)
    scores = np.asarray(margins, dtype=float) if margins is not None else probs
    by_pid = {}
    for r, sc in zip(rows, scores):
        pid = r.get("pair_id")
        if pid is not None:
            by_pid.setdefault(pid, {})[r["label"]] = float(sc)
    pairs = [v for v in by_pid.values() if 0 in v and 1 in v]
    pw_ok = sum(1 for v in pairs if v[1] > v[0])
    return {
        "n": int(len(rows)),
        "n_pairs": len(pairs),
        "acc": round(acc, 2),
        "classical_pct_fusha": round(f_acc, 2),
        "twin_pct_aranjiya": round(a_acc, 2),
        "pairwise_acc": round(pw_ok / max(1, len(pairs)) * 100.0, 2),
        "pairwise_ok": pw_ok,
    }


def eval_calque(cps, score_fn, is_bert: bool):
    def get_scores(texts):
        if is_bert:
            probs, margins = score_fn(texts)
            return margins, probs
        probs = score_fn(texts)
        return probs, probs

    sc_ctx, pc_ctx = get_scores([r["calque_context"] for r in cps])
    sn_ctx, pn_ctx = get_scores([r["native_context"] for r in cps])
    sc_snt, pc_snt = get_scores([r["calque_sentence"] for r in cps])
    sn_snt, pn_snt = get_scores([r["native_sentence"] for r in cps])

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

        wrongs = []
        for i, r in enumerate(cps):
            if sc[i] <= sn[i]:
                wrongs.append({
                    "id": r["id"],
                    "book": r["book"],
                    "direction": r["direction"],
                    "calque_types": r.get("calque_types"),
                    "score_calque": round(float(sc[i]), 4),
                    "score_native": round(float(sn[i]), 4),
                    "diff": round(float(sc[i] - sn[i]), 4),
                    "p_calque": round(float(pc[i]), 4),
                    "p_native": round(float(pn[i]), 4),
                    "calque_sentence": r["calque_sentence"],
                    "native_sentence": r["native_sentence"],
                })
        return {
            "overall": agg(),
            "by_direction": {d: agg([r["direction"] == d for r in cps]) for d in sorted({r["direction"] for r in cps})},
            "by_book": {b: agg([r["book"] == b for r in cps]) for b in sorted({r["book"] for r in cps})},
            "wrong_pairs": wrongs,
        }

    return {"contexts": pack(sc_ctx, sn_ctx, pc_ctx, pn_ctx), "sentences": pack(sc_snt, sn_snt, pc_snt, pn_snt)}


VOICE_WORDS = {
    "أنا": {"انا"},
    "أنت/أنتم": {"انت", "انتم", "انتما", "انتن"},
    "أيها": {"ايها", "ايتها"},
    "هل": {"هل"},
    "لقد": {"لقد"},
    "الآن": {"الان"},
    "أي": {"اي"},
}
CONV_TRIGGERS = {"انا", "انت", "انتم", "ايها", "ايتها", "هل"}


def voice_rates_per_10k(texts):
    toks = []
    for t in texts:
        toks.extend(normalize(t, keep_punct=False, fold_letters=True).split())
    tot = max(1, len(toks))
    c = Counter(toks)
    return {
        "total_words": tot,
        "rates_per_10k": {k: round(sum(c[w] for w in ws) / tot * 10000.0, 2) for k, ws in VOICE_WORDS.items()},
    }


def main():
    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")

    # Shared test sets (EXACT SAME for v1, v2a, v2b!)
    test_v2 = read_jsonl(os.path.join(ROOT, "data", "processed_v2a", "test.jsonl"))
    holdout_haiku = read_jsonl(os.path.join(ROOT, "data", "processed_v2a", "test_holdout_haiku.jsonl"))
    hard_rows = load_hard(os.path.join(ROOT, "data", "hard_test.jsonl"))
    cps = read_jsonl(os.path.join(ROOT, "data", "calque_pairs.jsonl"))

    # OPUS test per corpus (strictly test-split ids, unseen in v2a AND v2b!)
    human_all = read_jsonl(os.path.join(ROOT, "data", "raw", "human_translated.jsonl"))
    rng = random.Random(42)
    opus_test = {}
    for corpus in ["opus:OpenSubtitles", "opus:TED2020", "opus:News-Commentary", "opus:GlobalVoices"]:
        pool = [h["text"] for h in human_all if h["source"] == corpus and bucket(h["id"], 0.1, 0.1) == "test"]
        rng.shuffle(pool)
        opus_test[corpus] = pool[:1000]

    # Modern Arabic test sets
    adapted_chunks, essay_chunks, _, _, _ = map_nazarat_to_epub_chapters(
        os.path.join(dl, "manfaluti_nazarat_clean.txt"),
        os.path.join(dl, "94061468.epub"),
    )
    taj_chunks = load_modern_chunks(os.path.join(dl, "manfaluti_sabil_altaj_clean.txt"))
    hayati_chunks = load_modern_chunks(os.path.join(dl, "ahmad_amin_hayati_clean.txt"))

    # Split النظرات essays into WITH vs WITHOUT conversational-voice words for voice regression check
    essays_with_voice = []
    essays_no_voice = []
    for ch in essay_chunks:
        ws = set(normalize(ch, keep_punct=False, fold_letters=True).split())
        if ws & CONV_TRIGGERS:
            essays_with_voice.append(ch)
        else:
            essays_no_voice.append(ch)

    # Masking regression test on test_v2 (both classical فصحى and عرنجية twins)
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
            "sabil_altaj": len(taj_chunks),
            "hayati": len(hayati_chunks),
            "essays_with_voice": len(essays_with_voice),
            "essays_no_voice": len(essays_no_voice),
        },
        "models": {},
        "top25_tfidf_features": {},
        "voice_rates_train": {},
    }

    # Voice rates per 10k in v2a & v2b training sets
    for vname in ["v2a", "v2b"]:
        tr = read_jsonl(os.path.join(ROOT, "data", f"processed_{vname}", "train.jsonl"))
        results["voice_rates_train"][f"{vname}_fusha"] = voice_rates_per_10k([r["text"] for r in tr if r["label"] == 0])
        results["voice_rates_train"][f"{vname}_aranjiya"] = voice_rates_per_10k([r["text"] for r in tr if r["label"] == 1])

    hard_preds_by_model = {}

    def run_eval_for_model(mkey: str, score_fn, is_bert: bool):
        print(f"Evaluating {mkey}...", file=sys.stderr)
        def get_pm(texts):
            if is_bert:
                return score_fn(texts)
            p = score_fn(texts)
            return p, None

        p_tv2, m_tv2 = get_pm([r["text"] for r in test_v2])
        p_ho, m_ho = get_pm([r["text"] for r in holdout_haiku])
        p_hd, m_hd = get_pm([r["text"] for r in hard_rows])
        hard_preds_by_model[mkey] = [
            {"id": r["id"], "label": r["label"], "p": round(float(p), 4), "pred": int(p >= 0.5)}
            for r, p in zip(hard_rows, p_hd)
        ]

        opus_res = {}
        for corp, txts in opus_test.items():
            po, _ = get_pm(txts)
            opus_res[corp] = summarize_binary(po)

        p_ess, _ = get_pm(essay_chunks)
        p_adp, _ = get_pm(adapted_chunks)
        p_taj, _ = get_pm(taj_chunks)
        p_hay, _ = get_pm(hayati_chunks)

        # Voice regression gap on النظرات essays
        p_ev, _ = get_pm(essays_with_voice)
        p_env, _ = get_pm(essays_no_voice)

        # Masking regression on test_v2
        pf_o, _ = get_pm(test_fusha_orig)
        pf_m, _ = get_pm(test_fusha_mask)
        pt_o, _ = get_pm(test_twin_orig)
        pt_m, _ = get_pm(test_twin_mask)

        f_flips_f2a = int(((pf_o < 0.5) & (pf_m >= 0.5)).sum())
        t_flips_a2f = int(((pt_o >= 0.5) & (pt_m < 0.5)).sum())

        results["models"][mkey] = {
            "test_v2_pairs": eval_paired_set(test_v2, p_tv2, m_tv2),
            "holdout_haiku": eval_paired_set(holdout_haiku, p_ho, m_ho),
            "opus_test_per_corpus": opus_res,
            "hard_test": eval_paired_set(hard_rows, p_hd, m_hd),
            "modern_books": {
                "nazarat_essays": summarize_binary(p_ess),
                "nazarat_adapted": summarize_binary(p_adp),
                "sabil_altaj": summarize_binary(p_taj),
                "hayati": summarize_binary(p_hay),
            },
            "calque_pairs": eval_calque(cps, score_fn, is_bert),
            "regression_masking": {
                "classical_pct_fusha_orig": round(float((pf_o < 0.5).mean() * 100.0), 2),
                "classical_pct_fusha_masked": round(float((pf_m < 0.5).mean() * 100.0), 2),
                "classical_flips_f2a": f_flips_f2a,
                "classical_flip_rate_pct": round(f_flips_f2a / max(1, len(pf_o)) * 100.0, 2),
                "twins_pct_aranjiya_orig": round(float((pt_o >= 0.5).mean() * 100.0), 2),
                "twins_pct_aranjiya_masked": round(float((pt_m >= 0.5).mean() * 100.0), 2),
                "twins_flips_a2f": t_flips_a2f,
                "twins_flip_rate_pct": round(t_flips_a2f / max(1, len(pt_o)) * 100.0, 2),
            },
            "regression_voice_on_nazarat_essays": {
                "with_voice_words": summarize_binary(p_ev),
                "without_voice_words": summarize_binary(p_env),
                "pct_fusha_gap": round(float((p_env < 0.5).mean() * 100.0 - (p_ev < 0.5).mean() * 100.0), 2),
            },
        }

    # 1) TF-IDF models: v1, v2a, v2b
    for mkey, mpath in [
        ("v1_tfidf", os.path.join(ROOT, "models", "baseline", "baseline.joblib")),
        ("v2a_tfidf", os.path.join(ROOT, "models", "v2a_baseline", "baseline.joblib")),
        ("v2b_tfidf", os.path.join(ROOT, "models", "v2b_baseline", "baseline.joblib")),
    ]:
        mdl = joblib.load(mpath)
        run_eval_for_model(mkey, mdl.predict_proba, is_bert=False)
        results["top25_tfidf_features"][mkey] = mdl.top_word_features(k=25)

    # 2) CAMeLBERT models: v1 (max_len=512 eval or 128? Use 256 for fair comparison, fold_alef_ya=False as trained!),
    #    v2a (max_len=256, fold_alef_ya=True), v2b (max_len=256, fold_alef_ya=True)
    for mkey, mdir, mlen, fay in [
        ("v1_bert", os.path.join(ROOT, "models", "camelbert", "final"), 256, False),
        ("v2a_bert", os.path.join(ROOT, "models", "v2a_camelbert", "final"), 256, True),
        ("v2b_bert", os.path.join(ROOT, "models", "v2b_camelbert", "final"), 256, True),
    ]:
        bw = BertModelWrapper(mdir, max_len=mlen, fold_alef_ya=fay)
        run_eval_for_model(mkey, bw.score_all, is_bert=True)
        bw.close()

    # Hard test diffs across v1 / v2a / v2b (separately for TF-IDF and CAMeLBERT)
    def compute_hard_diffs(keys):
        diffs = []
        for idx, r in enumerate(hard_rows):
            preds = {k: hard_preds_by_model[k][idx]["pred"] for k in keys}
            if len(set(preds.values())) > 1:
                diffs.append({
                    "id": r["id"],
                    "label": r["label"],
                    "text": r["text"],
                    "scores": {k: hard_preds_by_model[k][idx]["p"] for k in keys},
                    "preds": preds,
                })
        return diffs

    results["hard_test_diffs"] = {
        "tfidf_v1_vs_v2a_vs_v2b": compute_hard_diffs(["v1_tfidf", "v2a_tfidf", "v2b_tfidf"]),
        "bert_v1_vs_v2a_vs_v2b": compute_hard_diffs(["v1_bert", "v2a_bert", "v2b_bert"]),
    }

    out_path = os.path.join(ROOT, "results", "v2_sectionE_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full Section E evaluation -> {out_path}")

    # Quick summary printout
    print("\n=== SUMMARY TABLE ===")
    for mk, d in results["models"].items():
        print(
            f"{mk:10s} | test={d['test_v2_pairs']['acc']:5.1f}% (pw {d['test_v2_pairs']['pairwise_acc']:5.1f}%) "
            f"| haiku={d['holdout_haiku']['acc']:5.1f}% (pw {d['holdout_haiku']['pairwise_acc']:5.1f}%) "
            f"| hard={d['hard_test']['acc']:5.1f}% (pw {d['hard_test']['pairwise_acc']:5.1f}%) "
            f"| naz_ess={d['modern_books']['nazarat_essays']['pct_fusha']:5.1f}% "
            f"| naz_adp={d['modern_books']['nazarat_adapted']['pct_fusha']:5.1f}% "
            f"| hayati={d['modern_books']['hayati']['pct_fusha']:5.1f}% "
            f"| taj={d['modern_books']['sabil_altaj']['pct_fusha']:5.1f}% "
            f"| calque_ctx={d['calque_pairs']['contexts']['overall']['correct']}/38"
        )


if __name__ == "__main__":
    main()
