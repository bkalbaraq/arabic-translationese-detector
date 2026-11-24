"""Score Arabic text with CAMeLBERT (single checkpoint or multi-seed ensemble)
and optionally highlight عرنجية-pushing words with the TF-IDF companion model.

Examples:
  # Single v3 CAMeLBERT + TF-IDF word highlights
  python src/predict.py \
      --bert models/v3_camelbert/final \
      --baseline models/v3_baseline/baseline.joblib \
      "لعب دورا مهما في فتح البلاد"

  # 3-seed ensemble (averages p(عرنجية) across checkpoints)
  python src/predict.py \
      --bert models/v3_camelbert/final models/v3_camelbert_s101_devloss/final models/v3_camelbert_s202_devloss/final \
      --baseline models/v3_baseline/baseline.joblib \
      "هذا الأمر يمثل تحديا كبيرا على مستوى المنطقة"
"""
import argparse
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize  # noqa: E402
from train_baseline import Baseline  # noqa: E402,F401  (needed for joblib unpickle)


class AranjiyaDetector:
    """Thin inference wrapper combining CAMeLBERT scoring and TF-IDF word highlights."""

    def __init__(
        self,
        bert_paths: Optional[Sequence[str]] = None,
        baseline_path: Optional[str] = None,
        max_len: int = 256,
        device: Optional[str] = None,
    ):
        self.max_len = max_len
        self.baseline = None
        self.bert_models = []
        self.device = device

        if baseline_path:
            import joblib
            self.baseline = joblib.load(baseline_path)

        if bert_paths:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            for path in bert_paths:
                tok = AutoTokenizer.from_pretrained(path)
                mdl = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
                self.bert_models.append((tok, mdl))

        if not self.bert_models and self.baseline is None:
            raise ValueError("Provide at least one of bert_paths or baseline_path.")

    def _bert_probs(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        import torch

        cleaned = [normalize(t, keep_punct=False, fold_letters=False, fold_alef_ya=True) for t in texts]
        all_probs = []
        with torch.no_grad():
            for tok, mdl in self.bert_models:
                probs = []
                for i in range(0, len(cleaned), batch_size):
                    enc = tok(
                        cleaned[i : i + batch_size],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_len,
                    ).to(self.device)
                    lg = mdl(**enc).logits.float().cpu().numpy()
                    e = np.exp(lg - lg.max(-1, keepdims=True))
                    probs.extend((e / e.sum(-1, keepdims=True))[:, 1].tolist())
                all_probs.append(np.array(probs, dtype=float))
        return np.mean(all_probs, axis=0)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.bert_models:
            return self._bert_probs(texts)
        return np.asarray(self.baseline.predict_proba(list(texts)), dtype=float)

    def explain_words(self, text: str) -> List[Tuple[str, float]]:
        """Map TF-IDF word unigram + bigram weights back onto the surface tokens."""
        raw_toks = text.split()
        if self.baseline is None:
            return [(w, 0.0) for w in raw_toks]
        n_char = len(self.baseline.char.vocabulary_)
        coefs = self.baseline.clf.coef_[0][n_char:]
        vocab = self.baseline.word.vocabulary_
        norm_toks = [normalize(w, keep_punct=False, fold_letters=True).split() for w in raw_toks]
        flat = [(i, p) for i, pieces in enumerate(norm_toks) for p in pieces]

        scores = [0.0] * len(raw_toks)
        for idx, piece in flat:
            if piece in vocab:
                scores[idx] += float(coefs[vocab[piece]])
        for (i1, p1), (i2, p2) in zip(flat, flat[1:]):
            bg = f"{p1} {p2}"
            if bg in vocab:
                half = 0.5 * float(coefs[vocab[bg]])
                scores[i1] += half
                scores[i2] += half
        return [(tok, round(sc, 3)) for tok, sc in zip(raw_toks, scores)]

    def score(self, text: str) -> dict:
        p_main = float(self.predict_proba([text])[0])
        res = {
            "p_aranjiya": round(p_main, 4),
            "label": "aranjiya" if p_main >= 0.5 else "fusha",
            "model": f"camelbert_x{len(self.bert_models)}" if self.bert_models else "tfidf",
        }
        if self.baseline is not None:
            res["p_tfidf"] = round(float(self.baseline.predict_proba([text])[0]), 4)
            res["word_weights"] = self.explain_words(text)
        return res


def _format_highlighted(word_weights: List[Tuple[str, float]]) -> str:
    pos = [w for _, w in word_weights if w > 0]
    top = max(pos) if pos else 0.0
    thr = max(0.5 * top, 0.35)
    return " ".join(f"[{tok}]" if top > 0 and wt >= thr else tok for tok, wt in word_weights)


def main():
    ap = argparse.ArgumentParser(description="Score Arabic text for عرنجية vs فصحى.")
    ap.add_argument("text", nargs="+", help="One or more Arabic strings to score.")
    ap.add_argument("--bert", nargs="+", default=None, help="One or more CAMeLBERT checkpoint directories.")
    ap.add_argument("--baseline", default=None, help="Path to TF-IDF baseline.joblib (for scoring or highlights).")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--json", action="store_true", help="Print JSON output.")
    args = ap.parse_args()

    det = AranjiyaDetector(bert_paths=args.bert, baseline_path=args.baseline, max_len=args.max_len)
    results = [det.score(t) for t in args.text]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for t, r in zip(args.text, results):
        pct = r["p_aranjiya"] * 100.0
        tag = "عرنجية" if r["label"] == "aranjiya" else "فصحى"
        if "word_weights" in r:
            body = _format_highlighted(r["word_weights"])
            extra = f" (tfidf {r['p_tfidf'] * 100:4.1f}%)" if det.bert_models else ""
        else:
            body = t
            extra = ""
        print(f"{tag:6s} {pct:5.1f}%{extra} | {body}")


if __name__ == "__main__":
    main()
