"""Baseline: character + word n-grams with logistic regression.

Fast (minutes on a laptop CPU), and it gives you a number BERT has to beat.
It also tells you WHICH words push toward عرنجية: a good sanity check.
If the top features are names or topics instead of style words, the data
has a leak; fix the data before training anything bigger.

  python src/train_baseline.py --data data/processed --hard data/hard_test.jsonl --out models/baseline
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import full_report, load_hard, read_jsonl  # noqa: E402
from normalize import normalize  # noqa: E402


class Baseline:
    def __init__(self, C=4.0):
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                    max_features=400_000)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    token_pattern=r"\S+", max_features=200_000)
        self.clf = LogisticRegression(C=C, max_iter=3000)

    def _x(self, texts, fit=False):
        t = [normalize(x) for x in texts]
        if fit:
            return hstack([self.char.fit_transform(t), self.word.fit_transform(t)]).tocsr()
        return hstack([self.char.transform(t), self.word.transform(t)]).tocsr()

    def fit(self, texts, y):
        self.clf.fit(self._x(texts, fit=True), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._x(texts))[:, 1]

    def top_word_features(self, k=40):
        n_char = len(self.char.vocabulary_)
        coefs = self.clf.coef_[0][n_char:]
        vocab = np.array(self.word.get_feature_names_out())
        order = np.argsort(coefs)
        return {"aranjiya": [(vocab[i], round(float(coefs[i]), 2)) for i in order[::-1][:k]],
                "fusha": [(vocab[i], round(float(coefs[i]), 2)) for i in order[:k]]}

    def explain(self, text):
        """Per-word contribution (word unigrams only) — for highlighting in a demo."""
        n_char = len(self.char.vocabulary_)
        coefs = self.clf.coef_[0][n_char:]
        vocab = self.word.vocabulary_
        return [(w, round(float(coefs[vocab[w]]), 2) if w in vocab else 0.0) for w in normalize(text).split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--hard", default="data/hard_test.jsonl")
    ap.add_argument("--out", default="models/baseline")
    ap.add_argument("--C", type=float, default=4.0)
    args = ap.parse_args()

    train = read_jsonl(os.path.join(args.data, "train.jsonl"))
    model = Baseline(C=args.C).fit([r["text"] for r in train], [r["label"] for r in train])

    results = {}
    for sp in ["dev", "test", "test_holdout_system"]:
        p = os.path.join(args.data, f"{sp}.jsonl")
        if os.path.exists(p):
            rows = read_jsonl(p)
            if rows:
                results[sp] = full_report(sp, rows, model.predict_proba([r["text"] for r in rows]))
    if os.path.exists(args.hard):
        hard = load_hard(args.hard)
        probs = model.predict_proba([r["text"] for r in hard])
        results["hard"] = full_report("hard test", hard, probs)
        results["hard_rows"] = [{"id": r["id"], "label": r["label"], "p_aranjiya": round(float(p), 3),
                                 "text": r["text"]} for r, p in zip(hard, probs)]

    feats = model.top_word_features()
    print("\nTop عرنجية words:", " ".join(w for w, _ in feats["aranjiya"][:30]))
    print("Top فصحى words:  ", " ".join(w for w, _ in feats["fusha"][:30]))
    results["top_features"] = feats

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(model, os.path.join(args.out, "baseline.joblib"))
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
