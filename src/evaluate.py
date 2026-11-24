"""Shared evaluation: the numbers that go in the README / CV.

  * accuracy, F1, ROC-AUC on dev/test (test = authors never seen in training)
  * accuracy per source           -> catches "it only learned to spot Google Translate"
  * pairwise accuracy             -> for each pair, does the عرنجية side score higher
                                     than its own fusha original? (topic held constant)
  * hard test, per category       -> own_pairs, book_contrasts, translated_old_topic, ...
"""
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def basic(y, p, threshold=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    out = {"n": int(len(y)), "acc": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred, zero_division=0), 4)}
    if len(set(y.tolist())) == 2:
        out["auc"] = round(roc_auc_score(y, p), 4)
    return out


def by_key(rows, probs, key, threshold=0.5):
    groups = defaultdict(lambda: ([], []))
    for r, p in zip(rows, probs):
        k = r.get(key) or "none"
        groups[k][0].append(r["label"])
        groups[k][1].append(p)
    return {k: {"n": len(y), "acc": round(float(np.mean((np.array(p) >= threshold) == np.array(y))), 4)}
            for k, (y, p) in sorted(groups.items())}


def pairwise(rows, probs):
    """Share of pairs where score(عرنجية) > score(its fusha original)."""
    by_pair = defaultdict(lambda: {"f": [], "a": []})
    for r, p in zip(rows, probs):
        if r.get("pair_id"):
            by_pair[r["pair_id"]]["f" if r["label"] == 0 else "a"].append(p)
    wins = total = 0
    for d in by_pair.values():
        for fa in d["f"]:
            for ar in d["a"]:
                total += 1
                wins += ar > fa
    return {"pairs": total, "pairwise_acc": round(wins / total, 4) if total else None}


def full_report(name, rows, probs, threshold=0.5):
    hard = rows and "category" in rows[0]
    rep = {"overall": basic([r["label"] for r in rows], probs, threshold), **pairwise(rows, probs)}
    rep["by_category" if hard else "by_source"] = by_key(rows, probs, "category" if hard else "source", threshold)
    if not hard:
        rep["by_genre"] = by_key(rows, probs, "genre", threshold)
    print(f"\n===== {name} =====")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep


def load_hard(path):
    rows = read_jsonl(path)
    for r in rows:
        r["label"] = 1 if r["label"] in ("aranjiya", 1) else 0
    return rows
