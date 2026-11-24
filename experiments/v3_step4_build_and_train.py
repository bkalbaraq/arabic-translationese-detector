"""v3 Step 4: Build processed_v3 (v2a + Rafii & Taha Hussein modern فصحى pairs in TRAIN only)
and train v3 TF-IDF + v3 CAMeLBERT with exact v2a settings and strict dev-only model selection.

dev.jsonl, test.jsonl, and test_holdout_haiku.jsonl are 100% IDENTICAL to v2a.
v1, v2a, v2b models and datasets remain completely untouched.
"""
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from spans import token_labels  # noqa: E402
from v2_pipeline_step4_build_and_train import read_jsonl, train_camelbert, train_tfidf, write_jsonl  # noqa: E402


def build_v3_dataset():
    rng = random.Random(11)
    dir_v2a = os.path.join(ROOT, "data", "processed_v2a")
    dir_v3 = os.path.join(ROOT, "data", "processed_v3")
    os.makedirs(dir_v3, exist_ok=True)

    train_rows = list(read_jsonl(os.path.join(dir_v2a, "train.jsonl")))
    modern_pairs = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v3_modern_clean.jsonl"))

    for r in modern_pairs:
        sys_name = f"{r['forward']}>{r['backward']}"
        train_rows.append({
            "text": r["fusha"],
            "label": 0,
            "source": f"modern_fusha:{r['author']}",
            "pair_id": r["pair_id"],
            "group": r["author"],
            "genre": r.get("genre", "modern_fusha"),
            "origin": "v3_modern_fusha",
        })
        toks, labs = token_labels(r["fusha"], r["aranjiya"])
        train_rows.append({
            "text": r["aranjiya"],
            "label": 1,
            "source": f"bt:{sys_name}",
            "pair_id": r["pair_id"],
            "group": r["author"],
            "genre": r.get("genre", "modern_fusha"),
            "origin": "v3_modern_fusha",
            "tokens": toks,
            "token_labels": labs,
        })

    rng.shuffle(train_rows)
    for i, r in enumerate(train_rows):
        r["id"] = f"train-{i}"
    write_jsonl(os.path.join(dir_v3, "train.jsonl"), train_rows)

    # Identical dev, test, holdout to v2a
    for fname in ["dev.jsonl", "test.jsonl", "test_holdout_haiku.jsonl"]:
        write_jsonl(os.path.join(dir_v3, fname), read_jsonl(os.path.join(dir_v2a, fname)))

    v2a_stats = json.load(open(os.path.join(dir_v2a, "stats.json"), encoding="utf-8"))
    v3_stats = dict(v2a_stats)
    v3_stats["train"] = {
        "rows": len(train_rows),
        "pairs": len(train_rows) // 2,
        "labels": dict(Counter(r["label"] for r in train_rows)),
        "sources": dict(Counter(r["source"] for r in train_rows)),
        "origins": dict(Counter(r.get("origin") for r in train_rows)),
        "authors": len({r["group"] for r in train_rows}),
    }
    with open(os.path.join(dir_v3, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(v3_stats, f, ensure_ascii=False, indent=2)

    print("=== BUILT v3 DATASET ===")
    print(json.dumps(v3_stats["train"], ensure_ascii=False, indent=2))


def main():
    build_v3_dataset()
    train_tfidf("v3")
    train_camelbert("v3", epochs=3, max_len=256)


if __name__ == "__main__":
    main()
