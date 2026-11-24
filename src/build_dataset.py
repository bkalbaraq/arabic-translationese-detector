"""Combine pairs + extra fusha + human translations into train/dev/test.

The split is BY AUTHOR, not by row. Test authors never appear in training, and
both sides of a pair always land in the same split. Otherwise the model could
memorize a passage's content in train and recognize its back-translation in test.

  python src/build_dataset.py \
      --pairs data/raw/pairs_*.jsonl \
      --fusha data/raw/fusha.jsonl \
      --human data/raw/human_translated.jsonl --human-share 0.3 \
      --out-dir data/processed

Labels: 0 = فصحى, 1 = عرنجية
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from spans import token_labels  # noqa: E402


def bucket(key: str, dev_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000 / 10000
    if h < test_frac:
        return "test"
    if h < test_frac + dev_frac:
        return "dev"
    return "train"


def read_jsonl(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="pair files (globs ok)")
    ap.add_argument("--fusha", default=None, help="extra unpaired fusha (from openiti_extract.py)")
    ap.add_argument("--human", default=None, help="human-translated عرنجية (from opus_download.py)")
    ap.add_argument("--human-share", type=float, default=0.3,
                    help="max share of the عرنجية side that comes from human translations")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--holdout-system", default=None,
                    help="e.g. 'claude>claude': every pair from this translation system goes to a separate "
                         "test file (test_holdout_system.jsonl). Proves the model learned عرنجية in general, "
                         "not the fingerprint of the systems it was trained on.")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pair_files = sorted({p for g in args.pairs for p in glob.glob(g)})
    splits = {"train": [], "dev": [], "test": []}
    holdout = []
    used_fusha = set()
    n_pairs = Counter()

    # 1) pairs: one fusha row + one عرنجية row, same split (by author)
    seen_pair_sys = set()
    for r in read_jsonl(pair_files):
        system = f"{r['forward']}>{r['backward']}"
        if (r["pair_id"], system) in seen_pair_sys:
            continue
        seen_pair_sys.add((r["pair_id"], system))
        sp = bucket(r.get("author") or r["pair_id"], args.dev_frac, args.test_frac)
        if args.holdout_system and system == args.holdout_system:
            # only keep it if its author is a test author, so the content is unseen too
            if sp == "test":
                holdout.append({"text": r["fusha"], "label": 0, "source": "openiti", "pair_id": r["pair_id"],
                                "group": r.get("author"), "genre": r.get("genre")})
                holdout.append({"text": r["aranjiya"], "label": 1, "source": f"bt:{system}",
                                "pair_id": r["pair_id"], "group": r.get("author"), "genre": r.get("genre")})
            continue
        if r["pair_id"] not in used_fusha:  # fusha side once, even if back-translated by several systems
            used_fusha.add(r["pair_id"])
            splits[sp].append({"text": r["fusha"], "label": 0, "source": "openiti", "pair_id": r["pair_id"],
                               "group": r.get("author"), "genre": r.get("genre")})
        toks, labs = token_labels(r["fusha"], r["aranjiya"])
        splits[sp].append({"text": r["aranjiya"], "label": 1, "source": f"bt:{system}", "pair_id": r["pair_id"],
                           "group": r.get("author"), "genre": r.get("genre"),
                           "tokens": toks, "token_labels": labs})
        n_pairs[sp] += 1

    # 2) human translations (unpaired) + matching amount of unpaired fusha
    if args.human and os.path.exists(args.human):
        human = list(read_jsonl([args.human]))
        rng.shuffle(human)
        for sp in splits:
            n_bt = n_pairs[sp]
            cap = int(n_bt * args.human_share / max(1e-9, 1 - args.human_share)) if n_bt else 0
            chosen = [h for h in human if bucket(h["id"], args.dev_frac, args.test_frac) == sp][:cap]
            for h in chosen:
                splits[sp].append({"text": h["text"], "label": 1, "source": h["source"], "pair_id": None,
                                   "group": h["source"], "genre": h.get("genre")})
            n_pairs[sp + "_human"] = len(chosen)

    # 3) unpaired fusha to keep classes balanced
    if args.fusha and os.path.exists(args.fusha):
        pool = [r for r in read_jsonl([args.fusha]) if r["id"] not in used_fusha]
        rng.shuffle(pool)
        for sp in splits:
            need = sum(1 for x in splits[sp] if x["label"] == 1) - sum(1 for x in splits[sp] if x["label"] == 0)
            if need <= 0:
                continue
            for r in pool:
                if need <= 0:
                    break
                if bucket(r["author"], args.dev_frac, args.test_frac) != sp:
                    continue
                splits[sp].append({"text": r["text"], "label": 0, "source": "openiti", "pair_id": None,
                                   "group": r["author"], "genre": r.get("genre")})
                need -= 1

    os.makedirs(args.out_dir, exist_ok=True)
    report = {}
    for sp, rows in splits.items():
        rng.shuffle(rows)
        with open(os.path.join(args.out_dir, f"{sp}.jsonl"), "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                r["id"] = f"{sp}-{i}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[sp] = {"rows": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
                      "sources": dict(Counter(r["source"] for r in rows)),
                      "authors": len({r["group"] for r in rows})}
    if holdout:
        with open(os.path.join(args.out_dir, "test_holdout_system.jsonl"), "w", encoding="utf-8") as f:
            for i, r in enumerate(holdout):
                r["id"] = f"holdout-{i}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["holdout_system"] = {"rows": len(holdout), "system": args.holdout_system}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
