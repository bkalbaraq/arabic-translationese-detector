"""Evaluate CAMeLBERT (single or ensemble) or TF-IDF on the two published test sets:
  - data/calque_pairs.jsonl  (38 minimal pairs from حياتي and النظرات)
  - data/hard_test.jsonl     (83 curated sentences / 36 pairs)

Example:
  python scripts/eval_calque.py \
      --bert models/v3_camelbert/final models/v3_camelbert_s101_devloss/final models/v3_camelbert_s202_devloss/final
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from evaluate import load_hard  # noqa: E402
from predict import AranjiyaDetector  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", nargs="+", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--calque", default=os.path.join(ROOT, "data", "calque_pairs.jsonl"))
    ap.add_argument("--hard", default=os.path.join(ROOT, "data", "hard_test.jsonl"))
    args = ap.parse_args()

    det = AranjiyaDetector(bert_paths=args.bert, baseline_path=args.baseline)

    if os.path.exists(args.calque):
        cps = read_jsonl(args.calque)
        pc_ctx = det.predict_proba([r["calque_context"] for r in cps])
        pn_ctx = det.predict_proba([r["native_context"] for r in cps])
        pc_snt = det.predict_proba([r["calque_sentence"] for r in cps])
        pn_snt = det.predict_proba([r["native_sentence"] for r in cps])

        ctx_ok = int((pc_ctx > pn_ctx).sum())
        snt_ok = int((pc_snt > pn_snt).sum())
        print(f"calque_pairs contexts : {ctx_ok}/{len(cps)} ({ctx_ok / len(cps) * 100:.1f}%)")
        for d in sorted({r["direction"] for r in cps}):
            idx = [i for i, r in enumerate(cps) if r["direction"] == d]
            ok = int((pc_ctx[idx] > pn_ctx[idx]).sum())
            print(f"  - {d:18s}: {ok}/{len(idx)}")
        print(f"calque_pairs sentences: {snt_ok}/{len(cps)} ({snt_ok / len(cps) * 100:.1f}%)")

    if os.path.exists(args.hard):
        hard = load_hard(args.hard)
        probs = det.predict_proba([r["text"] for r in hard])
        labels = np.array([r["label"] for r in hard])
        acc = float(((probs >= 0.5).astype(int) == labels).mean() * 100.0)
        by_pid = {}
        for r, p in zip(hard, probs):
            if r.get("pair_id") is not None:
                by_pid.setdefault(r["pair_id"], {})[r["label"]] = float(p)
        pairs = [v for v in by_pid.values() if 0 in v and 1 in v]
        pw_ok = sum(1 for v in pairs if v[1] > v[0])
        print(f"hard_test             : {acc:.2f}% ({int(round(acc / 100 * len(hard)))}/{len(hard)}), "
              f"pairwise {pw_ok}/{len(pairs)} ({pw_ok / max(1, len(pairs)) * 100:.1f}%)")


if __name__ == "__main__":
    main()
