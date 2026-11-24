"""v3 Step 2: Chunk Rafii (وحي القلم) & Taha Hussein (الأيام) like training data (15-60 words, NO re-cleaning)
and generate English pivots with claude-haiku-5-5 (Batch API, ar->en).

Samples 1,250 random chunks per author (2,500 total) so that after OpusMT + NLLB-1.3B + 8 rules +
calibrated Haiku judge + 1:1 dedup (~70-75% yield), each author reliably reaches >= 800 clean pairs
(then capped at 800 clean pairs per author, TRAIN only).

Outputs:
  data/raw/fusha_v3_modern_train.jsonl
  data/cache/translations.jsonl          (appends claude_haiku ar->en pivots)
  results/v3_step2_pivots_meta.json
"""
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from backtranslate import Cache  # noqa: E402
from openiti_extract import chunk_paragraph  # noqa: E402

MODEL = "claude-haiku-5-5"
BATCH_IN_PER_M = 0.05
BATCH_OUT_PER_M = 0.25

PIVOT_AR_EN_PROMPT = (
    "Translate this Arabic passage into clear, natural, modern English, "
    "the way a professional translator would for a general reader. "
    "Keep all meaning; do not add explanations. Return ONLY the English translation.\n\n"
    "{ar}"
)

SOURCES = [
    (
        os.path.join(os.environ.get("ARANJIYA_BOOKS_DIR", "data/external"), "rafii_wahy_alqalam_modern_clean.txt"),
        "modern_rafii",
        "Rafii.WahyAlQalam",
        1250,
    ),
    (
        os.path.join(os.environ.get("ARANJIYA_BOOKS_DIR", "data/external"), "taha_alayyam_clean.txt"),
        "modern_taha_hussein",
        "TahaHussein.AlAyyam",
        1250,
    ),
]


def chunk_clean_file(path: str, author: str, book: str, cap: int, rng: random.Random):
    """Chunk already-cleaned one-paragraph-per-line file using exact training chunker (15-60 words).
    Does NOT re-clean the text."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            for ch in chunk_paragraph(line, min_words=15, max_words=60):
                if len(ch.split()) >= 15:
                    chunks.append(ch)
    rng.shuffle(chunks)
    chosen = chunks[:cap]
    print(f"{book}: {len(chosen)}/{len(chunks)} chunks sampled", file=sys.stderr)
    return [
        {
            "id": f"v3mod-{author}-{i}",
            "text": ch,
            "author": author,
            "book": book,
            "genre": "modern_fusha",
            "origin": "v3_modern_fusha",
        }
        for i, ch in enumerate(chosen)
    ]


def main():
    import anthropic

    rng = random.Random(202603)
    rows = []
    for path, author, book, cap in SOURCES:
        rows.extend(chunk_clean_file(path, author, book, cap, rng))
    rng.shuffle(rows)

    out_path = os.path.join(ROOT, "data", "raw", "fusha_v3_modern_train.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} modern fusha chunks -> {out_path}", file=sys.stderr)

    cache = Cache(os.path.join(ROOT, "data", "cache", "translations.jsonl"))
    todo = [(i, r) for i, r in enumerate(rows) if cache.get(Cache.key("claude_haiku", "ar", "en", r["text"])) is None]
    print(f"v3 modern ar->en pivot cache hits: {len(rows)-len(todo)}/{len(rows)}, todo: {len(todo)}", file=sys.stderr)
    if not todo:
        return

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    reqs = [
        {
            "custom_id": f"v3piv_{i}",
            "params": {
                "model": MODEL,
                "max_tokens": 500,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": PIVOT_AR_EN_PROMPT.format(ar=r["text"])}],
            },
        }
        for i, r in todo
    ]
    batch = client.messages.batches.create(requests=reqs)
    bid = batch.id
    print(f"Submitted Batch v3_pivots: {bid} ({len(reqs)} reqs)", file=sys.stderr)

    t0 = time.time()
    while batch.processing_status != "ended":
        time.sleep(10)
        batch = client.messages.batches.retrieve(bid)
        c = batch.request_counts
        print(
            f"[v3_pivots] {bid} status={batch.processing_status} proc={c.processing} ok={c.succeeded} err={c.errored} ({time.time()-t0:.0f}s)",
            file=sys.stderr,
        )

    tot_in, tot_out, err, ok_n = 0, 0, 0, 0
    for entry in client.messages.batches.results(bid):
        idx = int(entry.custom_id.split("_")[1])
        r = rows[idx]
        if entry.result.type == "succeeded":
            msg = entry.result.message
            tot_in += msg.usage.input_tokens
            tot_out += msg.usage.output_tokens
            en = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
            if en:
                cache.put(Cache.key("claude_haiku", "ar", "en", r["text"]), en)
                ok_n += 1
        else:
            err += 1

    cost = tot_in / 1e6 * BATCH_IN_PER_M + tot_out / 1e6 * BATCH_OUT_PER_M
    meta = {
        "batch_id": bid,
        "n_todo": len(todo),
        "ok": ok_n,
        "errored": err,
        "in_tokens": tot_in,
        "out_tokens": tot_out,
        "cost_usd": round(cost, 4),
    }
    with open(os.path.join(ROOT, "results", "v3_step2_pivots_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("v3 modern pivots complete:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
