"""v2 Pipeline Step 2:
1. Extract ~3,500 conversational classical فصحى chunks (adab, dialogue, maqamat, letters, sermons)
   from OpenITI (including newly cloned 0300AH for Ibn Qutayba's Uyun al-Akhbar & Mubarrad's Kamil),
   strictly author-capped (<= 450 per author) and excluding any passage already in the existing 3,780 pivots.
2. Save to data/raw/fusha_stepD_conv.jsonl.
3. Submit Anthropic Batch API (claude-haiku-5-5, thinking disabled) for ar->en English pivots ONLY
   (English is just a pivot; Arabic twins come from OpusMT + NLLB-1.3B in Step 3).
4. Cache all English pivots in data/cache/translations.jsonl under backend 'claude_haiku'.
"""
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from backtranslate import Cache  # noqa: E402
from build_dataset import bucket  # noqa: E402
from openiti_extract import chunk_paragraph, clean_paragraph, parse_book, pick_versions  # noqa: E402

MODEL = "claude-haiku-5-5"
BATCH_IN_PER_M = 0.05
BATCH_OUT_PER_M = 0.25
_STRAY_FILENAME = re.compile(r"\S+\.(?:png|jpg|jpeg|tif|tiff|pdf|html?)\b", re.I)

PIVOT_AR_EN_PROMPT = (
    "Translate this classical Arabic passage into clear, natural, modern English, "
    "the way a professional translator would for a general reader. "
    "Keep all meaning; do not add explanations. Return ONLY the English translation.\n\n"
    "{ar}"
)

# Author caps and target book substrings in data/raw/fusha.jsonl
FUSHA_TARGETS = {
    "0255Jahiz": (["0255Jahiz.BayanWaTabyin", "0255Jahiz.Bukhala", "0255Jahiz.Rasail"], 450),
    "0362AbuFarajIsbahani": (["0362AbuFarajIsbahani.Aghani"], 450),
    "0516IbnCaliHariri": (["0516IbnCaliHariri.Maqamat"], 350),
    "0398BadicZamanHamadhani": (["0398BadicZamanHamadhani.Maqamat"], 350),
    "0384MuhassinTanukhi": (["0384MuhassinTanukhi.FarajBacdaShidda", "0384MuhassinTanukhi.NishwarMuhadara"], 350),
    "0356AbuCaliQali": (["0356AbuCaliQali.AmaliFiLugha"], 300),
    "0436SharifMurtada": (["0436SharifMurtada.Rasail"], 250),
    "0505Ghazali": (["0505Ghazali.MajmucatRasail"], 200),
    "0795IbnRajabHanbali": (["0795IbnRajabHanbali.MajmucRasail"], 200),
}

# Books to extract directly from newly cloned openiti/0300AH
REPO_0300_TARGETS = {
    "0276IbnQutaybaDinawari.CuyunAkhbar": ("0276IbnQutaybaDinawari", "adab", 450),
    "0285IbnYazidMubarrad.Kamil": ("0285IbnYazidMubarrad", "adab", 350),
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_from_0300(rng):
    repo_data = os.path.join(ROOT, "openiti", "0300AH", "data")
    picked = pick_versions(repo_data)
    out = []
    for book_uri, (author, genre, cap) in REPO_0300_TARGETS.items():
        # Match exact or prefix (e.g. KamilFiLugha)
        match_uri = None
        for u in picked:
            if u == book_uri or u.startswith(book_uri):
                match_uri = u
                break
        if not match_uri:
            print(f"WARNING: {book_uri} not found in 0300AH", file=sys.stderr)
            continue
        path = picked[match_uri]
        meta, paras = parse_book(path)
        chunks = []
        for p in paras:
            cp = clean_paragraph(p)
            if not cp:
                continue
            for ch in chunk_paragraph(cp, min_words=15, max_words=60):
                ch = re.sub(r"\s+", " ", _STRAY_FILENAME.sub(" ", ch)).strip()
                if len(ch.split()) >= 15 and not re.search(r"[A-Za-z]{3,}", ch):
                    chunks.append(ch)
        rng.shuffle(chunks)
        chosen = chunks[:cap]
        print(f"0300AH extracted {len(chosen)}/{len(chunks)} chunks from {match_uri} (split={bucket(author, 0.1, 0.1)})", file=sys.stderr)
        for i, ch in enumerate(chosen):
            out.append({
                "id": f"stepD-{match_uri}-{i}",
                "text": ch,
                "book": match_uri,
                "author": author,
                "genre": genre,
                "origin": "stepD_conv",
            })
    return out


def extract_stepD_chunks():
    rng = random.Random(2026)

    # Exclude all pair_ids already in existing 3,780 pivots
    existing_pids = {r["pair_id"] for r in read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_nllb_13b.jsonl"))}
    print(f"Excluding {len(existing_pids)} existing pair_ids", file=sys.stderr)

    # Collect target books from data/raw/fusha.jsonl
    wanted_books = {b for books, _ in FUSHA_TARGETS.values() for b in books}
    by_book = defaultdict(list)
    with open(os.path.join(ROOT, "data", "raw", "fusha.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            b = r.get("book")
            if b in wanted_books and r["id"] not in existing_pids:
                txt = re.sub(r"\s+", " ", _STRAY_FILENAME.sub(" ", r["text"])).strip()
                if len(txt.split()) >= 15 and not re.search(r"[A-Za-z]{3,}", txt):
                    r["text"] = txt
                    r["origin"] = "stepD_conv"
                    by_book[b].append(r)

    chosen_all = []
    for author, (books, cap) in FUSHA_TARGETS.items():
        # Balance across multiple books of the same author
        per_book = cap // len(books)
        rem = cap - per_book * len(books)
        auth_rows = []
        for idx, b in enumerate(books):
            pool = by_book[b]
            rng.shuffle(pool)
            take = per_book + (1 if idx < rem else 0)
            auth_rows.extend(pool[:take])
        print(f"fusha.jsonl sampled {len(auth_rows)} chunks for {author} (split={bucket(author, 0.1, 0.1)})", file=sys.stderr)
        chosen_all.extend(auth_rows)

    # Add 0300AH books
    chosen_all.extend(extract_from_0300(rng))
    rng.shuffle(chosen_all)

    out_path = os.path.join(ROOT, "data", "raw", "fusha_stepD_conv.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in chosen_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sp_counts = Counter(bucket(r["author"], 0.1, 0.1) for r in chosen_all)
    print(f"Wrote {len(chosen_all)} Step D conversational chunks -> {out_path} | splits={dict(sp_counts)}", file=sys.stderr)
    return chosen_all


def run_pivots_batch(rows):
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)

    cache = Cache(os.path.join(ROOT, "data", "cache", "translations.jsonl"))
    todo = []
    for i, r in enumerate(rows):
        if cache.get(Cache.key("claude_haiku", "ar", "en", r["text"])) is None:
            todo.append((i, r))
    print(f"Step D ar->en pivot cache hits: {len(rows) - len(todo)}/{len(rows)}, todo: {len(todo)}", file=sys.stderr)
    if not todo:
        return {"n": 0, "cost_usd": 0.0}

    reqs = [
        {
            "custom_id": f"piv_{i}",
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
    print(f"Submitted Batch C (stepD_pivots): {bid} ({len(reqs)} reqs)", file=sys.stderr)

    t0 = time.time()
    while batch.processing_status != "ended":
        time.sleep(10)
        batch = client.messages.batches.retrieve(bid)
        c = batch.request_counts
        print(
            f"[stepD_pivots] {bid} status={batch.processing_status} proc={c.processing} ok={c.succeeded} err={c.errored} ({time.time()-t0:.0f}s)",
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
    print("Step D pivots complete:", json.dumps(meta, indent=2))
    with open(os.path.join(ROOT, "results", "v2_step2_pivots_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    rows = extract_stepD_chunks()
    run_pivots_batch(rows)


if __name__ == "__main__":
    main()
