"""v2 Pipeline Step 3:
1. Load Step D conversational classical chunks (data/raw/fusha_stepD_conv.jsonl) + cached claude_haiku ar->en pivots.
2. Back-translate ALL pivots on GPU with:
   - OpusMT (Helsinki-NLP/opus-mt-tc-big-en-ar, length-sorted bs=32)
   - NLLB-1.3B (facebook/nllb-200-distilled-1.3B, length-sorted safe VRAM bs=16/8/4)
   All outputs cached in data/cache/translations.jsonl.
3. Score all Step D twins with LaBSE on GPU.
4. Apply 8-rule deterministic filter + calibrated Haiku Batch API judge (claude-haiku-5-5, thinking disabled).
5. Save:
   - data/raw/pairs_v2_stepD_raw.jsonl        (all raw OpusMT + NLLB-1.3B Step D twins with LaBSE)
   - data/raw/pairs_v2_stepD_clean.jsonl      (1:1 deduplicated Step D clean pairs passing 8 rules + Haiku well_formed=yes)
   - results/v2_step3_stepD_report.json       (counts, costs, audit samples)
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from backtranslate import Cache  # noqa: E402
from build_dataset import bucket  # noqa: E402
from normalize import strip_diacritics  # noqa: E402
from v2_pipeline_step1 import (  # noqa: E402
    BATCH_IN_PER_M,
    BATCH_OUT_PER_M,
    CALIBRATED_JUDGE_PROMPT,
    MODEL,
    evaluate_8_rules,
    parse_judge_json,
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def bt_opus_mt(items, cache):
    outs = [None] * len(items)
    todo_idx = []
    for i, (_, piv) in enumerate(items):
        v = cache.get(Cache.key("opus_mt", "en", "ar", piv))
        if v is not None:
            outs[i] = v
        else:
            todo_idx.append(i)
    print(f"Step D opus_mt cache hits: {len(items)-len(todo_idx)}/{len(items)}, todo: {len(todo_idx)}", file=sys.stderr)
    if not todo_idx:
        return outs

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "Helsinki-NLP/opus-mt-en-ar"
    tok = AutoTokenizer.from_pretrained(model_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()

    todo_sorted = sorted(todo_idx, key=lambda i: len(items[i][1]))
    done, t0, pos = 0, time.time(), 0
    while pos < len(todo_sorted):
        c_end = len(items[todo_sorted[min(pos + 15, len(todo_sorted) - 1)]][1])
        bs = 16 if c_end < 350 else 8
        bidx = todo_sorted[pos : pos + bs]
        pos += bs
        btxt = [items[i][1] for i in bidx]
        enc = tok(btxt, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=None, max_new_tokens=300, num_beams=4, no_repeat_ngram_size=4)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        del enc, gen
        for idx, tr in zip(bidx, dec):
            outs[idx] = tr
            if tr:
                cache.put(Cache.key("opus_mt", "en", "ar", items[idx][1]), tr)
        done += len(bidx)
        rate = done / max(1e-6, time.time() - t0)
        eta = (len(todo_sorted) - done) / max(1e-6, rate)
        print(f"\rStep D opus_mt: {done}/{len(todo_sorted)} ({rate:.1f}/s, ETA {eta/60:.1f}m)", end="", file=sys.stderr)
    print(file=sys.stderr)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outs


def bt_nllb_13b(items, cache):
    outs = [None] * len(items)
    todo_idx = []
    for i, (_, piv) in enumerate(items):
        v = cache.get(Cache.key("nllb_13b", "en", "ar", piv))
        if v is not None:
            outs[i] = v
        else:
            todo_idx.append(i)
    print(f"Step D nllb_13b cache hits: {len(items)-len(todo_idx)}/{len(items)}, todo: {len(todo_idx)}", file=sys.stderr)
    if not todo_idx:
        return outs

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "facebook/nllb-200-distilled-1.3B"
    tok = AutoTokenizer.from_pretrained(model_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()
    tok.src_lang = "eng_Latn"
    tgt_id = tok.convert_tokens_to_ids("arb_Arab")

    todo_sorted = sorted(todo_idx, key=lambda i: len(items[i][1]))
    done, t0, pos = 0, time.time(), 0
    while pos < len(todo_sorted):
        # Conservative VRAM sizing for 8 GB RTX 4060 (zero WDDM paging!)
        c_end12 = len(items[todo_sorted[min(pos + 11, len(todo_sorted) - 1)]][1])
        c_end8 = len(items[todo_sorted[min(pos + 7, len(todo_sorted) - 1)]][1])
        if c_end12 < 240:
            bs = 12
        elif c_end8 < 360:
            bs = 8
        else:
            bs = 4
        bidx = todo_sorted[pos : pos + bs]
        pos += bs
        btxt = [items[i][1] for i in bidx]
        enc = tok(btxt, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=tgt_id,
                max_length=None,
                max_new_tokens=300,
                num_beams=4,
                no_repeat_ngram_size=4,
            )
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        del enc, gen
        if bs <= 4 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        for idx, tr in zip(bidx, dec):
            outs[idx] = tr
            if tr:
                cache.put(Cache.key("nllb_13b", "en", "ar", items[idx][1]), tr)
        done += len(bidx)
        rate = done / max(1e-6, time.time() - t0)
        eta = (len(todo_sorted) - done) / max(1e-6, rate)
        print(f"\rStep D nllb_13b: {done}/{len(todo_sorted)} ({rate:.1f}/s, ETA {eta/60:.1f}m)", end="", file=sys.stderr)
    print(file=sys.stderr)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outs


def score_labse(pivots, arabs):
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labse = SentenceTransformer("sentence-transformers/LaBSE", device=dev)
    emb_en = labse.encode(pivots, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb_ar = labse.encode([strip_diacritics(a or "") for a in arabs], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sims = (emb_en * emb_ar).sum(axis=1).astype(float)
    del labse
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sims


def main():
    import anthropic

    cache = Cache(os.path.join(ROOT, "data", "cache", "translations.jsonl"))
    rows = read_jsonl(os.path.join(ROOT, "data", "raw", "fusha_stepD_conv.jsonl"))
    items = []
    for r in rows:
        piv = cache.get(Cache.key("claude_haiku", "ar", "en", r["text"]))
        if piv and piv.strip():
            items.append((r, piv))
    print(f"Loaded {len(items)}/{len(rows)} Step D chunks with cached claude_haiku English pivots", file=sys.stderr)
    if not items:
        raise RuntimeError("No cached Step D pivots found! Wait for Step 2 to finish.")

    # 1) Back-translate on GPU
    opus_outs = bt_opus_mt(items, cache)
    nllb_outs = bt_nllb_13b(items, cache)

    # 2) Score with LaBSE
    pivots = [piv for _, piv in items]
    sims_opus = score_labse(pivots, opus_outs)
    sims_nllb = score_labse(pivots, nllb_outs)

    raw_twins = {"opus_mt": [], "nllb_13b": []}
    raw_path = os.path.join(ROOT, "data", "raw", "pairs_v2_stepD_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as rf:
        for (r, piv), o_ar, o_sim, n_ar, n_sim in zip(items, opus_outs, sims_opus, nllb_outs, sims_nllb):
            for tr_name, ar, sim in [("opus_mt", o_ar, o_sim), ("nllb_13b", n_ar, n_sim)]:
                if not ar or not re.search(r"[ء-ي]", ar):
                    continue
                rec = {
                    "pair_id": r["id"],
                    "fusha": r["text"],
                    "aranjiya": ar,
                    "pivot": piv,
                    "pivot_lang": "en",
                    "forward": "claude_haiku",
                    "backward": tr_name,
                    "labse": round(float(sim), 4),
                    "book": r.get("book"),
                    "author": r.get("author"),
                    "genre": r.get("genre"),
                    "origin": "stepD_conv",
                }
                rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw_twins[tr_name].append(rec)
    print(f"Wrote raw Step D twins -> {raw_path} (opus_mt={len(raw_twins['opus_mt'])}, nllb_13b={len(raw_twins['nllb_13b'])})", file=sys.stderr)

    # 3) 8-rule filter
    jcache_path = os.path.join(ROOT, "data", "cache", "haiku_judge_v2.jsonl")
    jcache = {}
    if os.path.exists(jcache_path):
        for r in read_jsonl(jcache_path):
            jcache[r["k"]] = r["v"]

    rule_stats = {}
    rule_kept = []
    for tr_name, recs in raw_twins.items():
        rc = Counter()
        kn = 0
        for idx, r in enumerate(recs):
            flags, lr, dp, up = evaluate_8_rules(r)
            if flags:
                for fl in flags:
                    rc[fl] += 1
            else:
                kn += 1
                item = dict(r, len_ratio_vs_en=lr, diac_word_pct=dp, unk_word_pct=up)
                item["judge_key"] = f"{tr_name}|{r['pair_id']}"
                item["custom_id"] = f"sd_{tr_name}_{idx}"
                rule_kept.append(item)
        rule_stats[tr_name] = {
            "total": len(recs),
            "rule_kept": kn,
            "rule_removed": len(recs) - kn,
            "rule_triggers": dict(rc),
        }
    print("Step D 8-rule filter:", json.dumps(rule_stats, indent=2), file=sys.stderr)

    # 4) Calibrated Haiku Batch judge on rule-kept Step D twins
    todo = [it for it in rule_kept if it["judge_key"] not in jcache]
    print(f"Step D judge cache hits: {len(rule_kept)-len(todo)}/{len(rule_kept)}, todo: {len(todo)}", file=sys.stderr)
    judge_cost_meta = {"n": 0, "cost_usd": 0.0}
    if todo:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        reqs = [
            {
                "custom_id": it["custom_id"],
                "params": {
                    "model": MODEL,
                    "max_tokens": 300,
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": CALIBRATED_JUDGE_PROMPT.format(en=it["pivot"], ar=it["aranjiya"])}],
                },
            }
            for it in todo
        ]
        batch = client.messages.batches.create(requests=reqs)
        bid = batch.id
        print(f"Submitted Batch D (stepD_judge): {bid} ({len(reqs)} reqs)", file=sys.stderr)
        t0 = time.time()
        while batch.processing_status != "ended":
            time.sleep(10)
            batch = client.messages.batches.retrieve(bid)
            c = batch.request_counts
            print(
                f"[stepD_judge] {bid} status={batch.processing_status} proc={c.processing} ok={c.succeeded} err={c.errored} ({time.time()-t0:.0f}s)",
                file=sys.stderr,
            )
        cid_map = {it["custom_id"]: it for it in todo}
        tot_in, tot_out, err = 0, 0, 0
        with open(jcache_path, "a", encoding="utf-8") as jf:
            for entry in client.messages.batches.results(bid):
                it = cid_map[entry.custom_id]
                if entry.result.type == "succeeded":
                    msg = entry.result.message
                    tot_in += msg.usage.input_tokens
                    tot_out += msg.usage.output_tokens
                    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                    parsed = parse_judge_json(raw)
                else:
                    err += 1
                    parsed = {"faithful": "error", "well_formed": "error", "reason": str(entry.result)}
                jcache[it["judge_key"]] = parsed
                jf.write(json.dumps({"k": it["judge_key"], "v": parsed}, ensure_ascii=False) + "\n")
        cost = tot_in / 1e6 * BATCH_IN_PER_M + tot_out / 1e6 * BATCH_OUT_PER_M
        judge_cost_meta = {
            "batch_id": bid,
            "n": len(todo),
            "errored": err,
            "in_tokens": tot_in,
            "out_tokens": tot_out,
            "cost_usd": round(cost, 4),
        }

    # 5) Aggregate & 1:1 dedup
    kept_pool = {"opus_mt": [], "nllb_13b": []}
    dropped_pool = {"opus_mt": [], "nllb_13b": []}
    for it in rule_kept:
        j = jcache.get(it["judge_key"], {"faithful": "missing", "well_formed": "missing", "reason": ""})
        it["judge"] = j
        if j.get("well_formed") == "yes":
            kept_pool[it["backward"]].append(it)
        else:
            dropped_pool[it["backward"]].append(it)

    judge_stats = {}
    for tr in ["opus_mt", "nllb_13b"]:
        n_rk = len(kept_pool[tr]) + len(dropped_pool[tr])
        n_wf = len(kept_pool[tr])
        judge_stats[tr] = {
            "rule_kept_judged": n_rk,
            "haiku_well_formed_yes": n_wf,
            "haiku_well_formed_yes_pct": round(n_wf / max(1, n_rk) * 100.0, 2),
            "haiku_dropped": len(dropped_pool[tr]),
            "haiku_dropped_pct": round(len(dropped_pool[tr]) / max(1, n_rk) * 100.0, 2),
        }

    by_pair = {}
    for tr in ["opus_mt", "nllb_13b"]:
        for it in kept_pool[tr]:
            pid = it["pair_id"]
            if pid not in by_pair or it["labse"] > by_pair[pid]["labse"]:
                by_pair[pid] = it

    clean_path = os.path.join(ROOT, "data", "raw", "pairs_v2_stepD_clean.jsonl")
    sp_counts = Counter()
    sp_by_tr = {"train": Counter(), "dev": Counter(), "test": Counter()}
    with open(clean_path, "w", encoding="utf-8") as cf:
        for pid, it in by_pair.items():
            sp = bucket(it.get("author") or pid, 0.1, 0.1)
            sp_counts[sp] += 1
            sp_by_tr[sp][it["backward"]] += 1
            cf.write(json.dumps({
                "pair_id": it["pair_id"],
                "fusha": it["fusha"],
                "aranjiya": it["aranjiya"],
                "pivot": it["pivot"],
                "pivot_lang": "en",
                "forward": it["forward"],
                "backward": it["backward"],
                "labse": it["labse"],
                "book": it.get("book"),
                "author": it.get("author"),
                "genre": it.get("genre"),
                "origin": "stepD_conv",
            }, ensure_ascii=False) + "\n")

    rng = random.Random(88)
    audit_samples = {"stepD_haiku_kept": [], "stepD_haiku_dropped": []}
    for tr in ["opus_mt", "nllb_13b"]:
        kp, dp = list(kept_pool[tr]), list(dropped_pool[tr])
        rng.shuffle(kp); rng.shuffle(dp)
        for x in kp[:5]:
            audit_samples["stepD_haiku_kept"].append({
                "pair_id": x["pair_id"], "translator": tr, "labse": x["labse"],
                "judge": x["judge"], "aranjiya": x["aranjiya"], "pivot": x["pivot"],
            })
        for x in dp[:5]:
            audit_samples["stepD_haiku_dropped"].append({
                "pair_id": x["pair_id"], "translator": tr, "labse": x["labse"],
                "judge": x["judge"], "aranjiya": x["aranjiya"], "pivot": x["pivot"],
            })

    report = {
        "rule_8_stats": rule_stats,
        "calibrated_haiku_judge_stats": judge_stats,
        "dedup_1to1_stepD_clean": {
            "total": len(by_pair),
            "by_split": dict(sp_counts),
            "by_split_and_translator": {sp: dict(c) for sp, c in sp_by_tr.items()},
        },
        "judge_cost": judge_cost_meta,
        "audit_samples": audit_samples,
    }
    rep_path = os.path.join(ROOT, "results", "v2_step3_stepD_report.json")
    with open(rep_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)

    print("\n=== STEP 3 COMPLETE ===")
    print(json.dumps({k: v for k, v in report.items() if k != "audit_samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
