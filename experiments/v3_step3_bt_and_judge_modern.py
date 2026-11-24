"""v3 Step 3: Back-translate all 2,500 Rafii & Taha Hussein pivots with OpusMT (fp16) + NLLB-1.3B (fp16),
score with LaBSE, apply the exact same 8 rules + calibrated Haiku Batch judge, deduplicate 1:1
(preferring higher LaBSE), and cap at 800 clean pairs per author (TRAIN only).

Outputs:
  data/raw/pairs_v3_modern_raw.jsonl
  data/raw/pairs_v3_modern_clean.jsonl
  results/v3_step3_modern_report.json
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from backtranslate import Cache  # noqa: E402
from v2_pipeline_step1 import (  # noqa: E402
    BATCH_IN_PER_M,
    BATCH_OUT_PER_M,
    CALIBRATED_JUDGE_PROMPT,
    MODEL,
    evaluate_8_rules,
    parse_judge_json,
)
from v2_pipeline_step3_bt_and_judge_stepD import bt_nllb_13b, bt_opus_mt, read_jsonl, score_labse  # noqa: E402

CAP_PER_AUTHOR = 800


def main():
    import anthropic

    cache = Cache(os.path.join(ROOT, "data", "cache", "translations.jsonl"))
    rows = read_jsonl(os.path.join(ROOT, "data", "raw", "fusha_v3_modern_train.jsonl"))
    items = []
    for r in rows:
        piv = cache.get(Cache.key("claude_haiku", "ar", "en", r["text"]))
        if piv and piv.strip():
            items.append((r, piv))
    print(f"Loaded {len(items)}/{len(rows)} v3 modern chunks with cached claude_haiku pivots", file=sys.stderr)
    if not items:
        raise RuntimeError("No cached v3 modern pivots found!")

    # 1) Back-translate on GPU (reusing safe-VRAM fp16 helpers from Step 3)
    opus_outs = bt_opus_mt(items, cache)
    nllb_outs = bt_nllb_13b(items, cache)

    # 2) Score with LaBSE
    pivots = [piv for _, piv in items]
    sims_opus = score_labse(pivots, opus_outs)
    sims_nllb = score_labse(pivots, nllb_outs)

    raw_twins = {"opus_mt": [], "nllb_13b": []}
    raw_path = os.path.join(ROOT, "data", "raw", "pairs_v3_modern_raw.jsonl")
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
                    "origin": "v3_modern_fusha",
                }
                rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw_twins[tr_name].append(rec)
    print(f"Wrote raw v3 modern twins -> {raw_path}", file=sys.stderr)

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
                item["custom_id"] = f"v3m_{tr_name}_{idx}"
                rule_kept.append(item)
        rule_stats[tr_name] = {
            "total": len(recs),
            "rule_kept": kn,
            "rule_removed": len(recs) - kn,
            "rule_triggers": dict(rc),
        }
    print("v3 modern 8-rule filter:", json.dumps(rule_stats, indent=2), file=sys.stderr)

    # 4) Calibrated Haiku Batch judge
    todo = [it for it in rule_kept if it["judge_key"] not in jcache]
    print(f"v3 modern judge cache hits: {len(rule_kept)-len(todo)}/{len(rule_kept)}, todo: {len(todo)}", file=sys.stderr)
    judge_cost = {"n": 0, "cost_usd": 0.0}
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
        print(f"Submitted Batch v3_judge: {bid} ({len(reqs)} reqs)", file=sys.stderr)
        t0 = time.time()
        while batch.processing_status != "ended":
            time.sleep(10)
            batch = client.messages.batches.retrieve(bid)
            c = batch.request_counts
            print(
                f"[v3_judge] {bid} status={batch.processing_status} proc={c.processing} ok={c.succeeded} err={c.errored} ({time.time()-t0:.0f}s)",
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
        judge_cost = {
            "batch_id": bid,
            "n": len(todo),
            "errored": err,
            "in_tokens": tot_in,
            "out_tokens": tot_out,
            "cost_usd": round(cost, 4),
        }

    # 5) Aggregate, deduplicate 1:1 (prefer higher LaBSE), cap at 800 per author
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
        judge_stats[tr] = {
            "rule_kept_judged": n_rk,
            "haiku_well_formed_yes": len(kept_pool[tr]),
            "haiku_well_formed_yes_pct": round(len(kept_pool[tr]) / max(1, n_rk) * 100.0, 2),
            "haiku_dropped": len(dropped_pool[tr]),
            "haiku_dropped_pct": round(len(dropped_pool[tr]) / max(1, n_rk) * 100.0, 2),
        }

    by_pair = {}
    for tr in ["opus_mt", "nllb_13b"]:
        for it in kept_pool[tr]:
            pid = it["pair_id"]
            if pid not in by_pair or it["labse"] > by_pair[pid]["labse"]:
                by_pair[pid] = it

    by_author = defaultdict(list)
    for it in by_pair.values():
        by_author[it["author"]].append(it)

    rng = random.Random(2026)
    final_pairs = []
    author_counts = {}
    for auth, lst in sorted(by_author.items()):
        rng.shuffle(lst)
        capped = lst[:CAP_PER_AUTHOR]
        author_counts[auth] = {
            "available_clean_1to1": len(lst),
            "selected_capped": len(capped),
            "by_translator": dict(Counter(x["backward"] for x in capped)),
        }
        final_pairs.extend(capped)

    clean_path = os.path.join(ROOT, "data", "raw", "pairs_v3_modern_clean.jsonl")
    with open(clean_path, "w", encoding="utf-8") as cf:
        for it in final_pairs:
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
                "origin": "v3_modern_fusha",
            }, ensure_ascii=False) + "\n")

    # Audit samples
    audit = {"v3_modern_haiku_kept": [], "v3_modern_haiku_dropped": []}
    for tr in ["opus_mt", "nllb_13b"]:
        kp, dp = list(kept_pool[tr]), list(dropped_pool[tr])
        rng.shuffle(kp); rng.shuffle(dp)
        for x in kp[:3]:
            audit["v3_modern_haiku_kept"].append({
                "pair_id": x["pair_id"], "author": x["author"], "translator": tr,
                "labse": x["labse"], "judge": x["judge"], "aranjiya": x["aranjiya"],
            })
        for x in dp[:3]:
            audit["v3_modern_haiku_dropped"].append({
                "pair_id": x["pair_id"], "author": x["author"], "translator": tr,
                "labse": x["labse"], "judge": x["judge"], "aranjiya": x["aranjiya"],
            })

    rep = {
        "rule_8_stats": rule_stats,
        "calibrated_haiku_judge_stats": judge_stats,
        "author_clean_1to1_capped": author_counts,
        "total_v3_modern_pairs_selected": len(final_pairs),
        "judge_cost": judge_cost,
        "audit_samples": audit,
    }
    rep_path = os.path.join(ROOT, "results", "v3_step3_modern_report.json")
    with open(rep_path, "w", encoding="utf-8") as rf:
        json.dump(rep, rf, ensure_ascii=False, indent=2)

    print("\n=== v3 STEP 3 COMPLETE ===")
    print(json.dumps({k: v for k, v in rep.items() if k != "audit_samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
