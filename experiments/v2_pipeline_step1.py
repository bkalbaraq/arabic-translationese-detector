"""v2 Pipeline Step 1:
1. Apply 8-rule deterministic filter (7 rules + leftover_latin) to all existing OpusMT (3,605) & NLLB-1.3B (3,780) twins.
2. Submit & wait concurrently for two Anthropic Batch API jobs (claude-haiku-5-5, thinking disabled):
   - Batch A: Calibrated well-formedness judge on all 8-rule-kept existing twins (~6,000 requests).
   - Batch B: Held-out en->ar translations on ALL 444 test-split cached pivots (never used in training).
3. Save:
   - data/cache/haiku_judge_v2.jsonl          (persistent judge cache keyed by translator|pair_id)
   - data/raw/pairs_v2_existing_clean.jsonl   (1:1 deduplicated clean pairs passing 8 rules + Haiku well_formed=yes)
   - data/raw/pairs_v2_haiku_holdout.jsonl    (test-split held-out translator pairs: gemini>claude_haiku)
   - results/v2_step1_judge_report.json       (counts, costs, 10 random kept + 10 random dropped audit samples)
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
from wordfreq import word_frequency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from build_dataset import bucket  # noqa: E402
from normalize import normalize, strip_diacritics  # noqa: E402

# --------------------------------------------------------------------------- Constants & Rules
LABSE_MIN = 0.75
LEN_RATIO_MIN = 0.45
LEN_RATIO_MAX = 1.10
DIACRITIC_WORD_PCT_MAX = 0.30
UNKNOWN_WORD_PCT_MAX = 0.15

_GLOSS_AY = re.compile(r'["«»“”\'.,،:;؛\-–—\)\]]\s*أ[ًٌٍَُِّْ]*ي[ًٌٍَُِّْ]*(?=$|[\s،؛:.,!?؟"«»“”\)\]])')
_VOWEL_HARAKAT = re.compile(r"[\u064c-\u0650\u0652\u0670]")
_STRIP_NON_AR_RAW = re.compile(r"[^\u0621-\u064a\s]")
_LEFTOVER_LATIN = re.compile(r"[A-Za-z]{2,}")
_STRAY_FILENAME = re.compile(r"\S+\.(?:png|jpg|jpeg|tif|tiff|pdf|html?)\b", re.I)

MODEL = "claude-haiku-5-5"
BATCH_IN_PER_M = 0.05
BATCH_OUT_PER_M = 0.25

CALIBRATED_JUDGE_PROMPT = (
    "Here is an English text and its Arabic translation. "
    "Is the Arabic (a) faithful in meaning, and (b) real, grammatical Arabic without invented words or garbled names?\n"
    "Ignore minor missing hamzas, occasional case-ending slips (e.g. أبو after عن), or stiff/literal translationese "
    'phrasing (literalness is acceptable). Mark well_formed as "no" ONLY for invented/non-existent words, severely '
    "garbled names, leftover non-Arabic script, or broken/nonsensical syntax.\n"
    'Answer ONLY JSON {{"faithful": "yes" or "no", "well_formed": "yes" or "no", "reason": "short, max 15 words"}}.\n\n'
    "English:\n{en}\n\n"
    "Arabic:\n{ar}"
)

HOLDOUT_EN_AR_PROMPT = (
    "Translate this English passage into Arabic, the way a professional translator at a modern Arabic news site "
    "or publishing house would. Keep all meaning; do not add explanations. Return ONLY the Arabic translation.\n\n"
    "{en}"
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def clean_source_strays(text: str) -> str:
    return re.sub(r"\s+", " ", _STRAY_FILENAME.sub(" ", text)).strip()


def _rep_metrics(text: str):
    words = normalize(text).split()
    n = len(words)
    max_w5 = 0
    for i in range(n):
        c = Counter(words[i : i + 5]).most_common(1)[0][1]
        if c > max_w5:
            max_w5 = c
    bigrams = list(zip(words, words[1:]))
    max_b10 = 0
    for i in range(len(bigrams)):
        c = Counter(bigrams[i : i + 10]).most_common(1)[0][1]
        if c > max_b10:
            max_b10 = c
    max_tri = 0
    if n >= 9:
        trigrams = Counter(zip(words, words[1:], words[2:]))
        if trigrams:
            max_tri = trigrams.most_common(1)[0][1]
    uniq_bi = len(set(bigrams)) / len(bigrams) if len(bigrams) >= 10 else 1.0
    uniq_w = len(set(words)) / n if n >= 12 else 1.0
    return {"n": n, "max_w5": max_w5, "max_b10": max_b10, "max_tri": max_tri, "uniq_bi": uniq_bi, "uniq_w": uniq_w}


def has_new_repetition(fusha: str, aranjiya: str) -> bool:
    out = _rep_metrics(aranjiya)
    if out["n"] < 6:
        return False
    src = _rep_metrics(fusha)
    if out["max_w5"] >= 3 and src["max_w5"] < 3:
        return True
    if out["max_b10"] >= 3 and src["max_b10"] < 3:
        return True
    if out["max_tri"] >= 3 and src["max_tri"] < 3:
        return True
    if out["uniq_bi"] < 0.72 and src["uniq_bi"] >= 0.75 and (src["uniq_bi"] - out["uniq_bi"]) >= 0.08:
        return True
    if out["uniq_w"] < 0.50 and src["uniq_w"] >= 0.55 and (src["uniq_w"] - out["uniq_w"]) >= 0.08:
        return True
    return False


def evaluate_8_rules(rec: dict):
    fusha = clean_source_strays(rec["fusha"])
    ar = rec["aranjiya"]
    pivot = rec["pivot"]
    labse = float(rec["labse"])
    flags = []

    if len(_GLOSS_AY.findall(ar)) - len(_GLOSS_AY.findall(fusha)) >= 2:
        flags.append("gloss_pattern")

    wf = normalize(fusha).split()
    wa = normalize(ar).split()
    if sum(1 for w in wa if w == "اي") - sum(1 for w in wf if w == "اي") >= 2:
        flags.append("ay_excess")

    if labse < LABSE_MIN:
        flags.append("labse_low_sim")

    en_w = len(pivot.split())
    len_ratio = len(wa) / max(1, en_w)
    if not (LEN_RATIO_MIN <= len_ratio <= LEN_RATIO_MAX):
        flags.append("length_ratio_vs_en")

    if has_new_repetition(fusha, ar):
        flags.append("new_repetition")

    raw_words = [w for w in ar.split() if re.search(r"[\u0621-\u064a]", w)]
    diac_pct = sum(1 for w in raw_words if _VOWEL_HARAKAT.search(w)) / max(1, len(raw_words))
    if diac_pct >= DIACRITIC_WORD_PCT_MAX:
        flags.append("dense_diacritics")

    s_raw = _STRIP_NON_AR_RAW.sub(" ", strip_diacritics(ar))
    ws = [w for w in s_raw.split() if len(w) >= 2]
    unk_pct = (sum(1 for w in ws if word_frequency(w, "ar") == 0.0) / len(ws)) if ws else 1.0
    if unk_pct > UNKNOWN_WORD_PCT_MAX:
        flags.append("unknown_words")

    if _LEFTOVER_LATIN.search(ar):
        flags.append("leftover_latin")

    return flags, round(len_ratio, 3), round(diac_pct * 100.0, 1), round(unk_pct * 100.0, 1)


def parse_judge_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            fv = str(obj.get("faithful", "")).strip().lower()
            wv = str(obj.get("well_formed", "")).strip().lower()
            fv = "yes" if fv in ("yes", "true") else ("no" if fv in ("no", "false") else fv)
            wv = "yes" if wv in ("yes", "true") else ("no" if wv in ("no", "false") else wv)
            return {"faithful": fv, "well_formed": wv, "reason": str(obj.get("reason", "")).strip()}
        except Exception:
            pass
    mf = re.search(r'"faithful"\s*:\s*"?(yes|no|true|false)"?', t, flags=re.I)
    mw = re.search(r'"well_formed"\s*:\s*"?(yes|no|true|false)"?', t, flags=re.I)
    mr = re.search(r'"reason"\s*:\s*"(.*)', t, flags=re.S)
    if mf and mw:
        fv = "yes" if mf.group(1).lower() in ("yes", "true") else "no"
        wv = "yes" if mw.group(1).lower() in ("yes", "true") else "no"
        rv = mr.group(1).rstrip('}" \n\r') if mr else ""
        return {"faithful": fv, "well_formed": wv, "reason": rv}
    return {"faithful": "parse_error", "well_formed": "parse_error", "reason": t[:120]}


def load_existing_twins():
    pairs_v1 = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v1.jsonl"))
    labse_v1 = json.load(open(os.path.join(ROOT, "results", "pairs_v1_labse_scores.json"), encoding="utf-8"))
    opusmt = []
    for p, lr in zip(pairs_v1, labse_v1):
        if p["backward"] == "opus_mt":
            rec = dict(p)
            rec["fusha"] = clean_source_strays(rec["fusha"])
            rec["labse"] = round(float(lr["labse"]), 4)
            opusmt.append(rec)
    nllb13 = []
    for r in read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_nllb_13b.jsonl")):
        r["fusha"] = clean_source_strays(r["fusha"])
        nllb13.append(r)
    return {"opus_mt": opusmt, "nllb_13b": nllb13}


def main():
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)

    twins = load_existing_twins()

    # Load existing judge cache if resuming
    jcache_path = os.path.join(ROOT, "data", "cache", "haiku_judge_v2.jsonl")
    os.makedirs(os.path.dirname(jcache_path), exist_ok=True)
    jcache = {}
    if os.path.exists(jcache_path):
        for r in read_jsonl(jcache_path):
            jcache[r["k"]] = r["v"]

    # Step 1: 8-rule filter
    rule_stats = {}
    rule_kept_items = []
    for tr_name, recs in twins.items():
        rc = Counter()
        kept_n = 0
        for idx, r in enumerate(recs):
            flags, lr, dp, up = evaluate_8_rules(r)
            if flags:
                for fl in flags:
                    rc[fl] += 1
            else:
                kept_n += 1
                item = dict(r, len_ratio_vs_en=lr, diac_word_pct=dp, unk_word_pct=up)
                item["judge_key"] = f"{tr_name}|{r['pair_id']}"
                item["custom_id"] = f"j_{tr_name}_{idx}"
                rule_kept_items.append(item)
        rule_stats[tr_name] = {
            "total": len(recs),
            "rule_kept": kept_n,
            "rule_removed": len(recs) - kept_n,
            "rule_triggers": dict(rc),
        }
    print("8-rule filter:", json.dumps(rule_stats, indent=2), file=sys.stderr)

    # Prepare Batch A (judge) for uncached rule-kept items
    judge_todo = [it for it in rule_kept_items if it["judge_key"] not in jcache]
    print(f"Judge cache hits: {len(rule_kept_items) - len(judge_todo)}/{len(rule_kept_items)}, todo: {len(judge_todo)}", file=sys.stderr)

    # Prepare Batch B (Haiku held-out en->ar for all test-split unique passages)
    holdout_path = os.path.join(ROOT, "data", "raw", "pairs_v2_haiku_holdout.jsonl")
    existing_holdout = {}
    if os.path.exists(holdout_path):
        for r in read_jsonl(holdout_path):
            existing_holdout[r["pair_id"]] = r

    test_passages = {}
    for tr_name in ["nllb_13b", "opus_mt"]:
        for r in twins[tr_name]:
            sp = bucket(r.get("author") or r["pair_id"], 0.1, 0.1)
            if sp == "test" and r["pair_id"] not in test_passages:
                test_passages[r["pair_id"]] = r
    holdout_todo = [r for pid, r in test_passages.items() if pid not in existing_holdout]
    print(f"Haiku holdout test passages: total={len(test_passages)}, cached={len(existing_holdout)}, todo={len(holdout_todo)}", file=sys.stderr)

    # Submit batches concurrently
    active = {}
    if judge_todo:
        reqs_a = [
            {
                "custom_id": it["custom_id"],
                "params": {
                    "model": MODEL,
                    "max_tokens": 300,
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": CALIBRATED_JUDGE_PROMPT.format(en=it["pivot"], ar=it["aranjiya"])}],
                },
            }
            for it in judge_todo
        ]
        ba = client.messages.batches.create(requests=reqs_a)
        active["judge_existing"] = ba.id
        print(f"Submitted Batch A (judge_existing): {ba.id} ({len(reqs_a)} reqs)", file=sys.stderr)

    if holdout_todo:
        reqs_b = [
            {
                "custom_id": f"ho_{i}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 600,
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": HOLDOUT_EN_AR_PROMPT.format(en=r["pivot"])}],
                },
            }
            for i, r in enumerate(holdout_todo)
        ]
        bb = client.messages.batches.create(requests=reqs_b)
        active["haiku_holdout"] = bb.id
        print(f"Submitted Batch B (haiku_holdout): {bb.id} ({len(reqs_b)} reqs)", file=sys.stderr)

    # Poll all active batches until ended
    t0 = time.time()
    pending = dict(active)
    while pending:
        time.sleep(10)
        done_keys = []
        for name, bid in pending.items():
            b = client.messages.batches.retrieve(bid)
            c = b.request_counts
            print(
                f"[{name}] {bid} status={b.processing_status} proc={c.processing} ok={c.succeeded} err={c.errored} ({time.time()-t0:.0f}s)",
                file=sys.stderr,
            )
            if b.processing_status == "ended":
                done_keys.append(name)
        for k in done_keys:
            del pending[k]

    costs = {}

    # Collect Batch A results
    if "judge_existing" in active:
        cid_to_item = {it["custom_id"]: it for it in judge_todo}
        tot_in, tot_out, err = 0, 0, 0
        with open(jcache_path, "a", encoding="utf-8") as jf:
            for entry in client.messages.batches.results(active["judge_existing"]):
                it = cid_to_item[entry.custom_id]
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
        cost_a = tot_in / 1e6 * BATCH_IN_PER_M + tot_out / 1e6 * BATCH_OUT_PER_M
        costs["judge_existing"] = {
            "batch_id": active["judge_existing"],
            "n": len(judge_todo),
            "errored": err,
            "in_tokens": tot_in,
            "out_tokens": tot_out,
            "cost_usd": round(cost_a, 4),
        }

    # Collect Batch B results (Haiku held-out en->ar)
    if "haiku_holdout" in active:
        tot_in, tot_out, err = 0, 0, 0
        for entry in client.messages.batches.results(active["haiku_holdout"]):
            idx = int(entry.custom_id.split("_")[1])
            r = holdout_todo[idx]
            if entry.result.type == "succeeded":
                msg = entry.result.message
                tot_in += msg.usage.input_tokens
                tot_out += msg.usage.output_tokens
                ar = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
                if ar and re.search(r"[ء-ي]", ar):
                    existing_holdout[r["pair_id"]] = {
                        "pair_id": r["pair_id"],
                        "fusha": r["fusha"],
                        "aranjiya": ar,
                        "pivot": r["pivot"],
                        "pivot_lang": "en",
                        "forward": "gemini",
                        "backward": "claude_haiku",
                        "book": r.get("book"),
                        "author": r.get("author"),
                        "genre": r.get("genre"),
                    }
            else:
                err += 1
        with open(holdout_path, "w", encoding="utf-8") as hf:
            for rec in existing_holdout.values():
                hf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cost_b = tot_in / 1e6 * BATCH_IN_PER_M + tot_out / 1e6 * BATCH_OUT_PER_M
        costs["haiku_holdout"] = {
            "batch_id": active["haiku_holdout"],
            "n": len(holdout_todo),
            "written_total": len(existing_holdout),
            "errored": err,
            "in_tokens": tot_in,
            "out_tokens": tot_out,
            "cost_usd": round(cost_b, 4),
        }

    # Aggregate judge results & build 1:1 clean existing pairs
    judge_stats = {}
    kept_pool = {"opus_mt": [], "nllb_13b": []}
    dropped_pool = {"opus_mt": [], "nllb_13b": []}
    for it in rule_kept_items:
        j = jcache.get(it["judge_key"], {"faithful": "missing", "well_formed": "missing", "reason": ""})
        it["judge"] = j
        tr = it["backward"]
        if j.get("well_formed") == "yes":
            kept_pool[tr].append(it)
        else:
            dropped_pool[tr].append(it)

    for tr in ["opus_mt", "nllb_13b"]:
        n_rk = len(kept_pool[tr]) + len(dropped_pool[tr])
        n_wf = len(kept_pool[tr])
        n_f = sum(1 for x in kept_pool[tr] + dropped_pool[tr] if x["judge"].get("faithful") == "yes")
        n_pe = sum(1 for x in kept_pool[tr] + dropped_pool[tr] if x["judge"].get("well_formed") not in ("yes", "no"))
        judge_stats[tr] = {
            "rule_kept_judged": n_rk,
            "haiku_well_formed_yes": n_wf,
            "haiku_well_formed_yes_pct": round(n_wf / max(1, n_rk) * 100.0, 2),
            "haiku_dropped": len(dropped_pool[tr]),
            "haiku_dropped_pct": round(len(dropped_pool[tr]) / max(1, n_rk) * 100.0, 2),
            "haiku_faithful_yes_pct": round(n_f / max(1, n_rk) * 100.0, 2),
            "parse_or_api_errors": n_pe,
        }

    # Deduplicate 1:1 per pair_id (prefer higher LaBSE among kept twins)
    by_pair = {}
    for tr in ["opus_mt", "nllb_13b"]:
        for it in kept_pool[tr]:
            pid = it["pair_id"]
            if pid not in by_pair or it["labse"] > by_pair[pid]["labse"]:
                by_pair[pid] = it

    clean_out_path = os.path.join(ROOT, "data", "raw", "pairs_v2_existing_clean.jsonl")
    split_counts = Counter()
    split_by_tr = {"train": Counter(), "dev": Counter(), "test": Counter()}
    with open(clean_out_path, "w", encoding="utf-8") as cf:
        for pid, it in by_pair.items():
            sp = bucket(it.get("author") or pid, 0.1, 0.1)
            split_counts[sp] += 1
            split_by_tr[sp][it["backward"]] += 1
            out_rec = {
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
                "origin": "existing_3780",
            }
            cf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # Audit sample: 10 random kept + 10 random dropped (5 per translator)
    rng = random.Random(77)
    audit_samples = {"calibrated_haiku_kept": [], "calibrated_haiku_dropped": []}
    for tr in ["opus_mt", "nllb_13b"]:
        kp = list(kept_pool[tr])
        dp = list(dropped_pool[tr])
        rng.shuffle(kp)
        rng.shuffle(dp)
        for x in kp[:5]:
            audit_samples["calibrated_haiku_kept"].append({
                "pair_id": x["pair_id"], "translator": tr, "labse": x["labse"],
                "judge": x["judge"], "aranjiya": x["aranjiya"], "pivot": x["pivot"],
            })
        for x in dp[:5]:
            audit_samples["calibrated_haiku_dropped"].append({
                "pair_id": x["pair_id"], "translator": tr, "labse": x["labse"],
                "judge": x["judge"], "aranjiya": x["aranjiya"], "pivot": x["pivot"],
            })

    report = {
        "rule_8_stats": rule_stats,
        "calibrated_haiku_judge_stats": judge_stats,
        "dedup_1to1_existing_clean": {
            "total": len(by_pair),
            "by_split": dict(split_counts),
            "by_split_and_translator": {sp: dict(c) for sp, c in split_by_tr.items()},
        },
        "haiku_holdout_count": len(existing_holdout),
        "costs": costs,
        "audit_samples": audit_samples,
    }
    rep_path = os.path.join(ROOT, "results", "v2_step1_judge_report.json")
    with open(rep_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)

    print("\n=== STEP 1 COMPLETE ===")
    print(json.dumps({k: v for k, v in report.items() if k != "audit_samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
