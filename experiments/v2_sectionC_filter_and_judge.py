"""Section B.3 + Section C: Refined 7-rule filter + Claude Haiku 5.5 Batch API quality judge.

Modes:
  --mode filter    : Apply 7-rule filter to all OpusMT (3,605) and NLLB-1.3B (3,780) twins,
                     compute per-rule counts, 1:1 deduplicated per-split counts, and sample
                     20 KEPT + 20 REMOVED examples (10 per translator).
  --mode dryrun    : Run 20-item Anthropic Batch API dry run with claude-haiku-5-5; report exact
                     token usage, latency, and cost.
  --mode audit200  : Run 200-twin Anthropic Batch API audit (100 OpusMT + 100 NLLB-1.3B); compute
                     agreement with rule-based filter and extract 5+ disagreement examples.

All outputs saved under results/v2_sectionC_*.json. Never modifies v1 data or models.
"""
import argparse
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

# --------------------------------------------------------------------------- Filter thresholds
LABSE_MIN = 0.75
LEN_RATIO_MIN = 0.45   # ar_out words / en_pivot words
LEN_RATIO_MAX = 1.10
DIACRITIC_WORD_PCT_MAX = 0.30
UNKNOWN_WORD_PCT_MAX = 0.15

# Patterns
# Quote or punctuation mark followed by «أي» (allowing harakat on أي)
_GLOSS_AY = re.compile(r'["«»“”\'.,،:;؛\-–—\)\]]\s*أ[ًٌٍَُِّْ]*ي[ًٌٍَُِّْ]*(?=$|[\s،؛:.,!?؟"«»“”\)\]])')
# Short-vowel harakat + sukun + tanween damm/kasr (excludes everyday tanween-fath ً and shadda ّ)
_VOWEL_HARAKAT = re.compile(r"[\u064c-\u0650\u0652\u0670]")
# Strip everything except Arabic letters & whitespace (NO alef/hamza/ى folding)
_STRIP_NON_AR_RAW = re.compile(r"[^\u0621-\u064a\s]")

JUDGE_MODEL = "claude-haiku-5-5"
# Batch API pricing (50% off standard $0.10/1M in, $0.50/1M out)
BATCH_IN_PER_M = 0.05
BATCH_OUT_PER_M = 0.25

JUDGE_PROMPT_TEMPLATE = (
    "Here is an English text and its Arabic translation. "
    "Is the Arabic (a) faithful in meaning, and (b) real, grammatical Arabic without invented words or garbled names? "
    'Answer ONLY JSON {{"faithful": "yes" or "no", "well_formed": "yes" or "no", "reason": "short, max 15 words"}}.\n\n'
    "English:\n{en}\n\n"
    "Arabic:\n{ar}"
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_all_twins():
    # OpusMT from pairs_v1.jsonl + pairs_v1_labse_scores.json
    pairs_v1 = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v1.jsonl"))
    labse_v1 = json.load(open(os.path.join(ROOT, "results", "pairs_v1_labse_scores.json"), encoding="utf-8"))
    assert len(pairs_v1) == len(labse_v1)
    opusmt = []
    for p, lr in zip(pairs_v1, labse_v1):
        if p["backward"] == "opus_mt":
            rec = dict(p)
            rec["labse"] = round(float(lr["labse"]), 4)
            opusmt.append(rec)

    # NLLB-1.3B from pairs_nllb_13b.jsonl
    nllb13 = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_nllb_13b.jsonl"))
    return {"opus_mt": opusmt, "nllb_13b": nllb13}


def _rep_metrics(text: str):
    words = normalize(text).split()
    n = len(words)
    max_w5 = 0
    for i in range(n):
        win = words[i : i + 5]
        c = Counter(win).most_common(1)[0][1]
        if c > max_w5:
            max_w5 = c
    bigrams = list(zip(words, words[1:]))
    max_b10 = 0
    for i in range(len(bigrams)):
        win = bigrams[i : i + 10]
        c = Counter(win).most_common(1)[0][1]
        if c > max_b10:
            max_b10 = c
    max_tri = 0
    if n >= 9:
        trigrams = Counter(zip(words, words[1:], words[2:]))
        if trigrams:
            max_tri = trigrams.most_common(1)[0][1]
    uniq_bi = len(set(bigrams)) / len(bigrams) if len(bigrams) >= 10 else 1.0
    uniq_w = len(set(words)) / n if n >= 12 else 1.0
    return {
        "n": n,
        "max_w5": max_w5,
        "max_b10": max_b10,
        "max_tri": max_tri,
        "uniq_bi": uniq_bi,
        "uniq_w": uniq_w,
    }


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


def dense_diacritics_pct(raw_ar: str) -> float:
    # Split on whitespace and keep tokens that contain at least one Arabic letter
    raw_words = [w for w in raw_ar.split() if re.search(r"[\u0621-\u064a]", w)]
    if not raw_words:
        return 0.0
    vowelled = sum(1 for w in raw_words if _VOWEL_HARAKAT.search(w))
    return vowelled / len(raw_words)


def unknown_word_stats(raw_ar: str):
    """Run wordfreq on RAW output with ONLY diacritics & punctuation removed (NO hamza/alef/ى folding)."""
    s = strip_diacritics(raw_ar)
    s = _STRIP_NON_AR_RAW.sub(" ", s)
    words = [w for w in s.split() if len(w) >= 2]
    if not words:
        return 1.0, []
    unk = [w for w in words if word_frequency(w, "ar") == 0.0]
    return len(unk) / len(words), unk


def evaluate_twin(rec: dict, unk_thr: float = UNKNOWN_WORD_PCT_MAX):
    fusha = rec["fusha"]
    ar = rec["aranjiya"]
    pivot = rec["pivot"]
    labse = float(rec["labse"])

    flags = []

    # 1) gloss_pattern
    g_out = len(_GLOSS_AY.findall(ar))
    g_src = len(_GLOSS_AY.findall(fusha))
    if g_out - g_src >= 2:
        flags.append("gloss_pattern")

    # 2) ay_excess
    wf = normalize(fusha).split()
    wa = normalize(ar).split()
    ay_f = sum(1 for w in wf if w == "اي")
    ay_a = sum(1 for w in wa if w == "اي")
    if ay_a - ay_f >= 2:
        flags.append("ay_excess")

    # 3) labse_low_sim
    if labse < LABSE_MIN:
        flags.append("labse_low_sim")

    # 4) length_ratio_vs_en
    en_w = len(pivot.split())
    len_ratio = len(wa) / max(1, en_w)
    if not (LEN_RATIO_MIN <= len_ratio <= LEN_RATIO_MAX):
        flags.append("length_ratio_vs_en")

    # 5) new_repetition (only repetition NOT already present in source)
    if has_new_repetition(fusha, ar):
        flags.append("new_repetition")

    # 6) dense_diacritics (>= 30% of words carry harakat)
    diac_pct = dense_diacritics_pct(ar)
    if diac_pct >= DIACRITIC_WORD_PCT_MAX:
        flags.append("dense_diacritics")

    # 7) unknown_words
    unk_pct, unk_words = unknown_word_stats(ar)
    if unk_pct > unk_thr:
        flags.append("unknown_words")

    return {
        "flags": flags,
        "len_ratio_vs_en": round(len_ratio, 3),
        "diac_word_pct": round(diac_pct * 100.0, 1),
        "unk_word_pct": round(unk_pct * 100.0, 1),
        "unk_words_sample": unk_words[:8],
    }


def run_filter_mode():
    twins = load_all_twins()
    summary = {
        "thresholds": {
            "labse_min": LABSE_MIN,
            "len_ratio_vs_en": [LEN_RATIO_MIN, LEN_RATIO_MAX],
            "dense_diacritics_word_pct": DIACRITIC_WORD_PCT_MAX,
            "unknown_word_pct_max": UNKNOWN_WORD_PCT_MAX,
        },
        "per_translator": {},
        "sensitivity_unk_12pct": {},
    }

    annotated = {}
    for tr_name, recs in twins.items():
        rule_counts = Counter()
        kept, removed = [], []
        unk12_extra = 0
        for r in recs:
            ev = evaluate_twin(r)
            item = dict(r, **ev)
            if ev["flags"]:
                removed.append(item)
                for fl in ev["flags"]:
                    rule_counts[fl] += 1
            else:
                kept.append(item)
                if ev["unk_word_pct"] > 12.0:
                    unk12_extra += 1
        annotated[tr_name] = {"kept": kept, "removed": removed, "all": kept + removed}
        n = len(recs)
        summary["per_translator"][tr_name] = {
            "total": n,
            "kept": len(kept),
            "kept_pct": round(len(kept) / max(1, n) * 100.0, 2),
            "removed": len(removed),
            "removed_pct": round(len(removed) / max(1, n) * 100.0, 2),
            "mean_labse_all": round(float(np.mean([r["labse"] for r in recs])), 4),
            "mean_labse_kept": round(float(np.mean([r["labse"] for r in kept])), 4) if kept else 0.0,
            "rule_triggers": dict(rule_counts),
        }
        summary["sensitivity_unk_12pct"][tr_name] = {
            "additional_removed_if_unk_12pct": unk12_extra
        }

    # Deduplicate 1:1 per pair_id (keep ONE passing twin per classical passage, preferring higher LaBSE)
    by_pair = {}
    for tr_name in ["opus_mt", "nllb_13b"]:
        for item in annotated[tr_name]["kept"]:
            pid = item["pair_id"]
            if pid not in by_pair or item["labse"] > by_pair[pid]["labse"]:
                by_pair[pid] = item

    split_counts = Counter()
    split_by_tr = {"train": Counter(), "dev": Counter(), "test": Counter()}
    for pid, item in by_pair.items():
        sp = bucket(item.get("author") or pid, 0.1, 0.1)
        split_counts[sp] += 1
        split_by_tr[sp][item["backward"]] += 1

    # Also check how many total unique pair_ids exist across all 3,780 pivots by split
    all_pids = {}
    for tr_name in ["nllb_13b", "opus_mt"]:
        for r in twins[tr_name]:
            all_pids[r["pair_id"]] = r.get("author") or r["pair_id"]
    raw_splits = Counter(bucket(a, 0.1, 0.1) for a in all_pids.values())

    summary["deduplicated_1to1"] = {
        "total_unique_passages_available": len(all_pids),
        "raw_by_split": dict(raw_splits),
        "kept_1to1_total": len(by_pair),
        "kept_1to1_pct_of_3780": round(len(by_pair) / max(1, len(all_pids)) * 100.0, 2),
        "kept_1to1_by_split": dict(split_counts),
        "kept_1to1_by_split_and_translator": {sp: dict(c) for sp, c in split_by_tr.items()},
        "v1_comparison": {
            "v1_paired_train_pairs": 2894,
            "v2_filtered_1to1_train_pairs": split_counts["train"],
            "delta_train_pairs": split_counts["train"] - 2894,
        },
    }

    # Sample 20 random KEPT (10 per translator) and 20 random REMOVED (10 per translator)
    rng = random.Random(42)
    samples = {"kept": {}, "removed": {}}
    for tr_name in ["opus_mt", "nllb_13b"]:
        k_pool = list(annotated[tr_name]["kept"])
        r_pool = list(annotated[tr_name]["removed"])
        rng.shuffle(k_pool)
        rng.shuffle(r_pool)
        samples["kept"][tr_name] = [
            {
                "pair_id": x["pair_id"],
                "translator": tr_name,
                "labse": x["labse"],
                "len_ratio_vs_en": x["len_ratio_vs_en"],
                "unk_word_pct": x["unk_word_pct"],
                "diac_word_pct": x["diac_word_pct"],
                "flags": x["flags"],
                "fusha": x["fusha"],
                "pivot": x["pivot"],
                "aranjiya": x["aranjiya"],
            }
            for x in k_pool[:10]
        ]
        samples["removed"][tr_name] = [
            {
                "pair_id": x["pair_id"],
                "translator": tr_name,
                "labse": x["labse"],
                "len_ratio_vs_en": x["len_ratio_vs_en"],
                "unk_word_pct": x["unk_word_pct"],
                "unk_words_sample": x["unk_words_sample"],
                "diac_word_pct": x["diac_word_pct"],
                "flags": x["flags"],
                "fusha": x["fusha"],
                "pivot": x["pivot"],
                "aranjiya": x["aranjiya"],
            }
            for x in r_pool[:10]
        ]

    out_path = os.path.join(ROOT, "results", "v2_sectionC_filter.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": samples}, f, ensure_ascii=False, indent=2)

    print(f"Saved -> {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- Haiku Batch API Judge
def _get_anthropic_client():
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in environment")
    return anthropic.Anthropic(api_key=key)


def _parse_judge_json(text: str):
    """Robustly parse {faithful, well_formed, reason} JSON from Claude response."""
    t = text.strip()
    # Strip markdown fences if present
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return {"faithful": "parse_error", "well_formed": "parse_error", "reason": t[:120]}
    try:
        obj = json.loads(m.group(0))
        f_val = str(obj.get("faithful", "")).strip().lower()
        w_val = str(obj.get("well_formed", "")).strip().lower()
        if f_val in ("true", "yes"):
            f_val = "yes"
        elif f_val in ("false", "no"):
            f_val = "no"
        if w_val in ("true", "yes"):
            w_val = "yes"
        elif w_val in ("false", "no"):
            w_val = "no"
        return {
            "faithful": f_val,
            "well_formed": w_val,
            "reason": str(obj.get("reason", "")).strip(),
        }
    except Exception as e:
        mf = re.search(r'"faithful"\s*:\s*"?(yes|no|true|false)"?', t, flags=re.I)
        mw = re.search(r'"well_formed"\s*:\s*"?(yes|no|true|false)"?', t, flags=re.I)
        mr = re.search(r'"reason"\s*:\s*"(.*)', t, flags=re.S)
        if mf and mw:
            fv = "yes" if mf.group(1).lower() in ("yes", "true") else "no"
            wv = "yes" if mw.group(1).lower() in ("yes", "true") else "no"
            rv = mr.group(1).rstrip('}" \n\r') if mr else ""
            return {"faithful": fv, "well_formed": wv, "reason": rv}
        return {"faithful": "parse_error", "well_formed": "parse_error", "reason": f"{e}: {t[:100]}"}


def run_batch_judge(items: list, tag: str):
    """Submit a Batch API job for `items` (list of dicts with custom_id, pivot, aranjiya),
    poll until ended, and return parsed results + exact token usage & cost."""
    client = _get_anthropic_client()

    requests_payload = []
    for it in items:
        prompt = JUDGE_PROMPT_TEMPLATE.format(en=it["pivot"], ar=it["aranjiya"])
        requests_payload.append(
            {
                "custom_id": it["custom_id"],
                "params": {
                    "model": JUDGE_MODEL,
                    "max_tokens": 300,
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )

    t0 = time.time()
    batch = client.messages.batches.create(requests=requests_payload)
    batch_id = batch.id
    print(f"[{tag}] Created batch {batch_id} with {len(items)} requests (status={batch.processing_status})", file=sys.stderr)

    while batch.processing_status != "ended":
        time.sleep(5)
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        elapsed = time.time() - t0
        print(
            f"[{tag}] {batch_id} status={batch.processing_status} "
            f"proc={counts.processing} ok={counts.succeeded} err={counts.errored} ({elapsed:.0f}s)",
            file=sys.stderr,
        )

    elapsed = round(time.time() - t0, 1)
    results_map = {}
    tot_in = 0
    tot_out = 0
    errored = 0

    for entry in client.messages.batches.results(batch_id):
        cid = entry.custom_id
        if entry.result.type == "succeeded":
            msg = entry.result.message
            tot_in += msg.usage.input_tokens
            tot_out += msg.usage.output_tokens
            raw_txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            parsed = _parse_judge_json(raw_txt)
            results_map[cid] = {
                "status": "succeeded",
                "raw": raw_txt,
                "parsed": parsed,
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
            }
        else:
            errored += 1
            results_map[cid] = {"status": entry.result.type, "raw": str(entry.result), "parsed": None}

    cost_in = tot_in / 1_000_000 * BATCH_IN_PER_M
    cost_out = tot_out / 1_000_000 * BATCH_OUT_PER_M
    cost_total = cost_in + cost_out

    meta = {
        "batch_id": batch_id,
        "model": JUDGE_MODEL,
        "n_requests": len(items),
        "succeeded": len(items) - errored,
        "errored": errored,
        "elapsed_sec": elapsed,
        "total_input_tokens": tot_in,
        "total_output_tokens": tot_out,
        "avg_input_tokens": round(tot_in / max(1, len(items) - errored), 1),
        "avg_output_tokens": round(tot_out / max(1, len(items) - errored), 1),
        "batch_cost_usd": round(cost_total, 6),
        "batch_cost_per_1000_usd": round(cost_total / max(1, len(items)) * 1000.0, 4),
    }
    return meta, results_map


def _build_annotated_pool():
    twins = load_all_twins()
    pool = {}
    for tr_name, recs in twins.items():
        lst = []
        for idx, r in enumerate(recs):
            ev = evaluate_twin(r)
            item = dict(r, **ev)
            item["custom_id"] = f"{tr_name}_{idx}"
            item["rule_kept"] = len(ev["flags"]) == 0
            lst.append(item)
        pool[tr_name] = lst
    return pool


def run_dryrun_mode():
    pool = _build_annotated_pool()
    rng = random.Random(101)
    # 10 opus_mt (7 kept + 3 removed) + 10 nllb_13b (7 kept + 3 removed) = 20 items
    chosen = []
    for tr_name in ["opus_mt", "nllb_13b"]:
        kept = [x for x in pool[tr_name] if x["rule_kept"]]
        rem = [x for x in pool[tr_name] if not x["rule_kept"]]
        rng.shuffle(kept)
        rng.shuffle(rem)
        chosen.extend(kept[:7] + rem[:3])

    meta, res_map = run_batch_judge(chosen, "dryrun20")
    rows = []
    for it in chosen:
        jm = res_map[it["custom_id"]]
        rows.append(
            {
                "custom_id": it["custom_id"],
                "pair_id": it["pair_id"],
                "translator": it["backward"],
                "rule_kept": it["rule_kept"],
                "flags": it["flags"],
                "labse": it["labse"],
                "unk_word_pct": it["unk_word_pct"],
                "judge": jm["parsed"],
                "input_tokens": jm.get("input_tokens"),
                "output_tokens": jm.get("output_tokens"),
                "fusha": it["fusha"],
                "pivot": it["pivot"],
                "aranjiya": it["aranjiya"],
            }
        )

    out_path = os.path.join(ROOT, "results", "v2_sectionC_dryrun20.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"Saved -> {out_path}")
    print(json.dumps(meta, indent=2))
    wf_yes = sum(1 for r in rows if r["judge"] and r["judge"]["well_formed"] == "yes")
    f_yes = sum(1 for r in rows if r["judge"] and r["judge"]["faithful"] == "yes")
    parse_err = sum(1 for r in rows if not r["judge"] or r["judge"]["well_formed"] not in ("yes", "no"))
    print(f"Dry-run quick summary: well_formed=yes {wf_yes}/20, faithful=yes {f_yes}/20, parse_errors={parse_err}")


def run_audit200_mode():
    pool = _build_annotated_pool()
    rng = random.Random(202)
    # Unbiased random sample of 100 opus_mt + 100 nllb_13b (200 total)
    chosen = []
    for tr_name in ["opus_mt", "nllb_13b"]:
        arr = list(pool[tr_name])
        rng.shuffle(arr)
        chosen.extend(arr[:100])

    meta, res_map = run_batch_judge(chosen, "audit200")
    rows = []
    for it in chosen:
        jm = res_map[it["custom_id"]]
        rows.append(
            {
                "custom_id": it["custom_id"],
                "pair_id": it["pair_id"],
                "translator": it["backward"],
                "rule_kept": it["rule_kept"],
                "flags": it["flags"],
                "labse": it["labse"],
                "len_ratio_vs_en": it["len_ratio_vs_en"],
                "unk_word_pct": it["unk_word_pct"],
                "diac_word_pct": it["diac_word_pct"],
                "judge": jm["parsed"],
                "fusha": it["fusha"],
                "pivot": it["pivot"],
                "aranjiya": it["aranjiya"],
            }
        )

    # Compute confusion matrix & agreement (rule_kept vs well_formed == "yes") overall and per translator
    def summarize_rows(sub):
        valid = [r for r in sub if r["judge"] and r["judge"]["well_formed"] in ("yes", "no")]
        tp = sum(1 for r in valid if r["rule_kept"] and r["judge"]["well_formed"] == "yes")   # both keep
        tn = sum(1 for r in valid if (not r["rule_kept"]) and r["judge"]["well_formed"] == "no")  # both drop
        fp = sum(1 for r in valid if r["rule_kept"] and r["judge"]["well_formed"] == "no")    # rules kept, Haiku dropped
        fn = sum(1 for r in valid if (not r["rule_kept"]) and r["judge"]["well_formed"] == "yes") # rules dropped, Haiku kept
        wf_yes = sum(1 for r in valid if r["judge"]["well_formed"] == "yes")
        f_yes = sum(1 for r in valid if r["judge"]["faithful"] == "yes")
        unfaithful_but_wf = sum(
            1 for r in valid if r["judge"]["well_formed"] == "yes" and r["judge"]["faithful"] == "no"
        )
        n = len(valid)
        return {
            "n_valid": n,
            "agreement_pct": round((tp + tn) / max(1, n) * 100.0, 2),
            "both_keep": tp,
            "both_drop": tn,
            "rules_keep_haiku_drop": fp,
            "rules_drop_haiku_keep": fn,
            "haiku_well_formed_yes": wf_yes,
            "haiku_well_formed_yes_pct": round(wf_yes / max(1, n) * 100.0, 2),
            "haiku_faithful_yes": f_yes,
            "haiku_faithful_yes_pct": round(f_yes / max(1, n) * 100.0, 2),
            "unfaithful_but_well_formed": unfaithful_but_wf,
            "among_rule_kept_haiku_drops_pct": round(fp / max(1, tp + fp) * 100.0, 2),
            "among_rule_dropped_haiku_keeps_pct": round(fn / max(1, tn + fn) * 100.0, 2),
        }

    agreement = {
        "overall": summarize_rows(rows),
        "opus_mt": summarize_rows([r for r in rows if r["translator"] == "opus_mt"]),
        "nllb_13b": summarize_rows([r for r in rows if r["translator"] == "nllb_13b"]),
    }

    disagreements_rules_keep_haiku_drop = [
        r for r in rows if r["rule_kept"] and r["judge"] and r["judge"]["well_formed"] == "no"
    ]
    disagreements_rules_drop_haiku_keep = [
        r for r in rows if (not r["rule_kept"]) and r["judge"] and r["judge"]["well_formed"] == "yes"
    ]

    out_path = os.path.join(ROOT, "results", "v2_sectionC_audit200.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "agreement": agreement,
                "disagreements_rules_keep_haiku_drop": disagreements_rules_keep_haiku_drop,
                "disagreements_rules_drop_haiku_keep": disagreements_rules_drop_haiku_keep,
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved -> {out_path}")
    print("META:", json.dumps(meta, indent=2))
    print("AGREEMENT:", json.dumps(agreement, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["filter", "dryrun", "audit200"], required=True)
    args = ap.parse_args()
    if args.mode == "filter":
        run_filter_mode()
    elif args.mode == "dryrun":
        run_dryrun_mode()
    elif args.mode == "audit200":
        run_audit200_mode()


if __name__ == "__main__":
    main()
