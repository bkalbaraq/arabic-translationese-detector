"""Follow-up v1 Diagnostics (measurement only — does NOT overwrite v1 models).

1. OPUS ablation: TF-IDF baseline trained on paired-only train.jsonl vs v1 TF-IDF.
2. Voice check: rates per 10,000 normalized words of conversational/direct-address words
   across classical train, NLLB twins, OpusMT twins, Claude seed twins, OPUS (overall + by corpus),
   and النظرات.
3. The «أي» problem: count «أي» in each translator's output vs its source passage,
   plus 10 full examples where NLLB inserted «أي» absent from the source.
4. النظرات split: using 94061468.epub chapter boundaries to separate adapted/translated
   European stories from Manfaluti's own essays (scored with v1 TF-IDF, no-OPUS TF-IDF, and v1 CAMeLBERT).

Saves full JSON to results/v1_diagnostics.json.
"""
import json
import os
import re
import sys
import warnings
import zipfile
from collections import Counter

import joblib
import numpy as np
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from evaluate import full_report, load_hard  # noqa: E402
from normalize import light_clean, normalize, strip_diacritics  # noqa: E402
from openiti_extract import chunk_paragraph  # noqa: E402
from shortcut_check import BertScorer, load_modern_chunks, summarize  # noqa: E402
from train_baseline import Baseline  # noqa: E402


# --------------------------------------------------------------------------- helpers
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def voice_rates_per_10k(texts):
    """Rates per 10,000 normalized tokens for target voice/direct-address words."""
    # Note: normalize() folds أ/إ/آ -> ا, ى -> ي, ة -> ه.
    targets = {
        "أنا": ("انا",),
        "أنت/أنتم": ("انت", "انتم"),
        "أيها": ("ايها",),
        "هل": ("هل",),
        "لقد": ("لقد",),
        "الآن": ("الان",),
        "أي": ("اي",),
    }
    total_tokens = 0
    counts = Counter()
    for t in texts:
        toks = normalize(t).split()
        total_tokens += len(toks)
        counts.update(toks)
    rates = {}
    for label, keys in targets.items():
        c = sum(counts[k] for k in keys)
        rates[label] = round(c / max(1, total_tokens) * 10000.0, 2)
    return {"total_tokens": total_tokens, "chunks": len(texts), "rates_per_10k": rates}


def ay_count(text):
    return sum(1 for w in normalize(text).split() if w == "اي")


def analyze_ay(pairs, name):
    src_tokens, out_tokens = 0, 0
    src_ay, out_ay = 0, 0
    src_has, out_has, inserted_only = 0, 0, 0
    inserted_examples = []
    for p in pairs:
        sf, sa = p["fusha"], p["aranjiya"]
        tf, ta = normalize(sf).split(), normalize(sa).split()
        src_tokens += len(tf)
        out_tokens += len(ta)
        cf = sum(1 for w in tf if w == "اي")
        ca = sum(1 for w in ta if w == "اي")
        src_ay += cf
        out_ay += ca
        if cf > 0:
            src_has += 1
        if ca > 0:
            out_has += 1
        if cf == 0 and ca > 0:
            inserted_only += 1
            inserted_examples.append({
                "pair_id": p.get("pair_id"),
                "book": p.get("book"),
                "genre": p.get("genre"),
                "ay_inserted_count": ca,
                "fusha": sf,
                "pivot": p.get("pivot", ""),
                "aranjiya": sa,
            })
    # Sort examples so we show clear, varied insertions (mix of 1-insert and multi-insert)
    inserted_examples.sort(key=lambda x: (-x["ay_inserted_count"], len(x["fusha"])))
    return {
        "system": name,
        "n_pairs": len(pairs),
        "src_ay_total": src_ay,
        "out_ay_total": out_ay,
        "src_ay_per_10k": round(src_ay / max(1, src_tokens) * 10000.0, 2),
        "out_ay_per_10k": round(out_ay / max(1, out_tokens) * 10000.0, 2),
        "ratio_out_to_src": round(out_ay / max(1, src_ay), 2),
        "pairs_with_ay_in_src": src_has,
        "pairs_with_ay_in_out": out_has,
        "pairs_inserted_ay_absent_in_src": inserted_only,
        "pct_pairs_inserted_ay": round(inserted_only / max(1, len(pairs)) * 100.0, 2),
        "examples_inserted": inserted_examples,
    }


def map_nazarat_to_epub_chapters(clean_txt_path, epub_path):
    """Map every paragraph of manfaluti_nazarat_clean.txt to its exact chapter in 94061468.epub."""
    z = zipfile.ZipFile(epub_path)
    opf = BeautifulSoup(z.read("EPUB/package.opf"), "html.parser")
    manifest = {it["id"]: it["href"] for it in opf.find_all("item")}
    spine = [manifest[it["idref"]] for it in opf.find_all("itemref")]

    # Explicitly adapted/translated European stories & pieces in النظرات
    ADAPTED_TITLES = {
        "تأبين فولتير",        # Victor Hugo's eulogy of Voltaire (لابار، إميان، جان كالاس)
        "سحر البيان",          # Adaptation of Shakespeare's Julius Caesar speeches
        "الدعاء",              # Tagged: وهو ملخص قصيدة لفيكتور هوجو (بتصرف)
        "إيفون الصغيرة",       # Tagged: مترجمة
        "الموتى",              # Tagged: مترجمة (Gray's Elegy)
        "الانتقام",            # Tagged: مترجمة (المسيو كابريني، الكونت، باريس)
        "في أكواخ الفقراء",    # Tagged: مترجمة (ماري)
        "الشيخوخة المتمردة",   # European story (جورج، ماري، فرنسا)
    }

    # Build lookup from normalized paragraph prefix -> chapter info
    chapters = []
    para_lookup = {}
    for idx, href in enumerate(spine):
        path = "EPUB/" + href
        if path not in z.namelist():
            continue
        soup = BeautifulSoup(z.read(path), "html.parser")
        h = soup.find(["h1", "h2", "h3"])
        title = h.get_text(" ", strip=True) if h else href
        title_clean = strip_diacritics(title).strip()
        is_adapted = title_clean in {strip_diacritics(t) for t in ADAPTED_TITLES}
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        ch_info = {"spine_idx": idx, "title": title, "is_adapted": is_adapted}
        chapters.append(ch_info)
        for p in paras:
            np_txt = normalize(p)
            if len(np_txt) >= 20:
                para_lookup[np_txt[:60]] = ch_info
                para_lookup[np_txt[:35]] = ch_info

    # Walk manfaluti_nazarat_clean.txt sequentially, tracking current chapter when matched
    current_ch = {"spine_idx": -1, "title": "UNKNOWN", "is_adapted": False}
    adapted_chunks = []
    essay_chunks = []
    per_chapter_chunks = {}
    matched_paras, fallback_paras = 0, 0

    with open(clean_txt_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            np_line = normalize(line)
            ch = para_lookup.get(np_line[:60]) or para_lookup.get(np_line[:35])
            if ch is not None:
                current_ch = ch
                matched_paras += 1
            else:
                fallback_paras += 1
            clean = light_clean(strip_diacritics(line))
            if not clean:
                continue
            cks = chunk_paragraph(clean, 15, 60)
            per_chapter_chunks.setdefault(current_ch["title"], {"is_adapted": current_ch["is_adapted"], "chunks": []})
            per_chapter_chunks[current_ch["title"]]["chunks"].extend(cks)
            if current_ch["is_adapted"]:
                adapted_chunks.extend(cks)
            else:
                essay_chunks.extend(cks)

    return adapted_chunks, essay_chunks, per_chapter_chunks, matched_paras, fallback_paras


def main():
    # Load v1 models (read-only!)
    v1_tfidf = joblib.load(os.path.join(ROOT, "models", "baseline", "baseline.joblib"))
    v1_bert = BertScorer(os.path.join(ROOT, "models", "camelbert", "final"))

    # ----------------------------------------------------------------------- 1. OPUS ABLATION
    train_rows = read_jsonl(os.path.join(ROOT, "data", "processed", "train.jsonl"))
    paired_train = [r for r in train_rows if r.get("pair_id") is not None]
    no_opus_tfidf = Baseline(C=4.0).fit([r["text"] for r in paired_train], [r["label"] for r in paired_train])

    test_rows = read_jsonl(os.path.join(ROOT, "data", "processed", "test.jsonl"))
    holdout_rows = read_jsonl(os.path.join(ROOT, "data", "processed", "test_holdout_system.jsonl"))

    classical_test = [r["text"] for r in test_rows if r["label"] == 0]
    aranjiya_twins = [r["text"] for r in test_rows if r["label"] == 1 and r.get("pair_id") is not None]
    holdout_fusha = [r["text"] for r in holdout_rows if r["label"] == 0]
    holdout_aran = [r["text"] for r in holdout_rows if r["label"] == 1]

    dl = os.environ.get("ARANJIYA_BOOKS_DIR", "data/external")
    nazarat_chunks = load_modern_chunks(os.path.join(dl, "manfaluti_nazarat_clean.txt"))
    hayati_chunks = load_modern_chunks(os.path.join(dl, "ahmad_amin_hayati_clean.txt"))
    taj_chunks = load_modern_chunks(os.path.join(dl, "manfaluti_sabil_altaj_clean.txt"))

    ablation_sets = [
        ("classical_test_all", classical_test),
        ("aranjiya_twins_test", aranjiya_twins),
        ("holdout_system_fusha", holdout_fusha),
        ("holdout_system_opus_mt", holdout_aran),
        ("manfaluti_nazarat", nazarat_chunks),
        ("ahmad_amin_hayati", hayati_chunks),
        ("manfaluti_sabil_altaj", taj_chunks),
    ]

    ablation_table = {}
    for name, texts in ablation_sets:
        ablation_table[name] = {
            "v1_tfidf": summarize(v1_tfidf.predict_proba(texts)),
            "no_opus_tfidf": summarize(no_opus_tfidf.predict_proba(texts)),
        }

    # Also compute overall holdout + hard test metrics for no-OPUS vs v1 TF-IDF
    hard_rows = load_hard(os.path.join(ROOT, "data", "hard_test.jsonl"))
    ablation_extra_metrics = {
        "paired_train_rows": len(paired_train),
        "holdout_overall_v1": full_report("v1_tfidf_holdout", holdout_rows, v1_tfidf.predict_proba([r["text"] for r in holdout_rows])),
        "holdout_overall_no_opus": full_report("no_opus_tfidf_holdout", holdout_rows, no_opus_tfidf.predict_proba([r["text"] for r in holdout_rows])),
        "hard_test_no_opus": full_report("no_opus_tfidf_hard", hard_rows, no_opus_tfidf.predict_proba([r["text"] for r in hard_rows])),
        "no_opus_top_features": no_opus_tfidf.top_word_features(30),
    }

    # ----------------------------------------------------------------------- 2. VOICE CHECK
    pairs_v1 = read_jsonl(os.path.join(ROOT, "data", "raw", "pairs_v1.jsonl"))
    pairs_claude = read_jsonl(os.path.join(ROOT, "data", "seed", "pairs_claude_seed.jsonl"))
    opus_all = read_jsonl(os.path.join(ROOT, "data", "raw", "human_translated.jsonl"))

    nllb_pairs = [p for p in pairs_v1 if p["forward"] == "gemini" and p["backward"] == "nllb"]
    opusmt_pairs = [p for p in pairs_v1 if p["forward"] == "gemini" and p["backward"] == "opus_mt"]

    voice_sources = {
        "classical_fusha_train": [r["text"] for r in train_rows if r["source"] == "openiti"],
        "nllb_twins (gemini>nllb)": [p["aranjiya"] for p in nllb_pairs],
        "opus_mt_twins (gemini>opus_mt)": [p["aranjiya"] for p in opusmt_pairs],
        "claude_seed_twins (n=22)": [p["aranjiya"] for p in pairs_claude],
        "opus_all": [r["text"] for r in opus_all],
        "opus:TED2020": [r["text"] for r in opus_all if r["source"] == "opus:TED2020"],
        "opus:OpenSubtitles": [r["text"] for r in opus_all if r["source"] == "opus:OpenSubtitles"],
        "opus:News-Commentary": [r["text"] for r in opus_all if r["source"] == "opus:News-Commentary"],
        "opus:GlobalVoices": [r["text"] for r in opus_all if r["source"] == "opus:GlobalVoices"],
        "manfaluti_nazarat": nazarat_chunks,
    }
    voice_table = {k: voice_rates_per_10k(v) for k, v in voice_sources.items()}

    # ----------------------------------------------------------------------- 3. THE «أي» PROBLEM
    ay_nllb = analyze_ay(nllb_pairs, "gemini>nllb")
    ay_opusmt = analyze_ay(opusmt_pairs, "gemini>opus_mt")
    ay_claude = analyze_ay(pairs_claude, "claude-in-chat")

    # Pick 10 clean, varied NLLB «أي» insertion examples (5 with 2+ insertions, 5 with 1 insertion)
    ex_all = ay_nllb["examples_inserted"]
    multi = [e for e in ex_all if e["ay_inserted_count"] >= 2]
    single = [e for e in ex_all if e["ay_inserted_count"] == 1]
    nllb_ay_10 = (multi[:5] + single[:5])[:10]
    ay_summary = {
        "gemini>nllb": {k: v for k, v in ay_nllb.items() if k != "examples_inserted"},
        "gemini>opus_mt": {k: v for k, v in ay_opusmt.items() if k != "examples_inserted"},
        "claude-in-chat": {k: v for k, v in ay_claude.items() if k != "examples_inserted"},
        "nllb_10_inserted_examples": nllb_ay_10,
    }

    # ----------------------------------------------------------------------- 4. النظرات SPLIT
    adapted_chunks, essay_chunks, per_ch, matched_p, fallback_p = map_nazarat_to_epub_chapters(
        os.path.join(dl, "manfaluti_nazarat_clean.txt"),
        os.path.join(dl, "94061468.epub"),
    )

    nazarat_split = {
        "mapping_coverage": {"matched_paragraphs": matched_p, "carried_over_paragraphs": fallback_p},
        "adapted_european_chapters": sorted([t for t, d in per_ch.items() if d["is_adapted"]]),
        "adapted_european_stories": {
            "v1_tfidf": summarize(v1_tfidf.predict_proba(adapted_chunks)),
            "no_opus_tfidf": summarize(no_opus_tfidf.predict_proba(adapted_chunks)),
            "v1_camelbert": summarize(v1_bert.predict_proba(adapted_chunks)),
        },
        "manfaluti_own_essays": {
            "v1_tfidf": summarize(v1_tfidf.predict_proba(essay_chunks)),
            "no_opus_tfidf": summarize(no_opus_tfidf.predict_proba(essay_chunks)),
            "v1_camelbert": summarize(v1_bert.predict_proba(essay_chunks)),
        },
        "per_adapted_chapter": {},
    }
    for t, d in sorted(per_ch.items()):
        if d["is_adapted"] and d["chunks"]:
            nazarat_split["per_adapted_chapter"][t] = {
                "v1_tfidf": summarize(v1_tfidf.predict_proba(d["chunks"])),
                "no_opus_tfidf": summarize(no_opus_tfidf.predict_proba(d["chunks"])),
                "v1_camelbert": summarize(v1_bert.predict_proba(d["chunks"])),
            }

    out = {
        "diagnostic_1_opus_ablation": {
            "table": ablation_table,
            "extra_metrics": ablation_extra_metrics,
        },
        "diagnostic_2_voice_check": voice_table,
        "diagnostic_3_ay_problem": ay_summary,
        "diagnostic_4_nazarat_split": nazarat_split,
    }

    out_path = os.path.join(ROOT, "results", "v1_diagnostics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nSaved ->", out_path)
    print("\n=== 1. OPUS ABLATION (TF-IDF) ===")
    for k, v in ablation_table.items():
        a, b = v["v1_tfidf"], v["no_opus_tfidf"]
        print(f"{k:24s} | n={a['n']:4d} | v1: %fusha={a['pct_fusha']:6.2f} meanP={a['mean_p_aranjiya']:.4f} | no-OPUS: %fusha={b['pct_fusha']:6.2f} meanP={b['mean_p_aranjiya']:.4f}")
    print("no-OPUS top عرنجية:", " ".join(w for w, _ in ablation_extra_metrics["no_opus_top_features"]["aranjiya"][:25]))
    print("no-OPUS top فصحى:  ", " ".join(w for w, _ in ablation_extra_metrics["no_opus_top_features"]["fusha"][:25]))

    print("\n=== 2. VOICE RATES PER 10,000 WORDS ===")
    cols = ["أنا", "أنت/أنتم", "أيها", "هل", "لقد", "الآن", "أي"]
    print(f"{'source':32s} | {'tokens':8s} | " + " | ".join(f"{c:>8s}" for c in cols))
    for k, v in voice_table.items():
        r = v["rates_per_10k"]
        print(f"{k:32s} | {v['total_tokens']:8d} | " + " | ".join(f"{r[c]:8.2f}" for c in cols))

    print("\n=== 3. THE «أي» PROBLEM ===")
    for sys_name in ["gemini>nllb", "gemini>opus_mt", "claude-in-chat"]:
        s = ay_summary[sys_name]
        print(f"{sys_name:16s} | n={s['n_pairs']:4d} | src={s['src_ay_total']:4d} ({s['src_ay_per_10k']}/10k) | out={s['out_ay_total']:4d} ({s['out_ay_per_10k']}/10k) | ratio={s['ratio_out_to_src']}x | inserted_only={s['pairs_inserted_ay_absent_in_src']} ({s['pct_pairs_inserted_ay']}%)")

    print("\n=== 4. النظرات SPLIT ===")
    print("mapping:", nazarat_split["mapping_coverage"])
    for grp in ["adapted_european_stories", "manfaluti_own_essays"]:
        g = nazarat_split[grp]
        print(f"{grp:26s} | n={g['v1_tfidf']['n']:4d} | v1_tfidf={g['v1_tfidf']['pct_fusha']:6.2f}% ({g['v1_tfidf']['mean_p_aranjiya']:.4f}) | no_opus={g['no_opus_tfidf']['pct_fusha']:6.2f}% ({g['no_opus_tfidf']['mean_p_aranjiya']:.4f}) | v1_bert={g['v1_camelbert']['pct_fusha']:6.2f}% ({g['v1_camelbert']['mean_p_aranjiya']:.4f})")


if __name__ == "__main__":
    main()
