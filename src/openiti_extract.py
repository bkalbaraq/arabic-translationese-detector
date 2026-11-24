"""Extract clean classical-Arabic prose chunks from OpenITI repositories.

OpenITI (github.com/OpenITI) stores each book as an "mARkdown" text file inside
repos named by the author's death year in 25-year blocks (0275AH = died 251–275 AH).

What this script does, and why:
  * keeps one version per book (OpenITI often has 2–5 copies of the same book)
  * drops EDITOR sections and footnotes  -> those are MODERN Arabic by 20th-c. editors
  * drops verse (%~%)                     -> عرنجية data has no verse; keeping it teaches "poetry = fusha"
  * drops isnad-heavy paragraphs          -> "حدثنا فلان عن فلان" is a free giveaway
  * drops translators from Greek/Syriac   -> classical translationese, not real fusha
  * caps chunks per book and per genre    -> so one giant hadith collection can't dominate
  * cuts prose into chunks of similar length (default 15–60 words)

Usage:
  python src/openiti_extract.py --repos openiti/0275ah openiti/0325ah --out data/raw/fusha.jsonl
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from normalize import light_clean, normalize  # noqa: E402

MAX_DEATH_AH = 1250  # the book's cutoff: writers after this are affected by the translation era

# Translators (Greek/Syriac → Arabic) and other non-native-style sources.
EXCLUDE_AUTHOR_PATTERNS = [
    "Translator", "HunaynIbnIshaq", "IstifanIbnBasil", "ThayadurusIbnBasil",
    "IshaqIbnHunayn", "HubayshAcsam", "QustaIbnLuqa", "YahyaIbnCadi", "IbnNacima",
    "MattaIbnYunus", "AbuBishrMatta", "ThabitIbnQurra", "Anonymous",
    "Ustath", "IbnBitriq", "IbnZurca", "IbnSuwar", "IbnKhammar",
    "AbuCuthmanDimashqi", "IbnBakhtishu", "HajajIbnYusuf",
]

# Raw automated OCR dumps in OpenITI (noisy text + unstripped modern editor prefaces).
OCR_FILENAME_MARKERS = ("Kraken", "eScriptorium", ".OCR")

ISNAD_WORDS = re.compile(r"\b(?:حدثنا|حدثني|أخبرنا|أخبرني|أنبأنا|أنبأني|سمعت|عن|ثنا|نا|أنا|حدثه|أخبره|أخبرهم|حدثهم)\b")
RIJAL_MARK = re.compile(r"\b(?:روى عنه|وروى عنه|سمع منه|يروي عن|روى عن)\b")
EDITOR_CUE = re.compile(
    r"(?:المحقق|التحقيق|المخطوط(?:ة|ات)?|النسخة الخطية|الطبعة|دار النشر|الفقرة\s+ص|"
    r"نشرتنا|ميتافيزيق|القرن الهجري|القرن الميلادي|الدكتور\s+|المستشرق|إعادة صياغة المقدمة|"
    r"مضمون النص|وحدة النص|ترقيم الصفحات)"
)
EDITOR_HEADER = re.compile(r"(?:\|EDITOR\||مقدمة المحقق|تقديم المحقق|منهج التحقيق|دراسة عن المؤلف|ترجمة المؤلف)")
PAGE_MARK = re.compile(r"PageV\d+P\d+|ms\d+|\[ص:\s*\d+\]|\[\d+\]|@\w+@?|Milestone\d+")
TAG_MARK = re.compile(r"<[^>]+>|#~:\w+:|\$\w+\$|~~|^#\s*|\bAUTO\b|\b(?:span|div|br)\b|[&]|\b[A-Za-z]\b")
SENT_SPLIT = re.compile(r"(?<=[.!?؟:؛])\s+")
FOOTNOTE_START = re.compile(r"^\s*\(\d+\)")


def genre_bucket(subject: str) -> str:
    s = subject or ""
    if any(k in s for k in ["أدب", "بلاغة", "طرائف", "قصص", "أخلاق", "رقائق"]):
        return "adab"
    if any(k in s for k in ["تاريخ", "سير", "تراجم", "طبقات", "بلدان", "جغراف", "رحلات"]):
        return "history"
    if any(k in s for k in ["حديث", "سنن", "مسانيد", "أجزاء", "علل", "رجال"]):
        return "hadith"
    if any(k in s for k in ["فقه", "أصول"]):
        return "fiqh"
    if any(k in s for k in ["تفسير", "قرآن"]):
        return "tafsir"
    if any(k in s for k in ["لغة", "معاجم", "نحو", "غريب"]):
        return "language"
    return "other"


def pick_versions(repo_data_dir: str):
    """Return one text file per book: prefer .completed > .mARkdown > plain, then lowest araN.
    Skips raw automated OCR dumps (Kraken / eScriptorium), which contain heavy OCR noise and
    unstripped modern editor introductions."""
    books = defaultdict(list)
    for path in glob.glob(os.path.join(repo_data_dir, "*", "*", "*")):
        name = os.path.basename(path)
        if name.endswith((".yml", ".md")) or "-ara" not in name:
            continue
        if any(m in name for m in OCR_FILENAME_MARKERS):
            continue
        book_uri = os.path.basename(os.path.dirname(path))
        books[book_uri].append(path)

    def rank(p):
        n = os.path.basename(p)
        status = 0 if n.endswith(".completed") else 1 if n.endswith(".mARkdown") else 2
        m = re.search(r"-ara(\d+)", n)
        return (status, int(m.group(1)) if m else 9)

    return {uri: sorted(paths, key=rank)[0] for uri, paths in books.items()}


def parse_book(path: str):
    """Yield (meta, paragraphs) for a mARkdown file."""
    meta, paras = {}, []
    in_header, in_editor, in_footnotes = True, False, False
    current = []

    def flush():
        if current:
            paras.append(" ".join(current))
            current.clear()

    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n").lstrip("﻿")
            if in_header:
                if line.startswith("#META#") and "::" in line:
                    k, v = line[6:].split("::", 1)
                    meta[k.strip().split(".", 1)[-1]] = v.strip()
                if "#META#Header#End#" in line:
                    in_header = False
                continue
            if line.startswith("### "):
                flush()
                in_editor = bool(EDITOR_HEADER.search(line))
                in_footnotes = False
                continue
            if in_editor or line.startswith("ENDNOTES"):
                in_editor = True
                continue
            if re.match(r"^_{5,}\s*$", line.strip()):
                flush()
                in_footnotes = True  # until next paragraph start that isn't a footnote
                continue
            if line.startswith("# "):
                flush()
                body = line[2:]
                if in_footnotes and FOOTNOTE_START.match(body):
                    continue
                in_footnotes = False
                if FOOTNOTE_START.match(body):
                    continue
                current.append(body)
            elif line.startswith("~~"):
                if not in_footnotes:
                    current.append(line[2:])
            elif line.strip() == "" or line.startswith("PageV"):
                continue
            else:
                if in_footnotes:
                    continue
                if FOOTNOTE_START.match(line):
                    continue
                current.append(line)
    flush()
    return meta, paras


def _is_clean_prose(text: str) -> bool:
    words = text.split()
    if len(words) < 8:
        return False
    if EDITOR_CUE.search(text):
        return False
    if len(ISNAD_WORDS.findall(text)) / len(words) > 0.06:
        return False
    if RIJAL_MARK.search(text):
        return False
    # Dense genealogical / isnad name lists ("فلان بن فلان بن فلان...")
    bin_count = sum(1 for w in words if w in ("بن", "ابن", "أبو", "أبي"))
    if bin_count / len(words) > 0.18:
        return False
    # Mostly-Arabic check (skip tables, lists of numbers, leftovers)
    letters = sum(1 for c in text if "ء" <= c <= "ي")
    if letters / max(1, len(text.replace(" ", ""))) < 0.85:
        return False
    return True


def clean_paragraph(p: str):
    if "%~%" in p or "%" in p:      # verse
        return None
    p = PAGE_MARK.sub(" ", p)
    p = TAG_MARK.sub(" ", p)
    p = light_clean(p)
    # Strip stray leading punctuation left after removing tags/markers
    p = re.sub(r"^[\s:؛،.\-–—]+", "", p)
    if not _is_clean_prose(p):
        return None
    return p


def chunk_paragraph(p: str, min_words: int, max_words: int):
    sents = [s for s in SENT_SPLIT.split(p) if s.strip()]
    pieces = []
    for s in sents:  # split very long sentences at commas, then hard-cut
        w = s.split()
        if len(w) <= max_words:
            pieces.append(s)
            continue
        buf = []
        for part in re.split(r"(?<=[،,])\s+", s):
            if buf and len(" ".join(buf + [part]).split()) > max_words:
                pieces.append(" ".join(buf))
                buf = []
            buf.append(part)
        if buf:
            pieces.append(" ".join(buf))
    chunks, buf = [], []
    for piece in pieces:
        pw = piece.split()
        if len(pw) > max_words:  # still too long: hard cut
            for i in range(0, len(pw), max_words):
                chunks.append(" ".join(pw[i:i + max_words]))
            continue
        if buf and len(" ".join(buf + [piece]).split()) > max_words:
            chunks.append(" ".join(buf))
            buf = []
        buf.append(piece)
        if len(" ".join(buf).split()) >= min_words:
            chunks.append(" ".join(buf))
            buf = []
    if buf and len(" ".join(buf).split()) >= min_words:
        chunks.append(" ".join(buf))
    out = []
    for c in chunks:
        c = re.sub(r"^[\s:؛،.\-–—]+", "", c)
        if min_words <= len(c.split()) <= max_words and _is_clean_prose(c):
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True, help="paths to cloned OpenITI repos")
    ap.add_argument("--out", default="data/raw/fusha.jsonl")
    ap.add_argument("--min-words", type=int, default=15)
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--per-book", type=int, default=1500, help="max chunks kept per book")
    ap.add_argument("--per-genre", type=int, default=0,
                    help="max chunks per genre bucket overall (0 = no cap)")
    ap.add_argument("--skip-genres", nargs="*", default=[],
                    help="genre buckets to skip entirely, e.g. hadith fiqh")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stats = Counter()
    by_genre = defaultdict(list)
    seen = set()

    for repo in args.repos:
        data_dir = os.path.join(repo, "data")
        for uri, path in sorted(pick_versions(data_dir).items()):
            author_id = uri.split(".")[0]
            try:
                died = int(author_id[:4])
            except ValueError:
                continue
            if died > MAX_DEATH_AH or any(pat in author_id for pat in EXCLUDE_AUTHOR_PATTERNS):
                stats["books_excluded"] += 1
                continue
            meta, paras = parse_book(path)
            genre = genre_bucket(meta.get("BookSUBJ", ""))
            if genre in args.skip_genres:
                stats["books_skipped_genre"] += 1
                continue
            chunks = []
            for p in paras:
                cp = clean_paragraph(p)
                if cp:
                    chunks.extend(chunk_paragraph(cp, args.min_words, args.max_words))
            rng.shuffle(chunks)
            kept = 0
            for c in chunks:
                key = normalize(c)
                if key in seen:
                    continue
                seen.add(key)
                by_genre[genre].append({
                    "text": c, "label": "fusha", "source": "openiti",
                    "book": uri, "author": author_id, "died_ah": died, "genre": genre,
                    "title": meta.get("BookTITLE", ""),
                })
                kept += 1
                if kept >= args.per_book:
                    break
            stats["books_used"] += 1
            stats[f"genre_{genre}"] += kept

    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for genre, rows in by_genre.items():
            rng.shuffle(rows)
            if args.per_genre:
                rows = rows[: args.per_genre]
            for i, r in enumerate(rows):
                r["id"] = f"fusha-{genre}-{i}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    print(f"wrote {total} chunks -> {args.out}")


if __name__ == "__main__":
    main()
