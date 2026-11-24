"""Download REAL human-translated Arabic (عرنجية written by people) from OPUS.

The back-translation pairs are the backbone, but they are all machine-made.
These corpora add human translators' habits:

  OpenSubtitles  film/TV subtitles translated into Arabic  («اذهب إلى الجحيم»)
  UNPC           UN documents translated from English/French (formal, institutional)
  TED2020        talks translated by volunteers
  NewsCommentary opinion essays translated from English
  GlobalVoices   citizen-media articles translated from English

Only the Arabic side is kept. Short lines (subtitles) are merged with their
neighbours into chunks of the same length as the fusha chunks (15–60 words).

  python src/opus_download.py --out data/raw/human_translated.jsonl --per-corpus 20000

Note: this data is NOT paired with fusha, so topic differs (modern subjects).
Use it as a minority of the عرنجية side, and report results with and without it.
"""
import argparse
import io
import json
import os
import random
import re
import sys
import zipfile
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from normalize import light_clean, normalize  # noqa: E402

API = "https://opus.nlpl.eu/opusapi/?source=ar&target=en&corpus={c}&preprocessing=moses&version=latest"
# UNPC is excellent but very large (several GB); add it with --corpora ... UNPC when you have the disk.
DEFAULT_CORPORA = ["TED2020", "News-Commentary", "GlobalVoices", "OpenSubtitles"]


def corpus_url(corpus):
    r = requests.get(API.format(c=corpus), timeout=60)
    r.raise_for_status()
    data = r.json()
    files = [x for x in data.get("corpora", []) if x["url"].endswith(".zip")]
    if not files:
        raise RuntimeError(f"no moses zip for {corpus}")
    return files[0]["url"], files[0].get("size")


def download(url, dest):
    if os.path.exists(dest):
        return dest
    print("downloading", url, file=sys.stderr)
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp, dest)
    return dest


def arabic_lines(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".ar"))
        with z.open(name) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                yield line.strip()


def merge_into_chunks(lines, min_words, max_words):
    buf = []
    for line in lines:
        line = re.sub(r"^\s*-\s*", "", line)  # subtitle dialogue dashes
        line = light_clean(line)
        if not line:
            continue
        buf.append(line)
        n = len(" ".join(buf).split())
        if n >= min_words:
            if n <= max_words:
                yield " ".join(buf)
            buf = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/human_translated.jsonl")
    ap.add_argument("--corpora", nargs="+", default=DEFAULT_CORPORA)
    ap.add_argument("--per-corpus", type=int, default=20000)
    ap.add_argument("--max-lines", type=int, default=3_000_000,
                    help="read at most this many lines per corpus (big corpora are GBs)")
    ap.add_argument("--min-words", type=int, default=15)
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--dl-dir", default="data/opus")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.dl_dir, exist_ok=True)
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as out:
        for c in args.corpora:
            try:
                url, size = corpus_url(c)
            except Exception as e:
                print(f"skip {c}: {e}", file=sys.stderr)
                continue
            print(f"{c}: {url} ({size} bytes)", file=sys.stderr)
            path = download(url, os.path.join(args.dl_dir, f"{c}.ar-en.zip"))

            def limited():
                for i, l in enumerate(arabic_lines(path)):
                    if i >= args.max_lines:
                        break
                    yield l

            # reservoir sample so we don't keep millions of chunks in memory
            sample, seen = [], set()
            for i, ch in enumerate(merge_into_chunks(limited(), args.min_words, args.max_words)):
                key = normalize(ch)
                if key in seen:
                    continue
                seen.add(key)
                if len(sample) < args.per_corpus:
                    sample.append(ch)
                else:
                    j = rng.randint(0, i)
                    if j < args.per_corpus:
                        sample[j] = ch
            for k, ch in enumerate(sample):
                out.write(json.dumps({"id": f"human-{c}-{k}", "text": ch, "label": "aranjiya",
                                      "source": f"opus:{c}", "book": f"opus:{c}", "genre": c},
                                     ensure_ascii=False) + "\n")
            print(f"{c}: kept {len(sample)} chunks", file=sys.stderr)


if __name__ == "__main__":
    main()
