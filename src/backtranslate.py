"""Make عرنجية from fusha: Arabic → English (or French) → Arabic.

This is the method of «العرنجية» (al-Ghamdi): back-translate classical Arabic and
what comes back is the same meaning in translated-style Arabic. Every row we
produce is a PAIR — same content, same topic, same names — so the classifier
cannot win by looking at topic or era. Only style differs.

Backends (mix several so the model learns عرنجية, not one system's fingerprint):
  opus_mt  Helsinki-NLP Marian models (small, fast, free; GPU or CPU)
  nllb     Meta NLLB-200 (better quality; GPU recommended)
  google   Google Translate via deep-translator (free, rate-limited)
  gemini   Google Gemini API (needs GEMINI_API_KEY; pip install google-genai)
  claude   Anthropic API   (needs ANTHROPIC_API_KEY)
  openai   OpenAI API      (needs OPENAI_API_KEY)

Example:
  python src/backtranslate.py --inp data/raw/fusha.jsonl --out data/raw/pairs_nllb_google.jsonl \
      --n 20000 --forward nllb --backward google --pivot en

All translations are cached in data/cache/*.jsonl, so you can stop and resume.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize  # noqa: E402

LANG_NLLB = {"ar": "arb_Arab", "en": "eng_Latn", "fr": "fra_Latn"}


# --------------------------------------------------------------------------- cache
class Cache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.data[r["k"]] = r["v"]
                    except Exception:
                        pass
        self.f = open(path, "a", encoding="utf-8")

    @staticmethod
    def key(backend, src, tgt, text):
        return hashlib.md5(f"{backend}|{src}|{tgt}|{text}".encode()).hexdigest()

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v
        self.f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
        self.f.flush()


# ------------------------------------------------------------------------ backends
class Backend:
    name = "base"
    batch_size = 16

    def translate(self, texts, src, tgt):
        raise NotImplementedError


class OpusMT(Backend):
    name = "opus_mt"
    batch_size = 32

    def __init__(self):
        self.models = {}

    def _load(self, src, tgt):
        from transformers import MarianMTModel, MarianTokenizer
        import torch
        key = f"{src}-{tgt}"
        if key not in self.models:
            mid = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            tok = MarianTokenizer.from_pretrained(mid)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = MarianMTModel.from_pretrained(mid).to(dev).eval()
            self.models[key] = (tok, model, dev)
        return self.models[key]

    def translate(self, texts, src, tgt):
        import torch
        tok, model, dev = self._load(src, tgt)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        with torch.no_grad():
            out = model.generate(
                **enc, max_length=None, max_new_tokens=300, num_beams=4, no_repeat_ngram_size=4)
        return tok.batch_decode(out, skip_special_tokens=True)


class NLLB(Backend):
    name = "nllb"
    batch_size = 16

    def __init__(self, model_id="facebook/nllb-200-distilled-600M"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(self.dev).eval()

    def translate(self, texts, src, tgt):
        import torch
        self.tok.src_lang = LANG_NLLB[src]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.dev)
        with torch.no_grad():
            out = self.model.generate(
                **enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(LANG_NLLB[tgt]),
                max_length=None, max_new_tokens=300, num_beams=4, no_repeat_ngram_size=4)
        return self.tok.batch_decode(out, skip_special_tokens=True)


class Google(Backend):
    name = "google"
    batch_size = 8

    def translate(self, texts, src, tgt):
        from deep_translator import GoogleTranslator
        tr = GoogleTranslator(source=src, target=tgt)
        out = []
        for t in texts:
            for attempt in range(4):
                try:
                    out.append(tr.translate(t) or "")
                    break
                except Exception as e:  # rate limit / network: back off
                    time.sleep(2 ** attempt * 2)
                    if attempt == 3:
                        print("google failed:", e, file=sys.stderr)
                        out.append("")
            time.sleep(0.3)
        return out


_LLM_PROMPT = {
    ("ar", "en"): ("Translate each numbered classical Arabic passage into clear, natural, modern English, "
                   "the way a professional translator would for a general reader. Keep all meaning; do not "
                   "add explanations."),
    ("ar", "fr"): ("Traduisez chaque passage arabe classique numéroté en français moderne, clair et naturel, "
                   "comme le ferait un traducteur professionnel. Gardez tout le sens, sans explications."),
    ("en", "ar"): ("Translate each numbered English passage into Arabic, the way a professional translator "
                   "at a modern Arabic news site or publishing house would. Keep all meaning."),
    ("fr", "ar"): ("Translate each numbered French passage into Arabic, the way a professional translator "
                   "at a modern Arabic news site or publishing house would. Keep all meaning."),
}
_LLM_FORMAT = ('Return ONLY a JSON array of strings, one translation per passage, in the same order, '
               'with exactly {n} items.')


def _numbered(texts):
    return "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))


def _parse_json_list(s, n):
    m = re.search(r"\[.*\]", s, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(arr, list) or len(arr) != n:
        return None
    return [str(x) for x in arr]


class Claude(Backend):
    name = "claude"
    batch_size = int(os.environ.get("CLAUDE_BATCH_SIZE", "20"))

    def __init__(self, model=None):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY in your environment before using the claude backend.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
        self.sleep = float(os.environ.get("CLAUDE_SLEEP", "0.2"))

    def translate(self, texts, src, tgt):
        prompt = _LLM_PROMPT[(src, tgt)] + "\n" + _LLM_FORMAT.format(n=len(texts)) + "\n\n" + _numbered(texts)
        for attempt in range(5):
            try:
                r = self.client.messages.create(
                    model=self.model,
                    max_tokens=8000,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                )
                out = _parse_json_list(r.content[0].text, len(texts))
                if out:
                    if self.sleep > 0:
                        time.sleep(self.sleep)
                    return out
                print(f"claude bad json length (attempt {attempt + 1})", file=sys.stderr)
            except Exception as e:
                print(f"claude error (attempt {attempt + 1}):", e, file=sys.stderr)
                msg = str(e)
                if any(k in msg for k in ("401", "403", "404", "authentication_error", "not_found_error", "credit balance")):
                    raise
            time.sleep(min(60, 2 ** attempt * 2))
        return [""] * len(texts)


class OpenAI(Backend):
    name = "openai"
    batch_size = 10

    def __init__(self, model=None):
        from openai import OpenAI as _C
        self.client = _C()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def translate(self, texts, src, tgt):
        prompt = _LLM_PROMPT[(src, tgt)] + "\n" + _LLM_FORMAT.format(n=len(texts)) + "\n\n" + _numbered(texts)
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(model=self.model,
                                                        messages=[{"role": "user", "content": prompt}])
                out = _parse_json_list(r.choices[0].message.content, len(texts))
                if out:
                    return out
            except Exception as e:
                print("openai error:", e, file=sys.stderr)
            time.sleep(2 ** attempt * 2)
        return [""] * len(texts)


class Gemini(Backend):
    name = "gemini"
    batch_size = int(os.environ.get("GEMINI_BATCH_SIZE", "30"))

    def __init__(self, model=None):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in your environment before using the gemini backend.")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.sleep = float(os.environ.get("GEMINI_SLEEP", "0.5"))

    def translate(self, texts, src, tgt):
        from google.genai import types
        prompt = _LLM_PROMPT[(src, tgt)] + "\n" + _LLM_FORMAT.format(n=len(texts)) + "\n\n" + _numbered(texts)
        for attempt in range(5):
            try:
                r = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                out = _parse_json_list(r.text or "", len(texts))
                if out:
                    if self.sleep > 0:
                        time.sleep(self.sleep)
                    return out
                print(f"gemini bad json length (attempt {attempt + 1})", file=sys.stderr)
            except Exception as e:
                print(f"gemini error (attempt {attempt + 1}):", e, file=sys.stderr)
                msg = str(e)
                if any(k in msg for k in ("404", "401", "403", "API_KEY_INVALID", "PerDay", "per day")):
                    raise
            time.sleep(min(60, 2 ** attempt * 3))
        return [""] * len(texts)


BACKENDS = {
    "opus_mt": OpusMT, "nllb": NLLB, "google": Google,
    "gemini": Gemini, "claude": Claude, "openai": OpenAI,
}


def run_backend(backend, cache, texts, src, tgt, desc=""):
    out = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        if not t or not t.strip():
            out[i] = ""
            continue
        v = cache.get(Cache.key(backend.name, src, tgt, t))
        if v is not None:
            out[i] = v
        else:
            todo.append(i)
    bs = backend.batch_size
    for start in range(0, len(todo), bs):
        idx = todo[start:start + bs]
        res = backend.translate([texts[i] for i in idx], src, tgt)
        for i, r in zip(idx, res):
            out[i] = r
            if r:
                cache.put(Cache.key(backend.name, src, tgt, texts[i]), r)
        done = min(start + bs, len(todo))
        print(f"\r{desc} {backend.name} {src}->{tgt}: {done}/{len(todo)}", end="", file=sys.stderr)
    if todo:
        print(file=sys.stderr)
    return out


# ------------------------------------------------------------------------ filtering
def char_ngrams(s, n=3):
    s = normalize(s).replace(" ", "_")
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def similarity(a, b):
    A, B = char_ngrams(a), char_ngrams(b)
    return len(A & B) / max(1, len(A | B))


def has_repetition(text):
    words = normalize(text).split()
    if len(words) < 12:
        return False
    if len(set(words)) / len(words) < 0.45:
        return True
    from collections import Counter
    trigrams = Counter(zip(words, words[1:], words[2:]))
    if trigrams and trigrams.most_common(1)[0][1] >= 3:
        return True
    return False


def keep_pair(fusha, aranjiya, max_sim, min_sim):
    if not aranjiya or not re.search(r"[ء-ي]", aranjiya):
        return False, "empty"
    lf, la = len(normalize(fusha).split()), len(normalize(aranjiya).split())
    if la == 0 or not (0.5 <= la / max(1, lf) <= 2.5):
        return False, "length"
    if re.search(r"[A-Za-z]{3,}", aranjiya):
        return False, "latin"  # untranslated chunks
    if has_repetition(aranjiya):
        return False, "repetition"  # decoder loop
    sim = similarity(fusha, aranjiya)
    if sim > max_sim:
        return False, "too_similar"  # system memorized/copied the original
    if sim < min_sim:
        return False, "too_different"  # likely garbage / hallucination
    return True, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000, help="how many fusha chunks to back-translate")
    ap.add_argument("--forward", default="nllb", choices=BACKENDS)
    ap.add_argument("--backward", default="google", choices=BACKENDS)
    ap.add_argument("--pivot", default="en", choices=["en", "fr"])
    ap.add_argument("--max-sim", type=float, default=0.65)
    ap.add_argument("--min-sim", type=float, default=0.04)
    ap.add_argument("--genres", nargs="*", default=None, help="only use these genre buckets")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--cached-forward", action="store_true",
                    help="use only already-cached forward translations (0 API calls for forward)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    if args.genres:
        rows = [r for r in rows if r.get("genre") in args.genres]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    texts = [r["text"] for r in rows]

    cache = Cache(os.path.join(args.cache_dir, "translations.jsonl"))
    if args.cached_forward:
        pivot = [cache.get(Cache.key(args.forward, "ar", args.pivot, t)) or "" for t in texts]
        hits = sum(1 for p in pivot if p)
        print(f"cached-forward {args.forward} ar->{args.pivot}: {hits}/{len(texts)} hits", file=sys.stderr)
        bwd = BACKENDS[args.backward]()
    else:
        fwd = BACKENDS[args.forward]()
        bwd = fwd if args.backward == args.forward else BACKENDS[args.backward]()
        pivot = run_backend(fwd, cache, texts, "ar", args.pivot, "forward")

    back = run_backend(bwd, cache, [p or "" for p in pivot], args.pivot, "ar", "backward")

    from collections import Counter
    reasons = Counter()
    kept = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for r, p, b in zip(rows, pivot, back):
            ok, info = keep_pair(r["text"], b, args.max_sim, args.min_sim)
            if not ok:
                reasons[info] += 1
                continue
            kept += 1
            f.write(json.dumps({
                "pair_id": r["id"], "fusha": r["text"], "aranjiya": b, "pivot": p,
                "pivot_lang": args.pivot, "forward": args.forward, "backward": args.backward,
                "sim": round(info, 3), "book": r.get("book"), "author": r.get("author"),
                "genre": r.get("genre"),
            }, ensure_ascii=False) + "\n")
    print(f"kept {kept}/{len(rows)} pairs; dropped: {dict(reasons)} -> {args.out}")


if __name__ == "__main__":
    main()
