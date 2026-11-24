"""Shared Arabic text normalization.

Every text — fusha or عرنجية, training or test — goes through `normalize()`
so the model cannot use surface differences (diacritics, letter variants,
punctuation habits) as a shortcut. It has to learn style.
"""
import re
import unicodedata

# Tashkeel + Quranic annotation marks + tatweel
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ALEF = re.compile(r"[إأآٱ]")
_NON_ARABIC = re.compile(r"[^ء-غف-ي\s]")
_LATIN_DIGITS = re.compile(r"[0-9٠-٩۰-۹]+")
_SPACES = re.compile(r"\s+")

# Formulas that only appear in classical/religious text. Leaving them in would
# let the model "cheat": see صلى الله عليه وسلم → answer فصحى.
_HONORIFICS = [
    r"صل[يى] الله عليه(?: وآله)?(?: و ?سلم)?", r"عليه الصلاة والسلام", r"عليه السلام", r"عليها السلام",
    r"عليهم السلام", r"رض[يى] الله عنه(?:ما|م|ا)?", r"رض[يى] الله عنها", r"رحمه الله(?: تعالى)?",
    r"رحمها الله", r"رحمهم الله", r"تعالى", r"عز وجل", r"جل جلاله", r"سبحانه وتعالى",
    r"تبارك وتعالى", r"كرم الله وجهه",
]
_HONORIFIC_RE = re.compile("|".join(_HONORIFICS))
# The ﷺ-style ligatures
_LIGATURES = re.compile(r"[﵀-﷿ﹰ-﻿]")


def strip_diacritics(text: str) -> str:
    return _DIACRITICS.sub("", text)


def normalize(text: str, keep_punct: bool = False, fold_letters: bool = True, fold_alef_ya: bool = False) -> str:
    """Normalize Arabic text for classification.

    - remove diacritics and tatweel
    - fold_letters=True (TF-IDF): unify alef forms, ى→ي, ة→ه; ؤ/ئ kept
    - fold_letters=False, fold_alef_ya=True (v2 BERT): unify alef forms and ى→ي, keep ة distinct
    - fold_letters=False, fold_alef_ya=False (v1 BERT): keep normal spelling
    - remove honorific formulas, Quran/ligature symbols, digits, Latin, stray symbols, and punctuation
    """
    text = unicodedata.normalize("NFKC", text)
    text = _LIGATURES.sub(" ", text)
    text = strip_diacritics(text)
    text = _HONORIFIC_RE.sub(" ", text)
    if fold_letters:
        text = _ALEF.sub("ا", text)
        text = text.replace("ى", "ي").replace("ة", "ه")
    elif fold_alef_ya:
        text = _ALEF.sub("ا", text)
        text = text.replace("ى", "ي")
    text = _LATIN_DIGITS.sub(" ", text)
    if keep_punct:
        text = re.sub(r"[^ء-غف-ي\s.,،؛;:!?؟]", " ", text)
    else:
        text = _NON_ARABIC.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def light_clean(text: str) -> str:
    """Readable version for humans / translation systems: keeps diacritics-free
    text with punctuation, removes markup leftovers. Used before translation,
    because MT systems translate better with punctuation."""
    text = unicodedata.normalize("NFKC", text)
    text = _LIGATURES.sub(" ", text)
    text = strip_diacritics(text)
    text = _HONORIFIC_RE.sub(" ", text)
    text = re.sub(r"[\[\]{}«»\"'()<>*#|~_=+@^`\\/]", " ", text)
    text = _LATIN_DIGITS.sub(" ", text)
    text = re.sub(r"\s+([.,،؛;:!?؟])", r"\1", text)
    return _SPACES.sub(" ", text).strip()


def tokens(text: str):
    return normalize(text).split()


if __name__ == "__main__":
    s = "قَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ: «إِنَّ الدِّينَ يُسْرٌ» [ص: 12]"
    print(light_clean(s))
    print(normalize(s))
