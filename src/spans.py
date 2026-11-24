"""Phrase-level labels for free, from the pairs.

Line up the original fusha with its back-translation, word by word. Words in the
عرنجية side that were not in the original (inserted or replaced) are the
translation-introduced words: «يلعب دورًا»، «بالرغم من»، «سوف»، «بعض»...
Those become token labels (1 = عرنجية word, 0 = shared with the original),
so the model can learn to HIGHLIGHT phrases, not only score sentences.
"""
from difflib import SequenceMatcher

from normalize import normalize


def token_labels(fusha: str, aranjiya: str):
    a = normalize(fusha).split()
    b = normalize(aranjiya).split()
    labels = [1] * len(b)
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    for block in sm.get_matching_blocks():
        for j in range(block.b, block.b + block.size):
            labels[j] = 0
    return b, labels


def show(fusha, aranjiya):
    toks, labs = token_labels(fusha, aranjiya)
    return " ".join(f"[{t}]" if l else t for t, l in zip(toks, labs))


if __name__ == "__main__":
    f = "وتولى عنهم وقال يا أسفى على يوسف وابيضت عيناه من الحزن فهو كظيم"
    r = "واستدار بعيدا عنهم وقال يا لحزني على يوسف وأصبحت عيناه بيضاء من الحزن وكان يحاول أن يخفي مشاعره"
    print(show(f, r))
