# كاشف العرنجية · Aranjiya Detector

A model that scores Arabic prose on a scale from native فصحى to عرنجية.

The term عرنجية comes from Ahmad al-Ghamdi's book «العرنجية: بلسانٍ عربيٍّ هجين» (2022). He uses it for Arabic that is grammatically correct on the surface but built on English or French structures underneath, so Arabic words carrying European phrasing. Some examples of the difference:

| عرنجية | فصحى |
| --- | --- |
| لعب دورا مهما في فتح البلاد | كان له في فتح البلاد أثر جليل |
| شعرت بالحزن | حزنت |
| والوعاظ المعينين من قبل رئيس الأمور الدينية | والوعاظ الذين يوليهم رئيس الأمور الدينية |
| بقيت عنده لمدة ليلتين | بت عنده ليلتين |

In NLP terms this is translationese detection, but aimed at writing style rather than at grammar mistakes.

The detector gives a score between 0 (native) and 1 (translated-style). It uses two models:

- CAMeLBERT-mix, fine-tuned (an ensemble of 3 runs), for the score itself.
- A TF-IDF + logistic regression model, which is weaker but shows which words pushed the score up. The demo uses it to highlight words.

## In short

- On classical authors it never saw during training, it labels 95% of passages correctly, and in 99% of pairs it gives the translated twin a higher score than the native original.
- On 38 minimal pairs (the same sentence from «حياتي» or «النظرات», with and without a modern calque), the ensemble picks the calque version every time.
- On modern books it never saw, it scores المنفلوطي's essays 88% فصحى and أحمد أمين's «حياتي» 70%, while «ما بعد الأيام», a 1986 memoir full of translated-style patterns, comes out 77% عرنجية.
- It took three versions to get there. Each earlier version had learned a shortcut (conversational voice, broken machine translation, "modern = عرنجية") that only showed up on carefully chosen tests. The section "What went wrong along the way" explains how each one was found and fixed.

## How the training data is built

The obvious dataset is classical books labeled فصحى and modern translations labeled عرنجية. A classifier trained on that gets close to 100% accuracy and learns almost nothing about style. It picks up whatever differs between the two piles: topic, era, vocabulary, names.

So every training example comes as a pair. I take a native Arabic passage, translate it to English, and translate the English back into Arabic with a machine translation model. The result is a "twin": same meaning, same topic, but written the way translated Arabic is written.

```
native passage (فصحى)  ->  English  ->  back into Arabic (عرنجية twin)
```

A passage and its twin always stay in the same split, and the splits are made by author, so no author appears in both training and testing.

### Filtering the twins

Machine translation brings its own problem. If broken output stays in the data, the model learns to spot broken Arabic instead of translated Arabic. That actually happened in v1 (see below), so every twin now goes through:

1. Eight rule-based checks: leftover Latin letters, repetition that isn't in the source, heavy vowel marking (a sign of degenerate NLLB output), tafsir-style «أي ...» glosses, too many «أي» compared with the source, length far off from the English pivot, too many words not found in an Arabic word list, and low meaning similarity to the English pivot (LaBSE below 0.75).
2. A check by claude-haiku-5-5 (Batch API) that the Arabic is real Arabic, without invented words or garbled names. Awkward or literal phrasing is allowed, since that is exactly what the twins are supposed to have.
3. One twin per passage. When both OpusMT and NLLB-1.3B give a clean twin, the one with the higher LaBSE score is kept.

All text, in training and testing, goes through the same normalization (`src/normalize.py`): diacritics, tatweel, honorific formulas, punctuation and digits are removed, and alef forms and ى are folded. Digits are deleted rather than replaced with a token, because translators write «533» where classical prose writes «ثلاث وثلاثين وخمسمائة», and a digit token would become a shortcut.

### Training data (v3)

5,976 pairs (11,952 rows):

- 4,376 classical passages from [OpenITI](https://github.com/OpenITI), including conversational prose (الجاحظ, الأغاني, عيون الأخبار, الكامل, المقامات, التنوخي) so that direct address and dialogue appear on the فصحى side too.
- 1,600 modern native passages: 800 from الرافعي «وحي القلم» (modern-topic chapters only) and 800 from طه حسين «الأيام».

English pivots come from Gemini and Claude Haiku. The Arabic twins come from OpusMT (`Helsinki-NLP/opus-mt-en-ar`) and NLLB-1.3B, run locally. Dev (375 pairs) and test (461 pairs) use classical authors that are not in training. OPUS human translations and Haiku's own Arabic translations are never used in training; they are test sets.

## What went wrong along the way

Most of the work on this project was finding shortcuts the model had learned and removing them, not tuning hyperparameters.

**v1.** Trained on twins from NLLB-600M and OpusMT plus unpaired human translations from OPUS (subtitles, TED talks, news). Three problems showed up:

- The OPUS data, especially the subtitles and TED talks, was full of first and second person and conversation (أنا، هل، الآن), while the classical side mostly wasn't. The model learned "talking to the reader = عرنجية" and flagged native essays like المنفلوطي's «النظرات». Removing OPUS from training alone raised النظرات from 73% to 81.5% فصحى (TF-IDF), without hurting the other tests.
- Much of the NLLB-600M output was broken: invented tafsir-style glosses («أي ربي ... أي الله»), words repeating in loops. NLLB inserted «أي» about 8 times more often than it appeared in the sources, and «أي» became the model's top عرنجية feature. 44% of NLLB-600M twins failed the stricter filter.
- I also checked whether the model relied on classical names and isnad formulas (حدثنا، عن، أبي فلان) by masking them. It didn't: masking flipped 0 predictions.

**v2.** NLLB-600M dropped, OPUS moved to test-only, the filters above added, conversational classical prose added. The features got much cleaner and the model generalized better to an unseen translator. But CAMeLBERT now called most modern native Arabic عرنجية: أحمد أمين's «حياتي» came out only 26% فصحى, the النظرات essays 57%. Every فصحى example was medieval and every عرنجية example was clean modern Arabic, so "modern = عرنجية" had become the easiest rule.

**v3.** Added twins of modern native prose (وحي القلم and الأيام). The النظرات essays went to 88% فصحى and حياتي to 70% (3-run ensemble). محمد حسن الزيات's «ما بعد الأيام» (1986), which my marker counts showed full of translated-style patterns (عندما، الآن، بالنسبة لنا), still comes out 77% عرنجية. That matters, because it shows the model didn't just flip to "modern = فصحى".

Two checks on v3:

- The best v3 run used 1 training epoch and v2 used 3, so the gain could have come from less training. I retrained v2 with 1 epoch: it did worse on the modern books (حياتي 22%, النظرات 51%). The gain comes from the new data.
- I trained v3 with 3 different seeds. Most results barely change, but the modern books vary a lot between runs (حياتي anywhere from 63% to 79%), so I report the average of the three. The checkpoint for each run is chosen by lowest dev loss. I settled on that rule after seeing that epoch 1 worked better on the modern books, so it wasn't a blind choice.

One more finding: almost half of Claude Haiku's "translations" of classical religious texts weren't translations at all. Haiku recalled the original Quran or hadith text from memory, sometimes word for word. I measured character overlap with the source: 46% of Haiku outputs overlapped heavily, against about 3% for OpusMT and 7% for NLLB. The Haiku results below use only the outputs that don't.

## Results (v3)

Main numbers are for the 3-run ensemble. Full numbers, including each run separately, are in [`results/v3_final_metrics.json`](results/v3_final_metrics.json).

| Test | Ensemble | TF-IDF |
| --- | --- | --- |
| Unseen classical authors (461 pairs): accuracy / pairwise | 95.0% / 99.1% | 92.7% / 98.3% |
| Calque minimal pairs (38): in paragraph / sentence alone | 38/38 · 38/38 | 35/38 · 35/38 |
| Hard test (83 items, 36 pairs): accuracy / pairwise | 86.8% / 100% | 79.5% / 88.9% |
| Haiku translations, non-recalled (236 pairs): pairwise / caught | 90.7% / 46.2% | 85.6% / 31.4% |
| OPUS human translations (4 corpora, 1,000 each): caught | 98.5–100% | 84–95% |

"Pairwise" means: given a native passage and its translated twin, how often the model gives the twin the higher score. The calque pairs (`data/calque_pairs.jsonl`) are single sentences from حياتي and النظرات in two versions, one with a modern calque and one without. In 22 pairs the author's original has the calque and a native version was written; in 16 the original is native and a calque was added.

Modern books, none of them used in training (% scored فصحى):

| Book | Ensemble | 3 runs (range) | TF-IDF |
| --- | --- | --- | --- |
| المنفلوطي, «النظرات» (his own essays) | 87.7% | 81.8–91.8% | 89.2% |
| أحمد أمين, «حياتي» | 70.4% | 63.5–79.1% | 85.0% |
| المنفلوطي, «النظرات» (chapters adapted from French) | 66.0% | 52.4–81.9% | 79.8% |
| المنفلوطي, «في سبيل التاج» (adapted from a French play) | 60.8% | 51.3–74.3% | 79.0% |
| محمد حسن الزيات, «ما بعد الأيام» | 23.4% | 19.5–28.6% | 53.0% |

These books have no gold labels. The expectation that the first two are mostly native and the last one mostly عرنجية is a judgment, based on al-Ghamdi's criteria and on counts of the patterns he describes.

Shortcut checks on the ensemble: masking classical names and isnad formulas flips 0 of 461 predictions. On the النظرات essays, chunks containing أنا/أنت/هل/أيها score 6.4 points lower than chunks without them (81.8% vs 88.3% فصحى), down from 11.6 points in v1.

## Limitations

- **Short sentences with one calque.** «صب لي بعض الماء» or «كيف كان يبدو؟» can land just under 0.5. The model still ranks them above their native versions, so for short text, comparing two versions is more reliable than reading the yes/no label.
- **Fluent AI-written Arabic.** On Haiku's translations the model picks the right version of a pair 90.7% of the time, but labels only 46% of the translations عرنجية on their own. Part of that is fair: in a manual review of 20 of the misses, about 9 were genuinely close to native Arabic. But about 6 had clear calques («لهذا السبب»، «ومن بين»، «ومع ذلك»، «رأس بشري») and were passages full of classical scholarly vocabulary. The model seems to trust classical vocabulary even when the sentence structure is modern.
- **Variance between runs** on modern books, described above.
- **Small hand-built tests.** The calque test is 38 pairs and the hard test 83 items, so one item is worth 1–3%.
- **Highlights come from TF-IDF, not from CAMeLBERT.** A calque spread across a whole phrase can raise the score without any single word lighting up.

## Usage

```bash
pip install -r requirements.txt
```

Score text with one model plus word highlighting:

```bash
python src/predict.py \
  --bert models/v3_camelbert/final \
  --baseline models/v3_baseline/baseline.joblib \
  "لعب دورا مهما في فتح البلاد" \
  "كان له في فتح البلاد أثر جليل"
```

With the 3-run ensemble, pass all three model folders to `--bert`:

```bash
python src/predict.py \
  --bert models/v3_camelbert/final models/v3_camelbert_s101_devloss/final models/v3_camelbert_s202_devloss/final \
  --baseline models/v3_baseline/baseline.joblib \
  "وكل إنسان إلى حد كبير نتيجة لجميع ما ورثه عن آبائه"
```

Web demo (needs `pip install gradio`):

```bash
python app.py
```

`app.py` loads the ensemble if all three models are in `models/`, otherwise the single model. Paths can be overridden with `ARANJIYA_BERT_MODELS` (comma-separated, local folders or Hugging Face repo IDs) and `ARANJIYA_BASELINE`.

Evaluate on the published test sets:

```bash
python scripts/eval_calque.py \
  --bert models/v3_camelbert/final models/v3_camelbert_s101_devloss/final models/v3_camelbert_s202_devloss/final
```

Train:

```bash
python src/train_baseline.py --data data/processed_v3 --hard data/hard_test.jsonl --out models/v3_baseline
python src/train_bert.py --data data/processed_v3 --hard data/hard_test.jsonl --out models/v3_camelbert --seed 42
```

The model weights and the built datasets are not in this repository (see the copyright note below).

## Repository layout

```
app.py                  Gradio demo (CAMeLBERT score + TF-IDF word highlights)
src/
  normalize.py          Shared Arabic normalization
  train_bert.py         CAMeLBERT fine-tuning (v3 settings)
  train_baseline.py     TF-IDF (word + char n-grams) + logistic regression
  predict.py            CLI and AranjiyaDetector class (single model or ensemble)
  evaluate.py           Accuracy, F1, ROC-AUC, pairwise accuracy
  openiti_extract.py    OpenITI markdown -> clean prose chunks
  backtranslate.py      Back-translation (NLLB, OpusMT, LLMs) with caching
  opus_download.py      OPUS loader
  build_dataset.py      Author-grouped splits and balancing
  spans.py              Token alignment helper
scripts/
  make_hard_test.py     Builds data/hard_test.jsonl
  eval_calque.py        Evaluates calque_pairs.jsonl and hard_test.jsonl
experiments/            The scripts behind every experiment in this README (filters, ablations, checks), kept as they were run. They expect the source books in ARANJIYA_BOOKS_DIR.
data/
  calque_pairs.jsonl    38 calque minimal pairs from حياتي and النظرات
  hard_test.jsonl       83 hand-picked test items
  seed/                 22 seed pairs for quick tests
results/
  v3_final_metrics.json Final v3 metrics
```

## Data and copyright

- «الأيام» (طه حسين) and «ما بعد الأيام» (محمد حسن الزيات) are still under copyright. They were used locally only, «الأيام» for training and «ما بعد الأيام» for testing. None of their text, chunks or twins is in this repository, only aggregate numbers.
- `data/calque_pairs.jsonl` is built from «حياتي» and «النظرات», both in the public domain. The rewritten versions were made for this project.
- `data/hard_test.jsonl` mixes three kinds of items: short example phrases quoted from al-Ghamdi's book (credited below), sentences written for this project, and my own examples, including lines of classical poetry.
- OpenITI and OPUS texts are not redistributed here. Each has its own license; see their sites.

## Credits

- The concept, the term and many examples: أحمد الغامدي، «العرنجية: بلسانٍ عربيٍّ هجين»، مركز تكوين، 2022.
- Base model: [CAMeL-Lab/bert-base-arabic-camelbert-mix](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix).
- Classical texts: [OpenITI](https://github.com/OpenITI). Human translations: [OPUS](https://opus.nlpl.eu). Modern books: [Hindawi](https://www.hindawi.org).
- AI assistance was used in this project, specifically Claude Opus 5.5 and Gemini Neon 4.
