"""Local Gradio demo for كاشف العرنجية (Aranjiya Detector).

Scores text with v3 CAMeLBERT (automatically uses the 3-seed ensemble if all three
local checkpoints exist, otherwise falls back to the single seed-42 checkpoint) and
highlights influential words using the v3 TF-IDF companion model.

Run:
  python app.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from predict import AranjiyaDetector  # noqa: E402


DEFAULT_ENSEMBLE = [
    os.path.join(ROOT, "models", "v3_camelbert", "final"),
    os.path.join(ROOT, "models", "v3_camelbert_s101_devloss", "final"),
    os.path.join(ROOT, "models", "v3_camelbert_s202_devloss", "final"),
]
DEFAULT_BASELINE = os.path.join(ROOT, "models", "v3_baseline", "baseline.joblib")


def load_detector() -> AranjiyaDetector:
    env_models = os.environ.get("ARANJIYA_BERT_MODELS")
    if env_models:
        bert_paths = [p.strip() for p in env_models.split(",") if p.strip()]
    else:
        bert_paths = [p for p in DEFAULT_ENSEMBLE if os.path.isdir(p)]
        if not bert_paths:
            single = os.path.join(ROOT, "models", "v3_camelbert", "final")
            bert_paths = [single] if os.path.isdir(single) else []

    baseline_path = os.environ.get("ARANJIYA_BASELINE", DEFAULT_BASELINE)
    if not os.path.isfile(baseline_path):
        baseline_path = None

    return AranjiyaDetector(bert_paths=bert_paths or None, baseline_path=baseline_path)


EXAMPLES = [
    ["لعب دورا مهما في فتح البلاد، وذلك على الرغم من التحديات الكبيرة التي واجهت الجيش."],
    ["كان له في فتح البلاد أثر جليل، مع ما لقي الجيش من شدائد وعوائق."],
    ["وكل إنسان — إلى حد كبير — نتيجة لجميع ما ورثه عن آبائه، وما اكتسبه من بيئته التي أحاطت به."],
    ["وكل إنسان — في أكثر أمره — ثمرة ما ورثه عن آبائه، وما اكتسبه ممن نشأ بينهم."],
    ["أصر على إبائه على الرغم من إلحاح رجالات الجمعية مثل الشيخ محمد عبده."],
    ["أصر على إبائه مع إلحاح رجالات الجمعية مثل الشيخ محمد عبده."],
]


def main():
    import gradio as gr

    detector = load_detector()
    model_desc = (
        f"CAMeLBERT v3 ({len(detector.bert_models)}-seed ensemble)"
        if len(detector.bert_models) > 1
        else ("CAMeLBERT v3" if detector.bert_models else "TF-IDF baseline")
    )

    def analyze(text: str):
        text = (text or "").strip()
        if not text:
            return {}, []
        res = detector.score(text)
        p_ar = res["p_aranjiya"]
        scores = {"عرنجية (translationese)": p_ar, "فصحى (native Arabic)": round(1.0 - p_ar, 4)}

        spans = []
        for tok, wt in res.get("word_weights", []):
            if wt >= 0.45:
                tag = "عرنجية"
            elif wt <= -0.45:
                tag = "فصحى"
            else:
                tag = None
            spans.append((tok + " ", tag))
        return scores, spans

    with gr.Blocks(title="كاشف العرنجية — Aranjiya Detector") as demo:
        gr.Markdown(
            f"# كاشف العرنجية — Aranjiya Detector\n"
            f"Scores Arabic prose between native **فصحى** and English-influenced **عرنجية**.\n\n"
            f"- **Scorer:** {model_desc}\n"
            f"- **Word highlights:** TF-IDF companion model (approximates word-level push; "
            f"the CAMeLBERT score itself uses full sentence context)."
        )
        inp = gr.Textbox(label="النص العربي", placeholder="اكتب أو الصق فقرة عربية هنا...", lines=4, rtl=True)
        btn = gr.Button("تحليل", variant="primary")
        out_label = gr.Label(label="النتيجة", num_top_classes=2)
        out_hl = gr.HighlightedText(
            label="الكلمات المؤثرة (من النموذج الخطي المرافق)",
            color_map={"عرنجية": "red", "فصحى": "green"},
            combine_adjacent=False,
        )
        gr.Examples(examples=EXAMPLES, inputs=inp)

        btn.click(analyze, inputs=inp, outputs=[out_label, out_hl])
        inp.submit(analyze, inputs=inp, outputs=[out_label, out_hl])

    demo.launch()


if __name__ == "__main__":
    main()
