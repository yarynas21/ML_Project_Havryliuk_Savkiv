import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from labeling import convert_to_ner_format
st.set_page_config(page_title="Ukrainian NER Demo", layout="wide")

MODEL_NAME = "savkivyaryna21/ukr_ner_mlproject"
SCORE_THRESHOLD = 0.55 

label_map = {
    "LABEL_1": "PER",
    "LABEL_2": "PER",
    "LABEL_3": "ORG",
    "LABEL_4": "ORG",
    "LABEL_5": "LOC",
    "LABEL_6": "LOC"
}

@st.cache_resource
def load_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

ner = load_ner_pipeline()

st.title("🇺🇦 Ukrainian NER Demo for Media Monitoring")
st.markdown("Analyze mentions of **people**, **organizations**, and **locations** in Ukrainian media texts.")

st.markdown("""
<style>
.badge {
  display: inline-block;
  padding: 3px 8px;
  margin-right: 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: bold;
}
.per { background-color: #ffd8d8; }
.org { background-color: #d8eafd; }
.loc { background-color: #d8fdd8; }
</style>
<div>
  <span class="badge per">PER = Person</span>
  <span class="badge org">ORG = Organization</span>
  <span class="badge loc">LOC = Location</span>
</div>
""", unsafe_allow_html=True)

text_input = st.text_area("Paste a news article or media post:", height=200)
export_csv = st.checkbox("📥 Export results as NER-labeled CSV")

highlight_colors = {
    "PER": "#ffd8d8",
    "ORG": "#d8eafd",
    "LOC": "#d8fdd8",
}

def highlight_text(text, entities):
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    for ent in entities:
        raw_label = ent['entity_group']
        label = label_map.get(raw_label)
        if not label or ent['score'] < SCORE_THRESHOLD:
            continue
        color = highlight_colors.get(label, '#eeeeee')
        word = text[ent['start']:ent['end']]
        score = ent['score']
        span = f'<span title="Score: {score:.2f}" style="background-color:{color}; padding:2px 4px; border-radius:4px">{word}</span>'
        text = text[:ent['start']] + span + text[ent['end']:]
    return text

if st.button("🔍 Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Running NER model..."):
            results = ner(text_input)
            print(f"Raw results: {results}")

        st.subheader("🖊️ Highlighted Entities")
        valid_results = [
            e for e in results
            if label_map.get(e['entity_group']) and e['score'] >= SCORE_THRESHOLD
        ]

        if export_csv and valid_results:
            import pandas as pd
            from io import StringIO
            id2label = {
                0: "O",
                1: "B-PER",
                2: "I-PER",
                3: "B-ORG",
                4: "I-ORG",
                5: "B-LOC",
                6: "I-LOC"
            }
            labeled_results = convert_to_ner_format(text_input, valid_results, id2label)
            df_ner = pd.DataFrame(labeled_results, columns=['word', 'label'])
            csv_buffer = StringIO()
            df_ner.to_csv(csv_buffer, index=False)
            st.download_button("⬇️ Download NER CSV", csv_buffer.getvalue(), file_name="ner_labeled_text.csv", mime="text/csv")

        highlighted_html = highlight_text(text_input, valid_results)
        st.markdown(highlighted_html, unsafe_allow_html=True)

        st.subheader("📃 Structured Output")
        collected = [
            (label_map.get(ent['entity_group']), ent['start'], ent['word'])
            for ent in valid_results
            if label_map.get(ent['entity_group'])
        ]
        collected = sorted(collected, key=lambda x: x[1])

        grouped = {"PER": [], "ORG": [], "LOC": []}
        for label, _, word in collected:
            clean_word = re.sub(r"^[^\w’ʼА-Яа-яІіЇїЄєҐґA-Za-z0-9]+|[^\w’ʼА-Яа-яІіЇїЄєҐґA-Za-z0-9]+$", "", word)
            if clean_word and clean_word not in grouped[label]:
                grouped[label].append(clean_word)

        for label, values in grouped.items():
            st.markdown(f"**{label}**: {', '.join(values) if values else '–'}")