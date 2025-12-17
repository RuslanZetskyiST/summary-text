from flask import Flask, render_template, request, send_file, Response
from transformers import pipeline
from fpdf import FPDF
import requests
import re
import string
import io

app = Flask(__name__)


print("Ładowanie modelu do streszczania...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Model załadowany.")


def summarize_auto(text, summary_length="medium"):
    tokenizer = summarizer.tokenizer
    model = summarizer.model

    length_profiles = {
        "short": {
            "final": {"max_length": 80, "min_length": 20},
            "partial": {"max_length": 60, "min_length": 20},
            "final_long": {"max_length": 90, "min_length": 25},
        },
        "medium": {
            "final": {"max_length": 150, "min_length": 40},
            "partial": {"max_length": 120, "min_length": 40},
            "final_long": {"max_length": 150, "min_length": 60},
        },
        "long": {
            "final": {"max_length": 220, "min_length": 80},
            "partial": {"max_length": 160, "min_length": 60},
            "final_long": {"max_length": 220, "min_length": 90},
        },
    }
    profile = length_profiles.get(summary_length, length_profiles["medium"])

    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)

    max_input_tokens = None
    if isinstance(model_limit, int) and model_limit > 0:
        max_input_tokens = model_limit
    elif isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        max_input_tokens = tokenizer_limit
    else:
        max_input_tokens = 1024

    chunk_size = max(128, max_input_tokens - 2)

    def summarize_once(text_part, max_len, min_len):
        out = summarizer(
            text_part,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        return out[0]["summary_text"]

    def split_by_tokens(full_text):
        input_ids = tokenizer.encode(full_text, add_special_tokens=False)
        for i in range(0, len(input_ids), chunk_size):
            chunk_ids = input_ids[i:i + chunk_size]
            yield tokenizer.decode(chunk_ids, skip_special_tokens=True)

    input_len = len(tokenizer.encode(text, add_special_tokens=False))
    if input_len <= max_input_tokens:
        return summarize_once(text, max_len=profile["final"]["max_length"], min_len=profile["final"]["min_length"])

    current = text
    for _ in range(3):
        parts = list(split_by_tokens(current))
        if len(parts) == 1:
            return summarize_once(
                parts[0],
                max_len=profile["final"]["max_length"],
                min_len=profile["final"]["min_length"],
            )

        partial_summaries = [
            summarize_once(
                p,
                max_len=profile["partial"]["max_length"],
                min_len=profile["partial"]["min_length"],
            )
            for p in parts
        ]
        current = " ".join(partial_summaries)

        if len(tokenizer.encode(current, add_special_tokens=False)) <= max_input_tokens:
            return summarize_once(
                current,
                max_len=profile["final_long"]["max_length"],
                min_len=profile["final_long"]["min_length"],
            )

    parts = list(split_by_tokens(current))
    partial_summaries = [
        summarize_once(
            p,
            max_len=profile["partial"]["max_length"],
            min_len=profile["partial"]["min_length"],
        )
        for p in parts
    ]
    return " ".join(partial_summaries)

LANG_CONFIG = {
    "pl": {
        "name": "polski",
        "markers": [r'znaczenia', r'rzeczownik', r'czasownik', r'przymiotnik', r'wyrażenie'],
    },
    "de": {
        "name": "Deutsch",
        "markers": [r'Bedeutungen', r'Substantiv', r'Verb', r'Adjektiv'],
    },
    "es": {
        "name": "Español",
        "markers": [r'Sustantivo', r'Verbo', r'Adjetivo', r'Forma verbal'],
    },
    "en": {
        "name": "English",
        "markers": [r'Noun', r'Verb', r'Adjective', r'Definition'],
    },
}

STOPWORDS = {
    "pl": set(["i","oraz","w","na","do","o","że","a","to","jest","z","się"]),
    "de": set(["und","oder","die","der","das","ein","eine","ist","zu","vom","im"]),
    "es": set(["y","o","de","la","el","que","es","en","un","una"]),
    "en": set(["and","or","the","is","are","of","to","in","for","on","with"]),
}

def extract_difficult_words(text, lang="en"):
    words = [
        w.strip(string.punctuation).lower()
        for w in text.split()
        if w.strip(string.punctuation)
    ]
    stop = STOPWORDS.get(lang, set())
    difficult = [
        w for w in words
        if len(w) > 5 and w not in stop and w.isalpha()
    ]
    return list(sorted(set(difficult)))

def get_definitions(words, lang="en"):
    result = {}
    cfg = LANG_CONFIG.get(lang, LANG_CONFIG["en"])
    markers_regex = re.compile("|".join(cfg["markers"]), re.IGNORECASE)

    for word in words:
        url = f"https://{lang}.wiktionary.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "titles": word,
            "format": "json",
            "explaintext": True,
            "redirects": True,
            "utf8": True
        }
        headers = {"User-Agent": "DefinitionBot/1.0"}

        try:
            r = requests.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
            
            pages = data.get("query", {}).get("pages", {})
            page_id = next(iter(pages))
            
            if page_id == "-1":
                result[word] = ["Brak definicji."]
                continue

            extract = pages[page_id].get("extract", "")
            if not extract:
                result[word] = ["Brak treści w artykule."]
                continue

            lines = extract.split("\n")
            in_lang_section = False
            in_def_block = False
            definitions = []
            
            lang_header = re.compile(
                rf"^==\s*{cfg['name']}\s*==$|^==\s*język {cfg['name']}\s*==$",
                re.IGNORECASE
            )

            for line in lines:
                stripped = line.strip()
                if re.match(r"^==[^=]+==$", stripped):
                    if lang_header.match(stripped):
                        in_lang_section = True
                        in_def_block = False
                    else:
                        in_lang_section = False
                    continue

                if not in_lang_section:
                    continue

                if markers_regex.search(stripped):
                    in_def_block = True
                    continue

                if in_def_block:
                    clean = re.sub(r'\{\{.*?\}\}', '', stripped)
                    clean = re.sub(r'\[\[(.*?)\|(.*?)\]\]', r'\2', clean)
                    clean = re.sub(r'\[\[(.*?)\]\]', r'\1', clean).strip()

                    if re.match(r"^\d+\.\s", clean) or re.match(r"^[\*\-]\s", clean):
                        if len(clean) > 5:
                            definitions.append(clean)

            if not definitions:
                first = next(
                    (l.strip() for l in lines if len(l.strip()) > 40 and not l.startswith("==")),
                    None
                )
                if first:
                    definitions = [first]
                else:
                    definitions = ["Nie udało się odczytać definicji."]

            result[word] = definitions[:5]

        except Exception as e:
            result[word] = [f"Błąd: {e}"]

    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        lang = request.form.get('lang', 'en')
        summary_length = request.form.get('summary_length', 'medium')
        
        if not text:
            return render_template('index.html', error="Please enter some text.")

        summary = summarize_auto(text, summary_length=summary_length)
        
        difficult_words = extract_difficult_words(text, lang)
        definitions = get_definitions(difficult_words[:10], lang)

        summary_length_labels = {
            "short": "krótkie",
            "medium": "średnie",
            "long": "długie",
        }
        return render_template(
            'result.html',
            summary=summary,
            definitions=definitions,
            original_text=text,
            lang=lang,
            summary_length_label=summary_length_labels.get(summary_length),
        )
    
    return render_template('index.html')

@app.route('/download/<format>', methods=['POST'])
def download(format):
    summary = request.form.get('summary')
    if not summary:
        return "No summary to download", 400

    if format == 'txt':
        return Response(
            summary,
            mimetype="text/plain",
            headers={"Content-disposition": "attachment; filename=summary.txt"}
        )
    
    elif format == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary.encode('latin-1', 'replace').decode('latin-1'))
        
        pdf_output = io.BytesIO()
        val = pdf.output(dest='S').encode('latin-1')
        return Response(
            val,
            mimetype="application/pdf",
            headers={"Content-disposition": "attachment; filename=summary.pdf"}
        )

    return "Invalid format", 400

if __name__ == '__main__':
    app.run(debug=True)
