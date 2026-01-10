from flask import Flask, render_template, request, send_file, Response
from flask import jsonify
from transformers import pipeline
from fpdf import FPDF
import requests
import re
import string
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer

SUPPORTED_TRANSLATION_LANGS = ["pl", "en", "de", "es"]

MARIAN_MODEL_MAP = {
    ("en", "pl"): "Helsinki-NLP/opus-mt-en-pl",
    ("pl", "en"): "Helsinki-NLP/opus-mt-pl-en",
    ("de", "pl"): "Helsinki-NLP/opus-mt-de-pl",
    ("pl", "de"): "Helsinki-NLP/opus-mt-pl-de",
    ("es", "pl"): "Helsinki-NLP/opus-mt-es-pl",
    ("pl", "es"): "Helsinki-NLP/opus-mt-pl-es",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("de", "es"): "Helsinki-NLP/opus-mt-de-es",
    ("es", "de"): "Helsinki-NLP/opus-mt-es-de",
}

_translation_cache = {}

def get_marian_translator(src_lang: str, tgt_lang: str):
    model_name = MARIAN_MODEL_MAP.get((src_lang, tgt_lang))
    if not model_name:
        return None, None, None

    if model_name in _translation_cache:
        return _translation_cache[model_name]

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    _translation_cache[model_name] = (model_name, tokenizer, model)
    return _translation_cache[model_name]

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    model_name, tokenizer, model = get_marian_translator(src_lang, tgt_lang)
    if not model_name:
        raise ValueError(f"Brak modelu Marian dla pary {src_lang}->{tgt_lang}")

    inputs = tokenizer([text], return_tensors="pt", truncation=True)
    translated = model.generate(**inputs, max_length=512)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

app = Flask(__name__)

def read_text_from_txt(file_storage) -> str:
    content = file_storage.read()
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")

print("Ładowanie modelu do streszczania...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Model załadowany.")
#test
MIN_INPUT_TOKENS = 60
MIN_CLEAN_WORDS = 30

def is_text_too_short(text):
    if not text or not text.strip():
        return True

    cleaned = re.sub(r"\s+", " ", text).strip()
    words = [w.strip(string.punctuation) for w in cleaned.split() if w.strip(string.punctuation)]
    clean_words_count = sum(1 for w in words if any(ch.isalpha() for ch in w))

    try:
        token_count = len(summarizer.tokenizer.encode(cleaned, add_special_tokens=False))
    except Exception:
        token_count = 0

    return token_count < MIN_INPUT_TOKENS or clean_words_count < MIN_CLEAN_WORDS
#test

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

lang_detector = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection"
)

@app.route('/detect-language', methods=['POST'])
def detect_language():
    text = request.json.get("text", "")
    if not text.strip():
        return {"lang": None}

    result = lang_detector(text[:500])
    lang = result[0]["label"]

    return {"lang": lang}

# --- DODANA FUNKCJONALNOŚĆ: ocena zgodności streszczenia (NLI entailment) ---
nli_checker = None
try:
    nli_checker = pipeline(
        "text-classification",
        model="facebook/bart-large-mnli",
        truncation=True,
    )
except Exception as e:
    print(f"Nie udało się załadować modelu NLI (sprawdzanie zgodności). Pomijam. Błąd: {e}")

def estimate_summary_faithfulness(original_text, summary):
    if not nli_checker:
        return None

    if not original_text or not original_text.strip() or not summary or not summary.strip():
        return 0.0

    tokenizer = summarizer.tokenizer

    chunk_size = 256
    ids = tokenizer.encode(original_text, add_special_tokens=False)

    chunks = []
    for i in range(0, len(ids), chunk_size):
        chunk_ids = ids[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

    if not chunks:
        return 0.0

    chunks = chunks[:10]

    entail_scores = []
    weights = []

    for ch in chunks:
        out = nli_checker(
            {"text": ch, "text_pair": summary},
            return_all_scores=True
        )

        scores = []
        if isinstance(out, list) and out:
            if isinstance(out[0], list):
                scores = out[0]
            elif isinstance(out[0], dict):
                scores = out

        entail = None

        for s in scores:
            lab = str(s.get("label", "")).upper()
            if "ENTAIL" in lab or lab == "LABEL_2":
                entail = float(s.get("score", 0.0))
                break

        if entail is None:
            entail = 0.0

        entail_scores.append(entail)
        weights.append(max(1, len(ch)))

    weighted = sum(s * w for s, w in zip(entail_scores, weights)) / sum(weights)
    return weighted * 100.0
# --- KONIEC DODANEJ FUNKCJONALNOŚCI ---
# --- DODANA FUNKCJONALNOŚĆ: offline zgodność (TF-IDF cosine) ---
def estimate_summary_similarity_tfidf(original_text, summary):
    if not original_text or not original_text.strip() or not summary or not summary.strip():
        return 0.0

    vect = TfidfVectorizer(stop_words=None)
    X = vect.fit_transform([original_text, summary])
    sim = cosine_similarity(X[0], X[1])[0][0]
    sim = max(0.0, min(1.0, float(sim)))
    return sim * 100.0
# --- KONIEC DODANEJ FUNKCJONALNOŚCI ---

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
        action = request.form.get("action", "summarize")
        
        text = request.form.get('text', '').strip()
        uploaded_file = request.files.get("text_file")

        if uploaded_file and uploaded_file.filename:
            if not uploaded_file.filename.lower().endswith(".txt"):
                return render_template(
                    "index.html",
                    error="Obsługiwane są tylko pliki .txt",
                    original_text=text
                )

            text = read_text_from_txt(uploaded_file).strip()
        
        
        lang = request.form.get('lang', 'en')
        summary_length = request.form.get('summary_length', 'medium')
        #test
        if not text:
            return render_template('index.html', error="Please enter some text.", original_text=text or "", lang=lang, summary_length=summary_length)
        
         # jeśli kliknięto "Tłumacz"
        if action == "translate":
            to_lang = request.form.get("to_lang", "en")

            # wykryj źródłowy język (masz już lang_detector)
            detected = lang_detector(text[:500])[0]["label"]

            # walidacja
            if detected not in SUPPORTED_TRANSLATION_LANGS:
                return render_template(
                    'index.html',
                    error=f"Nieobsługiwany język wejściowy: {detected}",
                    original_text=text,
                    lang=lang,
                    summary_length=summary_length
                )
            if to_lang not in SUPPORTED_TRANSLATION_LANGS:
                return render_template(
                    'index.html',
                    error=f"Nieobsługiwany język docelowy: {to_lang}",
                    original_text=text,
                    lang=lang,
                    summary_length=summary_length
                )
            if detected == to_lang:
                translated = text
            else:
                try:
                    translated = translate_text(text, detected, to_lang)
                except Exception as e:
                    return render_template(
                        'index.html',
                        error=f"Błąd tłumaczenia: {e}",
                        original_text=text,
                        lang=lang,
                        summary_length=summary_length
                    )

            return render_template(
                "result_translate.html",
                original_text=text,
                translated_text=translated,
                from_lang=detected,
                to_lang=to_lang
            )
            
        if is_text_too_short(text):
            return render_template('index.html', error="The provided text is too short.", original_text=text, lang=lang, summary_length=summary_length)
        
        #test
        summary = summarize_auto(text, summary_length=summary_length)
        
        # --- DODANA FUNKCJONALNOŚĆ: procent zgodności streszczenia z tekstem ---
        faithfulness_percent = estimate_summary_faithfulness(text, summary)
        # --- KONIEC DODANEJ FUNKCJONALNOŚCI ---
        
        # --- DODANA FUNKCJONALNOŚĆ: NLI jeśli dostępne, inaczej TF-IDF ---
        faithfulness_percent = estimate_summary_faithfulness(text, summary)
        if faithfulness_percent is None:
            faithfulness_percent = estimate_summary_similarity_tfidf(text, summary)
        # --- KONIEC DODANEJ FUNKCJONALNOŚCI ---

        
        # --- DODANA FUNKCJONALNOŚĆ: procent zgodności streszczenia z tekstem ---
        faithfulness_percent = estimate_summary_faithfulness(text, summary)
        # --- KONIEC DODANEJ FUNKCJONALNOŚCI ---
        
        # --- DODANA FUNKCJONALNOŚĆ: NLI jeśli dostępne, inaczej TF-IDF ---
        faithfulness_percent = estimate_summary_faithfulness(text, summary)
        if faithfulness_percent is None:
            faithfulness_percent = estimate_summary_similarity_tfidf(text, summary)
        # --- KONIEC DODANEJ FUNKCJONALNOŚCI ---

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
            faithfulness_percent=faithfulness_percent,
            faithfulness_percent=faithfulness_percent,
            definitions=definitions,
            original_text=text,
            lang=lang,
            summary_length_label=summary_length_labels.get(summary_length),
        )
    
    #return render_template('index.html')
    return render_template('index.html', original_text="", lang="en", summary_length="medium")

@app.route("/detect-lang", methods=["POST"])
def detect_lang():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"ok": False, "error": "empty"}), 400

    # bierzemy mały fragment żeby było szybciej
    sample = text[:500]

    try:
        detected = lang_detector(sample)[0]["label"]
        return jsonify({"ok": True, "lang": detected})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

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
