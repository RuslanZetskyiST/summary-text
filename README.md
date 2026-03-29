# Text Analysis Web Application (Summarization & Translation)

## Overview
This project is a Flask-based web application for automatic text summarization and translation using modern NLP models.  
The system supports language detection, translation with fallback mechanisms, and quality evaluation of generated summaries.

## Key Features
- Text summarization (short / medium / long)
- Translation between PL / EN / DE / ES
- Automatic language detection
- TF-IDF similarity score
- Semantic similarity using sentence embeddings
- Fallback translation with NLLB
- Support for `.txt` file uploads

## Technologies
- Python 3
- Flask
- Hugging Face Transformers
- Sentence-Transformers
- Bootstrap 5 (UI)

## Models Used
- Summarization: `facebook/bart-large-cnn`
- Translation (primary): MarianMT (Helsinki-NLP)
- Translation (fallback): `facebook/nllb-200-distilled-600M`
- Semantic similarity: `paraphrase-multilingual-MiniLM-L12-v2`

## Architecture
The application follows a modular design:
- models are loaded once and cached in memory,
- fallback mechanisms ensure system robustness,
- UI and backend logic are clearly separated.

## Installation
```bash
pip install -r requirements.txt
```

## Running the App
```bash
python app.py
```