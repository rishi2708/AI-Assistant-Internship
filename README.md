# AI-Assistant-Internship

Production-quality internship repository for a single integrated AI Assistant built with Python, Streamlit, Google Gemini, LangChain, FAISS, Sentence Transformers, MedQuAD-style medical data, arXiv-style scientific data, multilingual processing, sentiment analysis, multimodal image understanding, persistent memory, testing, reports, notebooks, and GitHub Actions.

> The complete application source is inside [`AI-Assistant-Internship/`](AI-Assistant-Internship/).

## Submission Summary

This repository is organized for internship evaluation. It does not contain multiple disconnected demos. All required features are implemented in one Streamlit AI Assistant application with shared configuration, shared memory, reusable RAG components, and a professional dashboard.

## Evaluation Task Coverage

| Internship requirement | Status | Evidence |
| --- | --- | --- |
| Dynamic knowledge-base expansion | Complete | Scheduler, cron workflow, SHA-256 manifest, incremental indexing: [`schedule_knowledge_updates.py`](AI-Assistant-Internship/schedule_knowledge_updates.py), [`chatbot/knowledge_updater.py`](AI-Assistant-Internship/chatbot/knowledge_updater.py), [root scheduled workflow](.github/workflows/scheduled-knowledge-update.yml) |
| Multi-modal assistant for text and images | Complete | Evidence/ambiguity/validation pipeline: [`chatbot/vision.py`](AI-Assistant-Internship/chatbot/vision.py), [`chatbot/multimodal_reasoning.py`](AI-Assistant-Internship/chatbot/multimodal_reasoning.py) |
| Medical Q&A chatbot using MedQuAD | Complete | [`chatbot/medical.py`](AI-Assistant-Internship/chatbot/medical.py), [`chatbot/medical_entities.py`](AI-Assistant-Internship/chatbot/medical_entities.py), citations, medical disclaimer |
| Domain expert chatbot using arXiv | Complete | CS subset retrieval, open-source summarization fallback, concept graph, follow-ups: [`chatbot/scientific.py`](AI-Assistant-Internship/chatbot/scientific.py), [`data/arxiv/computer_science_subset.jsonl`](AI-Assistant-Internship/data/arxiv/computer_science_subset.jsonl) |
| Sentiment analysis and tone adaptation | Complete | Full sentiment-aware generation/evaluation pipeline: [`chatbot/sentiment.py`](AI-Assistant-Internship/chatbot/sentiment.py), [`chatbot/sentiment_pipeline.py`](AI-Assistant-Internship/chatbot/sentiment_pipeline.py) |
| Multilingual conversations | Complete | Context-preserving multilingual state: [`chatbot/multilingual.py`](AI-Assistant-Internship/chatbot/multilingual.py), [`chatbot/multilingual_context.py`](AI-Assistant-Internship/chatbot/multilingual_context.py) |
| Complete RAG pipeline | Complete | [`chatbot/rag.py`](AI-Assistant-Internship/chatbot/rag.py), [`chatbot/retriever.py`](AI-Assistant-Internship/chatbot/retriever.py), [`chatbot/embeddings.py`](AI-Assistant-Internship/chatbot/embeddings.py) |
| Streamlit dashboard | Complete | [`app.py`](AI-Assistant-Internship/app.py), sidebar controls, uploads, modes, chat export |
| Visual outputs and metrics | Complete | [`images/`](AI-Assistant-Internship/images/), [`reports/evaluation_metrics.json`](AI-Assistant-Internship/reports/evaluation_metrics.json) |
| Notebook/report documentation | Complete | [`notebooks/`](AI-Assistant-Internship/notebooks/), [`reports/internship_report.md`](AI-Assistant-Internship/reports/internship_report.md) |
| Testing and CI | Complete | [`tests/`](AI-Assistant-Internship/tests/), [`.github/workflows/ci.yml`](AI-Assistant-Internship/.github/workflows/ci.yml) |

## Features

- General Gemini chatbot with conversation history, system prompt, temperature control, clear chat, error handling, and streaming-ready design.
- Medical RAG chatbot using MedQuAD-style data with preprocessing, chunking, embeddings, FAISS retrieval, top-k control, citations, and disclaimer.
- Scientific/domain expert chatbot using arXiv-style metadata for paper retrieval, summarization, context generation, and references.
- Dynamic knowledge base with PDF, TXT, Markdown, and DOCX upload support, text extraction, chunking, embeddings, FAISS persistence, and reload.
- Multimodal Gemini Vision support for image upload, OCR-assisted analysis, image reasoning, and mixed image plus text prompts.
- Multilingual support for English, Hindi, Bengali, and Spanish with language detection, translation, retrieval, generation, and context continuity.
- Sentiment analysis for positive, negative, and neutral messages with tone adaptation and visual sentiment distribution.
- Window memory and persistent memory for previous questions and chat export.
- Analytics dashboard with dataset statistics, response latency, accuracy metrics, confusion matrix, sentiment graphs, and embedding visualization.

## Repository Structure

```text
.
|-- README.md
`-- AI-Assistant-Internship/
    |-- app.py
    |-- config.py
    |-- requirements.txt
    |-- .env.example
    |-- chatbot/
    |-- utils/
    |-- data/
    |-- uploads/
    |-- vector_db/
    |-- images/
    |-- reports/
    |-- notebooks/
    |-- tests/
    `-- .github/
```

## Quick Start

```powershell
cd AI-Assistant-Internship
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_TEXT_MODEL=gemini-2.5-flash
GEMINI_VISION_MODEL=gemini-2.5-flash
APP_ENV=development
LOG_LEVEL=INFO
```

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Important Reproducibility Notes

- Use Python 3.10 or 3.11 for the most reliable FAISS, Sentence Transformers, NumPy, and scientific package compatibility.
- Keep `.env` private. API keys are intentionally excluded from Git.
- Sample MedQuAD and arXiv-style data are included for reproducible evaluation. Full datasets can be placed under `data/medical/` and `data/arxiv/`.
- Uploaded files and generated vector indexes are kept out of Git except for placeholder files.

## Documentation

- Full project README: [`AI-Assistant-Internship/README.md`](AI-Assistant-Internship/README.md)
- Evaluator remediation evidence: [`AI-Assistant-Internship/reports/evaluator_remediation.md`](AI-Assistant-Internship/reports/evaluator_remediation.md)
- Internship report: [`AI-Assistant-Internship/reports/internship_report.md`](AI-Assistant-Internship/reports/internship_report.md)
- Task completion audit: [`AI-Assistant-Internship/reports/task_completion_audit.md`](AI-Assistant-Internship/reports/task_completion_audit.md)
- Methodology details: [`AI-Assistant-Internship/reports/methodology_details.md`](AI-Assistant-Internship/reports/methodology_details.md)
- Model comparison: [`AI-Assistant-Internship/reports/model_comparison.md`](AI-Assistant-Internship/reports/model_comparison.md)
- Notebook artifact: [`AI-Assistant-Internship/notebooks/AI_Assistant_Internship_Analysis.ipynb`](AI-Assistant-Internship/notebooks/AI_Assistant_Internship_Analysis.ipynb)

## Testing

```powershell
cd AI-Assistant-Internship
.\.venv\Scripts\python.exe -m pytest
```

## License

This project uses the MIT License. See [`AI-Assistant-Internship/LICENSE`](AI-Assistant-Internship/LICENSE).
