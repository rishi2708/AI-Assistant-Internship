# Methodology Details

## Problem Statement

The goal is to build one professional AI Assistant that combines general conversation, domain-specific medical and scientific question answering, dynamic document knowledge expansion, image reasoning, multilingual support, sentiment-aware responses, memory, and analytics.

## Dataset Handling

### MedQuAD

- Input format: JSON, JSONL, CSV, or TXT records.
- Important fields: `question`, `answer`, `focus_area`, and `source`.
- Preprocessing: normalize whitespace, remove control characters, preserve question-answer structure.
- Feature engineering: extract disease/symptom/treatment/test entities using `MedicalEntityRecognizer`.
- Retrieval: chunk records, embed chunks, index with FAISS, retrieve top-k relevant chunks.

### arXiv

- Input format: JSON, JSONL, CSV, or TXT metadata records.
- Important fields: `id`, `title`, `authors`, `categories`, `abstract`, and `summary`.
- Preprocessing: normalize title, author, category, and abstract text.
- Feature engineering: keyword concept extraction, paper summaries, and references.
- Retrieval: chunk abstracts and metadata, embed chunks, retrieve top-k context for scientific answers.

## Model Selection

- Baseline: direct Gemini prompt without retrieval.
- Advanced model: Gemini with RAG, memory, citations, and task-specific preprocessing.
- Embeddings: Sentence Transformers `all-MiniLM-L6-v2`.
- Vector store: FAISS.
- Sentiment model: lexical baseline with optional upgrade path to VADER/spaCy when environment allows.

## Evaluation Metrics

- Response latency.
- Sentiment classification distribution.
- Dataset document counts.
- Confusion matrix for sentiment or labeled classification experiments.
- Retrieval quality through citation relevance and top-k context inspection.

## Reproducibility

- Dependencies are listed in `requirements.txt`.
- Secrets are stored in `.env` and excluded by `.gitignore`.
- Sample datasets are included in `data/medical` and `data/arxiv`.
- Tests are included under `tests/`.
- A GitHub Actions workflow is included under `.github/workflows/ci.yml`.
