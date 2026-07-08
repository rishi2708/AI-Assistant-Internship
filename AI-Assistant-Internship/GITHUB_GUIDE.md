# GitHub Guide

## Suggested Commit Sequence

```bash
git add .
git commit -m "Initialize integrated AI assistant project"
git commit -m "Add Gemini chatbot, memory, and sentiment modules"
git commit -m "Add RAG pipeline with FAISS and dataset loaders"
git commit -m "Add Streamlit dashboard and multimodal workflows"
git commit -m "Add documentation, tests, and CI"
```

## Suggested Issues

1. Add full MedQuAD dataset ingestion benchmark.
2. Add arXiv category filters and date range filters.
3. Add reranker for higher-quality retrieval.
4. Add Dockerfile and deployment workflow.
5. Add labeled evaluation set for answer accuracy.

## Release Plan

- `v0.1.0`: Initial internship submission.
- `v0.2.0`: Add Docker and deployment documentation.
- `v0.3.0`: Add advanced evaluation and reranking.
- `v1.0.0`: Stable production-ready release.

## Repository Hygiene

- Keep `.env` private.
- Do not commit uploaded user files or generated FAISS indexes.
- Add screenshots to `images/` before final submission.
- Run `pytest` before every release.
