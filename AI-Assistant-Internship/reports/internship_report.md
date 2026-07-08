# Internship Report: AI-Assistant-Internship

## Introduction

The AI-Assistant-Internship project is a unified artificial intelligence assistant built with Python, Streamlit, LangChain, Google Gemini, FAISS, and Sentence Transformers. The project integrates general conversation, medical question answering, scientific research assistance, dynamic document ingestion, image understanding, multilingual support, sentiment analysis, memory, and visual analytics into one production-style application.

## Background

Modern assistants are expected to answer general questions, retrieve grounded information from private knowledge, support multiple languages, understand images, and maintain context. This project demonstrates those capabilities with a modular architecture suitable for internship evaluation and future extension.

## Objectives

- Build one complete AI Assistant project rather than independent demos.
- Use Gemini for language and vision capabilities.
- Use LangChain-compatible components for retrieval workflows.
- Implement FAISS vector search with Sentence Transformer embeddings.
- Support MedQuAD and arXiv-style datasets.
- Provide a professional Streamlit dashboard.
- Include testing, documentation, CI, issue templates, and release guidance.

## Technology Stack

- Frontend: Streamlit
- Backend: Python
- LLM: Google Gemini API
- Framework: LangChain
- Vector database: FAISS
- Embeddings: Sentence Transformers `all-MiniLM-L6-v2`
- ML: scikit-learn
- NLP: NLTK and spaCy-ready environment
- Translation: deep-translator
- Language detection: langdetect
- Visualization: Matplotlib and Plotly
- Image understanding: Gemini Vision with OCR support
- Testing: pytest

## Tasks Completed

- Implemented Gemini chatbot with memory, system prompt, temperature control, and error handling.
- Implemented MedQuAD RAG with preprocessing, chunking, embeddings, FAISS, top-k retrieval, answer generation, citations, and disclaimer.
- Implemented basic medical entity recognition for symptoms, diseases, tests, and treatments.
- Implemented arXiv domain expert RAG for scientific retrieval and paper-style references.
- Implemented arXiv paper search, extractive summaries, references, and concept extraction for visualization-ready output.
- Implemented dynamic knowledge base expansion through PDF, TXT, MD, and DOCX uploads.
- Implemented configured source scanning for periodic knowledge-base updates.
- Implemented Gemini Vision image analysis and OCR-assisted reasoning.
- Implemented multilingual detection, mixed-language segment handling, translation, answer generation, and translation back to supported languages.
- Implemented sentiment detection and tone adaptation.
- Implemented persistent and windowed conversation memory.
- Implemented analytics for sentiment, latency, dataset statistics, confusion matrix, and embeddings.
- Added tests, README, GitHub Actions, issue templates, and release notes.

## Methodology

The project uses a retrieval augmented generation pipeline:

```text
Document Loader -> Text Cleaning -> Chunking -> Embeddings -> FAISS -> Retriever -> Gemini -> Answer
```

For dataset-backed modes, MedQuAD and arXiv records are converted to LangChain `Document` objects with metadata. Documents are chunked using sentence-aware windows. Embeddings are generated with `all-MiniLM-L6-v2` and stored in persisted FAISS indexes. At query time, the system retrieves relevant chunks, formats them into a grounded prompt, and asks Gemini to generate an answer with references.

## Preprocessing And Feature Engineering

- Text preprocessing normalizes whitespace, removes control characters, and preserves dataset metadata.
- MedQuAD feature engineering extracts symptoms, diseases, tests, and treatments.
- arXiv feature engineering extracts frequent scientific concepts, paper references, and summaries.
- Vision preprocessing extracts OCR text and combines it with image input.
- Multilingual preprocessing detects the dominant language and mixed-language segments.
- Sentiment preprocessing converts user messages into positive, negative, or neutral labels for tone adaptation.

## Model Comparison

The baseline is a direct Gemini prompt. The advanced system adds retrieval, metadata, citations, memory, multimodal inputs, multilingual translation, and sentiment-aware tone. The advanced system is preferred because it is more grounded and auditable. See `reports/model_comparison.md` and `reports/model_comparison.csv`.

## Visual Outputs

The repository includes visual outputs required for submission:

- `images/sentiment_distribution.svg`
- `images/confusion_matrix.svg`
- `images/model_comparison.svg`

The notebook artifact is available at `notebooks/AI_Assistant_Internship_Analysis.ipynb`.

## Implementation

The implementation is split into focused modules:

- `chatbot/base_chatbot.py`: Gemini text chatbot.
- `chatbot/rag.py`: reusable RAG pipeline.
- `chatbot/retriever.py`: FAISS persistence and retrieval.
- `chatbot/medical.py`: MedQuAD assistant.
- `chatbot/knowledge_base.py`: uploaded document assistant.
- `chatbot/vision.py`: Gemini Vision and OCR.
- `chatbot/multilingual.py`: detection and translation flow.
- `chatbot/sentiment.py`: sentiment analysis.
- `chatbot/memory.py`: persistent conversation memory.
- `utils/`: loaders, preprocessing, metrics, logging, visualization.
- `app.py`: Streamlit dashboard and mode router.

## Challenges

- Combining many internship tasks into one cohesive application.
- Keeping retrieval reusable across medical, scientific, and uploaded documents.
- Handling optional local resources such as OCR binaries and NLTK datasets.
- Avoiding hardcoded API keys while keeping setup simple.
- Designing the UI to expose many features without fragmenting the project.

## Solutions

- Created a shared `RAGPipeline` used by MedQuAD, arXiv, and dynamic knowledge base modules.
- Centralized settings in `config.py` and `.env.example`.
- Added defensive error handling and fallbacks for missing optional resources.
- Persisted FAISS indexes to `vector_db/` so retrieval indexes reload automatically.
- Added sample dataset records and tests for reproducibility.

## Results

The final repository is a complete AI assistant with multiple integrated modes. It supports dataset ingestion, document uploads, retrieval, citations, image reasoning, multilingual queries, sentiment-aware tone, memory, and analytics. The structure is suitable for GitHub submission and internship evaluation.

## Learning Outcomes

- Built an end-to-end RAG system with FAISS and Sentence Transformers.
- Integrated Gemini text and vision APIs.
- Designed a modular Streamlit application.
- Implemented persistent memory and user-uploaded knowledge expansion.
- Created testable Python modules with type hints, docstrings, and logging.
- Added professional repository assets such as CI, issue templates, and documentation.

## Conclusion

AI-Assistant-Internship demonstrates a production-style AI assistant that combines conversational AI, retrieval, vision, multilingual support, analytics, and software engineering practices. It can be extended with larger datasets, stronger evaluation, deployment workflows, and user authentication.
