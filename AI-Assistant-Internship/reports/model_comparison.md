# Model Comparison

This project compares a simple baseline assistant with the implemented advanced assistant.

| Capability | Baseline chatbot | Advanced integrated assistant |
| --- | --- | --- |
| General conversation | Direct LLM prompt | Gemini with system prompt, temperature, memory, and safe error handling |
| Medical Q&A | Direct LLM prompt | MedQuAD retrieval, entity recognition, citations, and disclaimer |
| Scientific Q&A | Direct LLM prompt | arXiv retrieval, paper summaries, references, and concept extraction |
| Knowledge updates | Static prompt | Upload ingestion plus configured source updater |
| Image understanding | Unsupported | Gemini Vision with OCR context and image + text reasoning |
| Multilingual support | English only | Detection, translation, mixed-language segments, and context retention |
| Sentiment response | Fixed tone | Positive/negative/neutral detection with tone adaptation |
| Evaluation | Manual checking | Latency, sentiment distribution, dataset stats, confusion matrix utility, task audit |

## Selected Approach

The advanced assistant was selected because it reduces hallucination risk through retrieval, supports more user workflows, and provides auditable evidence through references and visual metrics.

## Model Selection Notes

- LLM: Gemini Flash is used for fast response generation and multimodal support.
- Embeddings: `all-MiniLM-L6-v2` is used because it is compact, fast, and suitable for semantic search.
- Vector database: FAISS is used because it is local, efficient, and reproducible.
- Sentiment baseline: rule-based lexical sentiment is used as a robust fallback when compiled NLP dependencies are unavailable.
