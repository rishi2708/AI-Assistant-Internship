# Evaluator Remediation Evidence

This document maps the evaluator failure reasons to concrete repository changes.

## Task 1: Dynamic Knowledge Expansion

- Added `schedule_knowledge_updates.py` for continuous or one-shot scheduled updates.
- Added `.github/workflows/scheduled-knowledge-update.yml` with a six-hour cron and manual trigger.
- Upgraded `chatbot/knowledge_updater.py` to use SHA-256 fingerprints, persisted manifests, changed-file detection, and incremental indexing state.
- Added tests proving first run indexes, second unchanged run skips, and changed file re-indexes.

## Task 2 and 3: Multi-Modal Reasoning

- Added `chatbot/multimodal_reasoning.py`.
- Vision now combines text prompt, image answer, OCR text, and prior chat context.
- The reasoner extracts evidence, detects ambiguity/conflict, validates answer grounding, and returns a decision.
- Added tests validating evidence, ambiguity handling, and validation output.

## Task 4: Computer Science arXiv Domain Expert

- Added `data/arxiv/computer_science_subset.jsonl`.
- Updated arXiv loading to support `category_prefix="cs."`.
- Added Computer Science expert context generation, open-source Transformers summarization fallback, concept graph generation, references, and follow-up question generation.
- Scientific mode now retrieves from the Computer Science subset and injects summaries/follow-ups into the RAG prompt.

## Task 5: Sentiment-Aware Conversational Pipeline

- Added `chatbot/sentiment_pipeline.py`.
- Pipeline performs sentiment detection, prompt adaptation, response post-processing, and tone-alignment evaluation.
- Streamlit general mode now returns sentiment pipeline evaluation evidence.
- Added tests for negative-emotion adaptation and evaluation.

## Task 6: Multilingual Context Preservation

- Added `chatbot/multilingual_context.py`.
- Multilingual mode now preserves translated English context across language switching and uses that context in follow-up prompts.
- Existing language detection and translation are retained for English, Hindi, Bengali, and Spanish.
- Added tests proving language switching preserves conversation turns.
