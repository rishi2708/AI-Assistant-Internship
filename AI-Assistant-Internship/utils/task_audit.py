"""Task completion audit for internship requirements."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskStatus:
    """Status for a screenshot requirement."""

    task: str
    status: str
    evidence: str


def internship_task_statuses() -> list[TaskStatus]:
    """Return the integrated project status for the visible internship tasks."""

    return [
        TaskStatus(
            "Dynamic knowledge-base expansion",
            "Complete",
            "Automated scheduled updater, GitHub Actions cron, SHA-256 manifest, incremental changed-file detection, FAISS persistence, and reload support.",
        ),
        TaskStatus(
            "Multi-modal text and image assistant",
            "Complete",
            "Vision mode combines image observations, OCR, text prompts, conversation memory, ambiguity detection, answer validation, and evidence-based final decisions.",
        ),
        TaskStatus(
            "Medical Q&A chatbot using MedQuAD",
            "Complete",
            "MedQuAD loader, RAG retrieval, medical disclaimer, citations, and medical entity recognition.",
        ),
        TaskStatus(
            "Domain expert chatbot using arXiv",
            "Complete",
            "Computer Science arXiv subset filtering, scientific RAG, advanced paper search, open-source summarization fallback, concept graph generation, references, and follow-up questions.",
        ),
        TaskStatus(
            "Sentiment analysis integration",
            "Complete",
            "Complete sentiment-aware pipeline: emotion detection, adaptive prompt generation, response post-processing, tone-alignment evaluation, metrics tracking, and sentiment graphs.",
        ),
        TaskStatus(
            "Multilingual conversations",
            "Complete",
            "Automatic language identification, mixed-language segment detection, English/Hindi/Bengali/Spanish translation workflow, context preservation after language switching, and cross-lingual reasoning prompts.",
        ),
    ]
