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
            "Upload ingestion, persisted vector index, reload support, and configured source updater.",
        ),
        TaskStatus(
            "Multi-modal text and image assistant",
            "Complete",
            "Vision mode supports image upload, OCR context, text prompt, evidence-first reasoning, and memory.",
        ),
        TaskStatus(
            "Medical Q&A chatbot using MedQuAD",
            "Complete",
            "MedQuAD loader, RAG retrieval, medical disclaimer, citations, and medical entity recognition.",
        ),
        TaskStatus(
            "Domain expert chatbot using arXiv",
            "Complete",
            "arXiv loader, scientific RAG, paper search, summaries, references, and concept extraction.",
        ),
        TaskStatus(
            "Sentiment analysis integration",
            "Complete",
            "Positive/negative/neutral detection, tone adaptation, metrics tracking, and sentiment graphs.",
        ),
        TaskStatus(
            "Multilingual conversations",
            "Complete",
            "Language detection, mixed-language segment detection, translation workflow, and context preservation.",
        ),
    ]
