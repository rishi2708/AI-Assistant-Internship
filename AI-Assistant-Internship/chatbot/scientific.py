"""Scientific domain expert helpers for arXiv-style data."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class PaperSearchResult:
    """Search result for a scientific paper record."""

    title: str
    score: int
    summary: str
    reference: str
    concepts: list[str]


class ScientificPaperToolkit:
    """Pure-Python paper search, summarization, and concept extraction."""

    STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "using",
        "into",
        "about",
        "paper",
        "study",
        "method",
        "methods",
    }

    def search(self, query: str, documents: list[Document], limit: int = 5) -> list[PaperSearchResult]:
        """Search arXiv documents with keyword overlap."""

        query_terms = self._terms(query)
        results: list[PaperSearchResult] = []
        for document in documents:
            text = document.page_content
            doc_terms = self._terms(text)
            score = len(query_terms & doc_terms)
            if score == 0:
                continue
            title = str(document.metadata.get("title") or self._extract_title(text) or "Untitled paper")
            reference = self._reference(document)
            results.append(
                PaperSearchResult(
                    title=title,
                    score=score,
                    summary=self.summarize(text),
                    reference=reference,
                    concepts=self.extract_concepts(text, limit=8),
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Create a compact extractive summary."""

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        clean = [sentence.strip() for sentence in sentences if sentence.strip()]
        return " ".join(clean[:max_sentences])

    def extract_concepts(self, text: str, limit: int = 10) -> list[str]:
        """Extract frequent domain concepts."""

        terms = [
            term
            for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower())
            if term not in self.STOPWORDS
        ]
        return [term for term, _ in Counter(terms).most_common(limit)]

    def concept_edges(self, text: str, limit: int = 8) -> list[tuple[str, str]]:
        """Build lightweight concept co-occurrence edges for visualization."""

        concepts = self.extract_concepts(text, limit=limit)
        return list(zip(concepts, concepts[1:]))

    def concept_mermaid(self, text: str, limit: int = 8) -> str:
        """Return a Mermaid concept graph for scientific concept visualization."""

        edges = self.concept_edges(text, limit=limit)
        if not edges:
            return "graph TD\n    A[No concepts found]"
        lines = ["graph TD"]
        for left, right in edges:
            lines.append(f"    {self._node(left)}[{left}] --> {self._node(right)}[{right}]")
        return "\n".join(lines)

    def follow_up_questions(self, query: str, results: list[PaperSearchResult]) -> list[str]:
        """Generate follow-up questions grounded in retrieved CS papers."""

        concepts = []
        for result in results:
            concepts.extend(result.concepts[:3])
        unique = list(dict.fromkeys(concepts))
        if not unique:
            unique = list(self._terms(query))[:3]
        return [f"How does {concept} affect the method or result?" for concept in unique[:3]]

    def open_source_summary(self, text: str, model_name: str = "sshleifer/distilbart-cnn-12-6") -> str:
        """Summarize with an open-source Transformers model when available.

        The method falls back to deterministic extractive summarization so the
        project remains reproducible in CI and without downloading large models.
        """

        try:
            from transformers import pipeline

            summarizer = pipeline("summarization", model=model_name)
            output = summarizer(text[:3000], max_length=140, min_length=35, do_sample=False)
            return str(output[0]["summary_text"])
        except Exception:
            return self.summarize(text, max_sentences=4)

    def build_expert_context(self, query: str, documents: list[Document], limit: int = 3) -> dict[str, object]:
        """Build CS-domain expert context for retrieval, summarization, and follow-up."""

        cs_documents = [
            document
            for document in documents
            if any(category.startswith("cs.") for category in str(document.metadata.get("categories", "")).split())
        ]
        results = self.search(query, cs_documents, limit=limit)
        joined = "\n\n".join(result.summary for result in results)
        return {
            "subset": "Computer Science arXiv categories (cs.*)",
            "papers": results,
            "open_source_summary": self.open_source_summary(joined or query),
            "concept_graph": self.concept_mermaid(joined or query),
            "follow_up_questions": self.follow_up_questions(query, results),
        }

    def _terms(self, text: str) -> set[str]:
        """Normalize searchable terms."""

        return {
            term
            for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
            if term not in self.STOPWORDS
        }

    @staticmethod
    def _node(text: str) -> str:
        """Create a Mermaid-safe node id."""

        return re.sub(r"[^A-Za-z0-9_]", "_", text.title())

    @staticmethod
    def _extract_title(text: str) -> str:
        """Extract a title line from arXiv text."""

        match = re.search(r"Title:\s*(.+)", text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _reference(document: Document) -> str:
        """Format paper reference metadata."""

        arxiv_id = document.metadata.get("arxiv_id")
        authors = document.metadata.get("authors")
        title = document.metadata.get("title") or "Untitled paper"
        parts = [str(title)]
        if authors:
            parts.append(f"Authors: {authors}")
        if arxiv_id:
            parts.append(f"arXiv: {arxiv_id}")
        return " | ".join(parts)
