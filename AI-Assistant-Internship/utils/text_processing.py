"""Text normalization, tokenization, and chunking helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for text chunking."""

    chunk_size: int = 900
    chunk_overlap: int = 150


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_tokenize(text: str) -> list[str]:
    """Tokenize text into sentences with a safe fallback."""

    try:
        import nltk

        return nltk.sent_tokenize(text)
    except Exception:
        return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[str]:
    """Split text into overlapping chunks while preserving sentence boundaries."""

    cfg = config or ChunkingConfig()
    normalized = clean_text(text)
    if not normalized:
        return []

    sentences = sentence_tokenize(normalized)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > cfg.chunk_size:
            chunks.extend(_split_long_sentence(sentence, cfg.chunk_size, cfg.chunk_overlap))
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= cfg.chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            overlap = current[-cfg.chunk_overlap :] if cfg.chunk_overlap and current else ""
            current = f"{overlap} {sentence}".strip()

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _split_long_sentence(sentence: str, chunk_size: int, overlap: int) -> list[str]:
    """Split very long sentences by character window."""

    windows: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(sentence):
        windows.append(sentence[start : start + chunk_size].strip())
        start += step
    return [window for window in windows if window]


def documents_from_records(
    records: Iterable[dict[str, str]],
    text_key: str,
    source_key: str = "source",
) -> list[Document]:
    """Create LangChain documents from dictionary records."""

    documents: list[Document] = []
    for index, record in enumerate(records):
        text = clean_text(str(record.get(text_key, "")))
        if not text:
            continue
        metadata = {key: value for key, value in record.items() if key != text_key}
        metadata.setdefault("source", record.get(source_key, f"record-{index}"))
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def chunk_documents(
    documents: Iterable[Document],
    config: ChunkingConfig | None = None,
) -> list[Document]:
    """Chunk LangChain documents and retain metadata."""

    chunks: list[Document] = []
    for doc_index, document in enumerate(documents):
        for chunk_index, chunk in enumerate(chunk_text(document.page_content, config)):
            metadata = dict(document.metadata)
            metadata["doc_index"] = doc_index
            metadata["chunk_index"] = chunk_index
            chunks.append(Document(page_content=chunk, metadata=metadata))
    return chunks
