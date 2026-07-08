from langchain_core.documents import Document

from utils.text_processing import ChunkingConfig, chunk_documents, chunk_text, clean_text


def test_clean_text_normalizes_whitespace():
    assert clean_text("Hello\n\n   world\t!") == "Hello world !"


def test_chunk_text_produces_overlapping_chunks():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_text(text, ChunkingConfig(chunk_size=28, chunk_overlap=8))

    assert len(chunks) >= 2
    assert all(len(chunk) <= 36 for chunk in chunks)


def test_chunk_documents_preserves_metadata():
    docs = [Document(page_content="Alpha. Beta. Gamma.", metadata={"source": "unit"})]
    chunks = chunk_documents(docs, ChunkingConfig(chunk_size=20, chunk_overlap=4))

    assert chunks
    assert chunks[0].metadata["source"] == "unit"
    assert chunks[0].metadata["doc_index"] == 0
