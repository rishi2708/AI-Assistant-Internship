import json
from pathlib import Path

from langchain_core.documents import Document

from chatbot.knowledge_updater import KnowledgeBaseUpdater
from chatbot.multilingual_context import ContextPreservingMultilingualAssistant
from chatbot.multimodal_reasoning import MultimodalReasoner
from chatbot.scientific import ScientificPaperToolkit
from chatbot.sentiment_pipeline import SentimentAwareResponder


class DummyKnowledgeBase:
    def __init__(self):
        self.files = []

    def add_file(self, path):
        self.files.append(Path(path).name)
        return 2


def test_incremental_updater_uses_hash_manifest(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("first version", encoding="utf-8")
    sources_file = tmp_path / "sources.json"
    sources_file.write_text(
        json.dumps({"sources": [{"path": str(source), "enabled": True}]}),
        encoding="utf-8",
    )
    updater = KnowledgeBaseUpdater(DummyKnowledgeBase(), sources_file, tmp_path / "manifest.json")

    first = updater.update()
    second = updater.update()
    source.write_text("second version", encoding="utf-8")
    third = updater.update()

    assert first["chunks_added"] == 2
    assert second["chunks_added"] == 0
    assert third["chunks_added"] == 2


def test_multimodal_reasoner_returns_evidence_ambiguity_and_validation():
    result = MultimodalReasoner().reason(
        user_text="Is the chart showing blue bars?",
        generated_answer="The chart appears to show blue bars based on the image.",
        image_observations="The image contains blue bars and axis labels.",
        ocr_text="Revenue by month",
        prior_context=["Earlier question asked about chart colors."],
    )

    assert result.evidence
    assert "Response validation" in result.answer
    assert result.decision == "answer"


def test_scientific_toolkit_filters_cs_context_and_followups():
    docs = [
        Document(
            page_content="Title: Neural Retrieval\nAbstract: Dense retrieval improves question answering.",
            metadata={"title": "Neural Retrieval", "categories": "cs.IR cs.CL", "arxiv_id": "1"},
        ),
        Document(
            page_content="Title: Biology Paper\nAbstract: Cells and proteins.",
            metadata={"title": "Biology Paper", "categories": "q-bio.BM", "arxiv_id": "2"},
        ),
    ]
    context = ScientificPaperToolkit().build_expert_context("retrieval question answering", docs)

    assert context["subset"] == "Computer Science arXiv categories (cs.*)"
    assert len(context["papers"]) == 1
    assert "graph TD" in context["concept_graph"]
    assert context["follow_up_questions"]


def test_sentiment_pipeline_adapts_and_evaluates_negative_response():
    responder = SentimentAwareResponder()
    result = responder.generate("I am angry about this problem", lambda prompt: "Here is the fix.")

    assert result.sentiment.label == "Negative"
    assert result.response.startswith("I understand")
    assert "aligned" in result.evaluation


def test_multilingual_context_preserves_turns(monkeypatch):
    assistant = ContextPreservingMultilingualAssistant()
    monkeypatch.setattr(assistant.translator, "detect_language", lambda text: "es" if "hola" in text.lower() else "en")
    monkeypatch.setattr(assistant.translator, "to_english", lambda text, lang: "hello" if lang == "es" else text)
    monkeypatch.setattr(assistant.translator, "from_english", lambda text, lang: text)

    first = assistant.answer("hola", lambda prompt: "Hello response")
    second = assistant.answer("What did I say?", lambda prompt: prompt)

    assert first.detected_language == "es"
    assert len(assistant.state.turns) == 2
    assert "Previous cross-lingual context" in second.answer
