"""Streamlit dashboard for the AI Assistant Internship project."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from chatbot.base_chatbot import GeminiChatbot
from chatbot.memory import ConversationMemory
from chatbot.sentiment import SentimentAnalyzer
from config import (
    ARXIV_DATA_DIR,
    DEFAULT_SYSTEM_PROMPT,
    SCIENTIFIC_SYSTEM_PROMPT,
    UPLOAD_DIR,
    VECTOR_DB_DIR,
    AppConfig,
    ensure_directories,
)
from utils.logging import configure_logging
from utils.metrics import MetricsTracker
from utils.task_audit import internship_task_statuses
from utils.visualization import dataset_statistics_chart, latency_chart, sentiment_distribution


def init_state() -> None:
    """Initialize Streamlit session state."""

    ensure_directories()
    config = AppConfig.from_env()
    configure_logging(config.log_level)
    st.session_state["config"] = config
    st.session_state.setdefault("memory", ConversationMemory(VECTOR_DB_DIR / "conversation_memory.json"))
    st.session_state.setdefault("metrics", MetricsTracker())
    st.session_state.setdefault("messages", [])


def get_chatbot(system_prompt: str, temperature: float) -> GeminiChatbot:
    """Create a chatbot using current settings."""

    return GeminiChatbot(
        st.session_state.config,
        st.session_state.memory,
        system_prompt=system_prompt,
        temperature=temperature,
    )


def render_sidebar() -> dict[str, object]:
    """Render sidebar controls."""

    with st.sidebar:
        st.title("AI Assistant")
        mode = st.selectbox(
            "Assistant mode",
            ["General", "Medical", "Scientific", "Knowledge Base", "Vision", "Multilingual"],
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
        top_k = st.slider("Top-k retrieval", 1, 10, st.session_state.config.top_k)
        system_prompt = st.text_area("System prompt", DEFAULT_SYSTEM_PROMPT, height=110)

        st.divider()
        st.subheader("Uploads")
        documents = st.file_uploader(
            "Upload PDF, TXT, MD, or DOCX",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )
        image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Task Audit")
        if st.button("Show task status", use_container_width=True):
            st.session_state["show_task_audit"] = not st.session_state.get("show_task_audit", False)

        chat_export = st.session_state.memory.export_markdown()
        st.download_button(
            "Download chat",
            chat_export,
            file_name="ai_assistant_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    return {
        "mode": mode,
        "temperature": temperature,
        "top_k": top_k,
        "system_prompt": system_prompt,
        "documents": documents,
        "image": image,
    }


def save_uploaded_file(uploaded_file) -> Path:
    """Persist a Streamlit upload to the uploads directory."""

    destination = UPLOAD_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def ingest_sidebar_documents(files, chatbot: GeminiChatbot) -> None:
    """Ingest documents uploaded from the sidebar."""

    if not files:
        return
    from chatbot.knowledge_base import DynamicKnowledgeBase

    kb = DynamicKnowledgeBase(st.session_state.config, chatbot)
    total_chunks = 0
    for uploaded_file in files:
        path = save_uploaded_file(uploaded_file)
        total_chunks += kb.add_file(path)
    st.sidebar.success(f"Knowledge base updated with {total_chunks} chunks.")


def bootstrap_scientific(chatbot: GeminiChatbot):
    """Create and lazily bootstrap the arXiv RAG pipeline."""

    from chatbot.rag import RAGPipeline
    from utils.document_loader import load_arxiv_documents

    pipeline = RAGPipeline(st.session_state.config, chatbot, VECTOR_DB_DIR, "scientific_arxiv")
    if not pipeline.store.load():
        documents = load_arxiv_documents(ARXIV_DATA_DIR, max_records=5000)
        if documents:
            pipeline.ingest_documents(documents)
    return pipeline


def answer_prompt(prompt: str, controls: dict[str, object]) -> str:
    """Route a user prompt to the selected assistant mode."""

    mode = str(controls["mode"])
    temperature = float(controls["temperature"])
    top_k = int(controls["top_k"])
    system_prompt = str(controls["system_prompt"])

    sentiment = SentimentAnalyzer().analyze(prompt)
    st.session_state.metrics.add_sentiment(sentiment.label)
    prompt_with_tone = f"{prompt}\n\nTone guidance: {sentiment.tone_instruction}"

    if mode == "Medical":
        from chatbot.medical import MedicalChatbot

        chatbot = get_chatbot(system_prompt, temperature)
        medical = MedicalChatbot(st.session_state.config, chatbot)
        medical.bootstrap()
        return str(medical.answer(prompt_with_tone, top_k=top_k)["answer"])

    if mode == "Scientific":
        chatbot = get_chatbot(SCIENTIFIC_SYSTEM_PROMPT, temperature)
        try:
            from chatbot.scientific import ScientificPaperToolkit
            from utils.document_loader import load_arxiv_documents

            papers = load_arxiv_documents(ARXIV_DATA_DIR, max_records=5000)
            toolkit = ScientificPaperToolkit()
            paper_results = toolkit.search(prompt, papers, limit=3)
            if paper_results:
                paper_context = "\n\n".join(
                    f"Paper: {item.title}\nSummary: {item.summary}\nReference: {item.reference}\n"
                    f"Concepts: {', '.join(item.concepts)}"
                    for item in paper_results
                )
                prompt_with_tone = f"{prompt_with_tone}\n\nScientific paper search context:\n{paper_context}"
        except Exception:
            pass
        pipeline = bootstrap_scientific(chatbot)
        return str(pipeline.answer(prompt_with_tone, mode="Scientific", top_k=top_k)["answer"])

    if mode == "Knowledge Base":
        from chatbot.knowledge_base import DynamicKnowledgeBase

        chatbot = get_chatbot(system_prompt, temperature)
        kb = DynamicKnowledgeBase(st.session_state.config, chatbot)
        return str(kb.query(prompt_with_tone, top_k=top_k)["answer"])

    if mode == "Multilingual":
        from chatbot.multilingual import MultilingualAssistant

        chatbot = get_chatbot(system_prompt, temperature)
        multilingual = MultilingualAssistant()

        def _answer_fn(english_query: str) -> str:
            return chatbot.generate_response(f"{english_query}\n\nTone guidance: {sentiment.tone_instruction}")

        result = multilingual.answer(prompt, _answer_fn)
        segments = ", ".join(f"{item['language']}: {item['text']}" for item in result.language_segments)
        return f"{result.answer}\n\nDetected language: {result.detected_language}\n\nLanguage segments: {segments}"

    chatbot = get_chatbot(system_prompt, temperature)
    return chatbot.generate_response(prompt_with_tone, mode=mode)


def render_chat(controls: dict[str, object]) -> None:
    """Render chat messages and handle new input."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about medicine, science, documents, images, or general questions")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.session_state.metrics.track_latency():
                answer = answer_prompt(prompt, controls)
            placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def render_vision(controls: dict[str, object]) -> None:
    """Render image understanding workflow."""

    image_file = controls.get("image")
    if not image_file:
        st.info("Upload an image in the sidebar to use Gemini Vision.")
        return

    path = save_uploaded_file(image_file)
    st.image(str(path), caption=image_file.name, use_column_width=True)
    prompt = st.text_area("Image question", "Describe the image and extract important text.")
    if st.button("Analyze image", type="primary"):
        from chatbot.vision import GeminiVisionAssistant

        vision = GeminiVisionAssistant(st.session_state.config, float(controls["temperature"]))
        with st.session_state.metrics.track_latency():
            answer = vision.analyze_image(path, prompt)
        st.markdown(answer)
        st.session_state.memory.add("user", f"[Image: {image_file.name}] {prompt}", "Vision")
        st.session_state.memory.add("assistant", answer, "Vision")


def render_dashboard() -> None:
    """Render analytics visualizations."""

    with st.expander("Analytics and Evaluation", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sentiment_distribution(st.session_state.metrics.sentiment_labels), use_container_width=True)
            st.plotly_chart(latency_chart(st.session_state.metrics.latencies), use_container_width=True)
        with col2:
            stats = {
                "MedQuAD files": len(list((Path("data") / "medical").glob("*"))),
                "arXiv files": len(list((Path("data") / "arxiv").glob("*"))),
                "Uploads": len(list(UPLOAD_DIR.glob("*"))),
            }
            st.plotly_chart(dataset_statistics_chart(stats), use_container_width=True)
            st.json(st.session_state.metrics.latency_summary())


def render_task_audit() -> None:
    """Render internship task completion evidence."""

    if not st.session_state.get("show_task_audit", False):
        return
    with st.expander("Internship Task Completion Audit", expanded=True):
        for item in internship_task_statuses():
            st.markdown(f"**{item.task}** - {item.status}")
            st.caption(item.evidence)


def main() -> None:
    """Application entry point."""

    st.set_page_config(page_title="AI Assistant Internship", page_icon=":robot_face:", layout="wide")
    init_state()
    st.markdown(
        """
        <style>
        .stApp { background: #0f172a; color: #e5e7eb; }
        section[data-testid="stSidebar"] { background: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Assistant Internship")
    st.caption("General chatbot, MedQuAD medical RAG, arXiv scientific RAG, dynamic knowledge base, vision, multilingual support, sentiment, and memory.")

    controls = render_sidebar()
    chatbot = get_chatbot(str(controls["system_prompt"]), float(controls["temperature"]))
    ingest_sidebar_documents(controls.get("documents"), chatbot)

    if controls["mode"] == "Vision":
        render_vision(controls)
    else:
        render_chat(controls)

    render_task_audit()
    render_dashboard()


if __name__ == "__main__":
    main()
