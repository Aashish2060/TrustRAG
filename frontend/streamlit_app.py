import streamlit as st
from config.settings import settings
from src.ingestion.document_store import DocumentStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient, GenerationError
from src.verification.correction_controller import CorrectionController

@st.cache_resource
def get_store():
    return DocumentStore(settings.data_dir, settings.chunk_size, settings.chunk_overlap)

def main():
    st.set_page_config(page_title="TrustRAG Research Lab", page_icon="🔎", layout="wide")
    store = get_store()
    st.title("TrustRAG Research Lab")
    st.caption("Evidence-grounded question answering with transparent verification and bounded self-correction")
    page = st.sidebar.radio("Navigate", ["Overview", "Knowledge Base", "Ask AI", "Evaluation Lab", "Experiment Details"])
    mode_label = "API-backed generation" if settings.llm_base_url and settings.llm_api_key and settings.llm_model else "Local extractive demo"
    if page == "Overview":
        a, b, c = st.columns(3)
        a.metric("Indexed documents", len(store.documents))
        b.metric("Evidence chunks", len(store.chunks))
        c.metric("Generation mode", mode_label)
        st.subheader("Research question")
        st.write("To what extent can evidence-based verification and bounded self-correction improve factual reliability compared with a conventional RAG baseline?")
        st.info("Reliability scores in this prototype are heuristic indicators, not calibrated probabilities. No live evaluation metrics are shown until an experiment is run.")
        st.markdown("**Pipeline:** Upload → Extract → Chunk → Retrieve → Generate → Verify claims → Bounded correction → Final answer or abstention → Evaluate")
    elif page == "Knowledge Base":
        st.subheader("Upload documents")
        uploads = st.file_uploader("Choose PDF, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        if st.button("Index selected documents", type="primary", disabled=not uploads):
            for upload in uploads or []:
                try:
                    doc = store.add(upload.name, upload.getvalue())
                    st.success(f"Indexed {doc['filename']} ({doc['chunk_count']} chunks)")
                except ValueError as exc:
                    st.error(f"{upload.name}: {exc}")
        st.subheader("Indexed documents")
        for doc in store.list():
            left, right = st.columns([5, 1])
            left.write(f"**{doc['filename']}** · {doc['chunk_count']} chunks · {doc['characters']} characters")
            if right.button("Delete", key=f"delete-{doc['document_id']}"):
                store.delete(doc["document_id"])
                st.rerun()
        if store.documents and st.button("Reset knowledge base", type="secondary"):
            store.reset()
            st.rerun()
    elif page == "Ask AI":
        st.subheader("Ask a question about your documents")
        question = st.text_area("Question", placeholder="What do the uploaded documents say about…")
        mode = st.selectbox("Pipeline", ["baseline", "verified", "self_correcting"], index=2)
        top_k = st.slider("Initial retrieved passages", 1, 10, settings.top_k)
        if st.button("Run TrustRAG", type="primary", disabled=len(question.strip()) < 3):
            try:
                result = CorrectionController(Retriever(store, settings.min_relevance), LLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model), settings.max_corrections).run(question, top_k, mode)
                st.subheader("Final answer")
                st.write(result["final_answer"])
                st.caption(f"Status: {result['answer_status']} · Heuristic reliability: {result['reliability_score']:.3f} · Latency: {result['latency_seconds']:.3f}s")
                with st.expander("Initial answer"):
                    st.write(result["initial_answer"])
                with st.expander("Claim verification"):
                    st.dataframe(result["claims"], width="stretch")
                with st.expander("Retrieved evidence"):
                    for item in result["evidence"]:
                        st.markdown(f"**{item['filename']}** · page {item.get('page') or 'N/A'} · chunk `{item['chunk_id']}` · similarity {item.get('similarity', 0):.3f}")
                        st.write(item["text"])
                with st.expander("Correction audit trail"):
                    st.json(result["correction_attempts"])
                st.download_button("Download result JSON", data=__import__('json').dumps(result, indent=2), file_name="trustrag_query_result.json", mime="application/json")
            except GenerationError as exc:
                st.error(str(exc))
    elif page == "Evaluation Lab":
        st.subheader("Reproducible evaluation")
        st.write("The starter evaluation uses a small synthetic dataset. Its results are suitable only for smoke testing, not broad research claims.")
        if st.button("Run starter evaluation", type="primary"):
            import subprocess, sys
            proc = subprocess.run([sys.executable, "scripts/run_evaluation.py"], capture_output=True, text=True)
            if proc.returncode == 0:
                st.success("Evaluation completed")
                st.code(proc.stdout)
                st.caption("Files written to reports/. See README for metric limitations.")
            else:
                st.error("Evaluation failed")
                st.code(proc.stderr)
    else:
        st.subheader("Experiment configuration")
        st.json({"generation_mode": mode_label, "embedding_retrieval": "TF-IDF lexical retrieval", "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap, "top_k_default": settings.top_k, "max_corrections": settings.max_corrections, "minimum_relevance": settings.min_relevance, "api_configured": bool(settings.llm_base_url and settings.llm_api_key and settings.llm_model)})
        st.warning("The included verifier uses lexical overlap and cannot reliably infer semantic entailment or contradictions. Use a validated verifier and human-labelled test data before making scientific claims.")

if __name__ == "__main__":
    main()
