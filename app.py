"""Streamlit chat UI for the p53 RAG pipeline.

Run with:  streamlit run app.py
"""

import streamlit as st

import config
import rag

st.set_page_config(page_title="p53 RAG", layout="wide")
st.title("🧬 p53 Tumor Suppressor — Literature Q&A")
st.caption(
    f"Retrieval-augmented over 100 PubMed abstracts · "
    f"embeddings: `{config.EMBED_MODEL}` · chat: `{config.CHAT_MODEL}` · all local via Ollama"
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 2, 12, config.TOP_K)
    st.markdown("---")
    try:
        count = rag.get_collection().count()
        st.success(f"Index ready: {count} chunks")
    except Exception:
        st.error("No index found. Run `python build_index.py` first.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['pmid']}] {s['title']} — [{s['url']}]({s['url']})")

if prompt := st.chat_input("Ask about p53..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            stream, hits = rag.answer(prompt, top_k=top_k, stream=True)
            full = st.write_stream(stream)

        # Dedupe sources by PMID for display.
        seen, sources = set(), []
        for h in hits:
            m = h["meta"]
            if m["pmid"] in seen:
                continue
            seen.add(m["pmid"])
            sources.append({"pmid": m["pmid"], "title": m["title"], "url": m["url"]})

        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- [{s['pmid']}] {s['title']} — [{s['url']}]({s['url']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": full, "sources": sources}
    )
