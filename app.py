"""PolicyQA - UI Streamlit. Asistente RAG sobre documentos."""
import os

import streamlit as st

import config
import embeddings as emb
import ingest
from query import load_vectorstore, build_qa

st.set_page_config(page_title="PolicyQA", page_icon="📘", layout="wide")
st.title("📘 PolicyQA — Asistente de políticas y documentos (RAG)")

backend = "Bedrock" if emb.has_bedrock() else "Local (sin AWS)"
st.caption(f"Backend: **{backend}** · Modelo embeddings: "
           f"{config.TITAN_EMBED_MODEL if config.EMBEDDINGS_BACKEND=='bedrock' else 'fastembed bge-small'}")

# ---- Sidebar: ingesta ----
with st.sidebar:
    st.subheader("Ingesta")
    st.write("Docs en `./docs` (PDF/TXT/MD). Sube uno o reindexa:")
    up = st.file_uploader("Subir PDF/MD", type=["pdf", "txt", "md"])
    if up and st.button("Guardar + reindexar"):
        os.makedirs(config.DOCS_DIR, exist_ok=True)
        path = os.path.join(config.DOCS_DIR, up.name)
        with open(path, "wb") as f:
            f.write(up.getbuffer())
        docs = ingest.load_documents(config.DOCS_DIR)
        ingest.build_index(docs, emb.build_embeddings(), config.INDEX_DIR)
        st.success("Documento indexado.")
    if st.button("Reindexar ./docs"):
        docs = ingest.load_documents(config.DOCS_DIR)
        ingest.build_index(docs, emb.build_embeddings(), config.INDEX_DIR)
        st.success("Indexado.")
    st.caption("`EMBEDDINGS_BACKEND=local` para dev, `bedrock` para prod.")

# ---- Chat ----
# cache del vectorstore
@st.cache_resource
def get_vs():
    try:
        return load_vectorstore()
    except SystemExit:
        return None

vs = get_vs()
if vs is None:
    st.warning("No hay índice. Sube un doc o corre `python ingest.py`.")
    st.stop()

answer = build_qa(vs)

if "history" not in st.session_state:
    st.session_state.history = []

for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h["q"])
    with st.chat_message("assistant"):
        st.write(h["a"])

q = st.chat_input("Haz una pregunta sobre los documentos…")
if q:
    st.session_state.history.append({"q": q, "a": answer(q)})
    st.rerun()
