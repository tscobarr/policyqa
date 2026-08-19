"""Consulta CLI de PolicyQA: RAG sobre el indice FAISS.

Uso:
    python query.py "¿Cual es la politica de reembolso?"   (interactivo si sin args)
"""
import os
import sys

import config
import embeddings as emb


def load_vectorstore():
    from langchain_community.vectorstores import FAISS

    if not os.path.isdir(config.INDEX_DIR):
        sys.exit(f"[!] No hay indice en {config.INDEX_DIR}. Corri primero: python ingest.py")
    return FAISS.load_local(
        config.INDEX_DIR,
        index_name="policyqa",
        embeddings=emb.build_embeddings(),
        allow_dangerous_deserialization=True,
    )


def doc_inventory():
    """Inventario de documentos disponibles (para responder preguntas de metadato)."""
    files = sorted(
        f for f in os.listdir(config.DOCS_DIR)
        if f.lower().endswith((".md", ".txt", ".pdf"))
    )
    if not files:
        return ""
    return "[Documentos cargados: %d] %s" % (len(files), ", ".join(files))


def build_qa(vectorstore):
    """RAG: recupera top-k del indice y responde con el LLM de Bedrock.
    Sin Bedrock -> modo extractivo local (dev)."""
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": config.TOP_K}
    )

    if not emb.has_bedrock():
        print("[modo local] Sin credenciales AWS -> respuesta extractiva (no LLM).\n")
        return lambda q: emb.answer_from_sources(retriever.invoke(q), q)

    def invoke(q):
        docs = retriever.invoke(q)
        chunks = "\n\n".join(d.page_content for d in docs)
        inv = doc_inventory()
        context = (inv + "\n\n" + chunks) if inv else chunks
        prompt = emb.build_prompt(config.SYSTEM_PROMPT, context, q)
        return emb.invoke_llm(prompt)

    return invoke


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else None
    vs = load_vectorstore()
    answer = build_qa(vs)

    if question:
        print(f"\nQ: {question}\nA: {answer(question)}\n")
        return

    print(f"PolicyQA - LLM {config.LLM_MODEL}. Escribe tu pregunta (Ctrl+C para salir)\n")
    while True:
        q = input("Pregunta > ").strip()
        if not q:
            continue
        print(f"A: {answer(q)}\n")


if __name__ == "__main__":
    main()
