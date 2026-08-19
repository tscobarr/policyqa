"""Ingesta de PolicyQA: PDFs/documentos -> chunks -> embeddings -> indice FAISS.

Uso:
    python ingest.py                 # indexa todo lo de ./docs en ./index
    python ingest.py --s3            # ademas sube el indice a S3 (si S3_BUCKET set)
"""
import argparse
import os

import config
from embeddings import build_embeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(docs_dir):
    """Carga todos los PDF/TXT/MD de docs_dir como documentos LangChain."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    docs = []
    for root, _, files in os.walk(docs_dir):
        for fname in sorted(files):
            path = os.path.join(root, fname)
            try:
                if fname.lower().endswith(".pdf"):
                    docs.extend(PyPDFLoader(path).load())
                elif fname.lower().endswith((".txt", ".md", ".markdown")):
                    docs.extend(TextLoader(path, encoding="utf-8").load())
                else:
                    print(f"  [skip] {fname} (formato no soportado)")
            except Exception as e:
                print(f"  [error] {fname}: {e}")
    print(f"Cargados {len(docs)} documentos.")
    return docs


def build_index(docs, embeddings, persist_dir):
    from langchain_community.vectorstores import FAISS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunks tras el split.")

    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_dir, index_name="policyqa")
    print(f"Indice FAISS guardado en {persist_dir}")
    return vectorstore


def upload_to_s3(persist_dir):
    import boto3

    if not config.S3_BUCKET:
        print("S3_BUCKET no configurado -> no subo a S3.")
        return
    s3 = boto3.client("s3", region_name=config.BEDROCK_REGION)
    for fname in ("policyqa.faiss", "policyqa.pkl"):
        key = f"{config.S3_INDEX_PREFIX}/{fname}"
        s3.upload_file(os.path.join(persist_dir, fname), config.S3_BUCKET, key)
        print(f"Subido s3://{config.S3_BUCKET}/{key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3", action="store_true", help="sube indice a S3")
    args = parser.parse_args()

    if not os.path.isdir(config.DOCS_DIR):
        os.makedirs(config.DOCS_DIR, exist_ok=True)
        print(f"[!] Creé {config.DOCS_DIR}: pon tus PDFs/politicas ahi y corre de nuevo.")

    docs = load_documents(config.DOCS_DIR)
    if not docs:
        print("[!] No hay documentos que indexar.")
        return

    embeddings = build_embeddings()
    build_index(docs, embeddings, config.INDEX_DIR)
    if args.s3:
        upload_to_s3(config.INDEX_DIR)
    print("Ingesta completa.")


if __name__ == "__main__":
    main()
