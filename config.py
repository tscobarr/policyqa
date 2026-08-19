# ============================================================
#  PolicyQA - Asistente RAG de preguntas sobre documentos
#  Stack: Amazon Bedrock (Titan embeddings + Claude) | LangChain
#         | FAISS | Streamlit | S3 (opcional)
#
#  Dual-mode:
#    - EMBEDDINGS_BACKEND=bedrock -> Titan embeddings (prod, CV)
#    - EMBEDDINGS_BACKEND=local   -> fastembed (dev sin AWS)
# ============================================================

import os

# ------- Rutas -------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")        # documentos fuentes (PDF/TXT/MD)
INDEX_DIR = os.path.join(BASE_DIR, "index")      # indice FAISS local
CATALOG_DIR = os.path.join(BASE_DIR, "catalog")  # catalogo Parquet (equivalente Athena)

# ------- Backends -------
# Auto-detect: si hay credenciales AWS y se pide bedrock, usa Bedrock.
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "bedrock").lower()

# ------- Bedrock -------
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
TITAN_EMBED_MODEL = "amazon.titan-embed-text-v1"
# LLM: Nova-lite (accesible en esta cuenta). Alternativas: amazon.nova-micro-v1:0,
# meta.llama3-8b-instruct-v1:0, mistral.mistral-7b-instruct-v0:2
LLM_MODEL = os.getenv("LLM_MODEL", "amazon.nova-lite-v1:0")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ------- S3 (opcional) -------
S3_BUCKET = os.getenv("S3_BUCKET", "")     # si se setea, sube/baja indice a S3
S3_INDEX_PREFIX = "policyqa"

# ------- Chunking -------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))       # documentos recuperados por query

# ------- Prompt del asistente -------
SYSTEM_PROMPT = """Eres PolicyQA, un asistente interno de una entidad financiera.
Respondes preguntas de agentes de servicio basandote EXCLUSIVAMENTE en las politicas,
manuales y procedimientos (SOP) que tienes como contexto.

Reglas:
1. Usa solo el contexto dado. Si no encuentras la respuesta, di:
   "No encuentro eso en los documentos cargados."
2. Cita la seccion o documento fuente cuando sea posible.
3. Responde breve y directo, en el idioma de la pregunta.
4. Las preguntas de metadato (cuantos documentos hay, sus nombres) se responden
   usando la linea "[Documentos cargados: N] nombres" del contexto; ese dato es valido.
"""
