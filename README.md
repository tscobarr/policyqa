# 📘 PolicyQA — Asistente RAG de preguntas sobre documentos

Bot de preguntas-respuestas sobre **políticas, manuales y procedimientos** (PDF/MD/TXT)
usando **Retrieval-Augmented Generation (RAG)** sobre **Amazon Bedrock**.

> **Problema que resuelve:** un banco/fintech tiene cientos de páginas de políticas
> y SOP. Sus agentes de servicio pierden tiempo leyendo manuales para responder
> consultas de clientes. PolicyQA indexa esos documentos y responde en lenguaje
> natural, citando la fuente.

---

## Arquitectura

```
┌─────────────────────────── SERVERLESS (AWS) ────────────────────────────┐
│  S3 (raw docs) ─► AWS Glue ETL (PySpark) ─► Parquet catalog + Data Catalog │
│                                                        │                 │
│                                        Amazon Athena (SQL sobre el catálogo)│
└──────────────────────────────┬────────────────────────┘                 │
                               │  catálogo Parquet (local: catalog/)      │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                            PIPELINE RAG                                │
│  ingest.py: docs -> chunk -> embeddings (Titan/fastembed) -> FAISS      │
│  query/app: recupera top-k + contexto -> Claude (Bedrock) -> respuesta  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Capa serverless (cubre Bedrock + Athena + Glue):**
- **AWS Glue** (`glue/etl_job.py`): job PySpark que lee docs crudos de S3, limpia,
  genera señales (keyword flags), escribe Parquet y registra tabla en el Glue Data Catalog.
- **Amazon Athena** (`athena/table.ddl.sql` + `athena_query.py`): SQL serverless
  sobre el catálogo (`SELECT ... WHERE has_reembolso`).
- **Local equivalente (sin AWS, verificable):** `catalog.py` + `query_catalog.py`
  (DuckDB sobre Parquet) — misma lógica de consulta.

**Capa RAG (embeddings + generación):**
- **Embeddings:** Amazon Titan `amazon.titan-embed-text-v1` (o `fastembed` en modo local/dev)
- **Vector store:** FAISS (local; opcional S3 para compartir)
- **LLM:** Amazon Bedrock — Claude (si está disponible) o **Amazon Nova-lite** (desplegado, accesible en esta cuenta)
- **Orquestación:** LangChain (`RecursiveCharacterTextSplitter`, LCEL RAG chain)
- **UI:** Streamlit · **Chunking:** 1000 chars / overlap 150 · **Top-K:** 5

## Requisitos

- Python 3.10+
- Modo prod: cuenta AWS con **Amazon Bedrock** habilitado (acceso a modelos Titan + Claude)

## Instalación

```bash
cd policyqa
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido (modo local, sin AWS)

```bash
export EMBEDDINGS_BACKEND=local    # Windows: set EMBEDDINGS_BACKEND=local
python ingest.py                   # indexa ./docs -> ./index
python query.py "¿Cuál es la política de reembolso?"
streamlit run app.py               # UI web
```

> Modo local usa `fastembed` (embedding ONNX ligero, sin AWS) y una respuesta
> **extractiva** (devuelve el contexto recuperado, sin LLM). Sirve para probar la
> tubería. Para respuestas generadas por LLM, usá el modo Bedrock.

## Uso en producción (Bedrock)

```bash
# 1) Configurá credenciales AWS (default profile / AWS_ACCESS_KEY_ID...)
export AWS_REGION=us-east-1
export EMBEDDINGS_BACKEND=bedrock
# opcional: compartir índice en S3
export S3_BUCKET=mi-bucket

# 2) Indexá
python ingest.py --s3

# 3) Respondé / serví
python query.py "qué requisitos hay para abrir una cuenta?"
streamlit run app.py
```

> Requerís que tu rol AWS tenga permisos `bedrock:InvokeModel` (`amazon.titan-embed-text-v1`
> y `anthropic.claude-3-haiku-20240307-v1`) y acceso a Bedrock en la región elegida.

## Estructura

```
policyqa/
├── config.py          # rutas, modelos, chunking, prompt
├── embeddings.py      # backend pluggable: bedrock | local
├── ingest.py          # doc -> chunks -> embeddings -> FAISS
├── query.py           # CLI RAG (recupera + respuesta)
├── app.py             # UI Streamlit
├── catalog.py         # (local) docs -> catálogo Parquet  [equiv. Glue]
├── query_catalog.py   # (local) SQL DuckDB sobre el catálogo [equiv. Athena]
├── athena_query.py    # (AWS) SQL Amazon Athena sobre Data Catalog
├── glue/etl_job.py    # (AWS) job Glue PySpark serverless
├── athena/table.ddl.sql  # (AWS) DDL tabla externa Athena
├── docs/              # documentos fuente (pon tus PDFs acá)
├── catalog/           # catálogo Parquet generado
└── index/             # índice FAISS generado
```

## Correr la capa analítica (Glue/Athena)

```bash
# Local (sin AWS) — verificar la tubería:
python catalog.py                   # docs -> catalog/policyqa_docs.parquet
python query_catalog.py             # SQL DuckDB sobre el catálogo
python query_catalog.py "SELECT source FROM docs WHERE has_prestamo"

# AWS (producción):
# 1) Registrá el job Glue (ver encabezado de glue/etl_job.py) y corrélo
# 2) Creá la tabla Athena (ver athena/table.ddl.sql)
# 3) Consultá:
python athena_query.py "SELECT source, n_chars FROM policyqa_docs WHERE has_reembolso = true"
```

## Lo que demuestra (para CV)

Pipeline RAG completo y desplegable: ingesta multiformato (PDF/MD/TXT), chunking,
embeddings y recuperación semántica con FAISS, generación con LLM vía Bedrock,
backend desacoplado (prod/local) y UI de chat.
