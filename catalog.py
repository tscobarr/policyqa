"""Catalogo de documentos en Parquet (equivalente local a un ETL Glue).

Genera catalog/policyqa_docs.parquet con metadatos + senales por documento:
source, path, ext, size, n_chars y banderas has_<keyword>.

En AWS esto lo hace el job de Glue (glue/etl_job.py) leyendo desde S3;
el Parquet resultante se consulta con Athena (ver query_catalog.py / athena_query.py).

Uso:
    python catalog.py
"""
import os

import config

try:
    import pandas as pd
except ImportError:
    pd = None

KEYWORDS = ["reembolso", "cuenta", "prestamo", "horario", "tarjeta", "seguro"]


def scan_file(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()
    except Exception:
        text = ""
    return {
        "source": os.path.basename(path),
        "path": path,
        "ext": os.path.splitext(path)[1].lstrip(".").lower(),
        "size_bytes": os.path.getsize(path),
        "n_chars": len(text),
        **{f"has_{k}": (k in text) for k in KEYWORDS},
    }


def build():
    if pd is None:
        raise SystemExit("[!] Falta pandas/pyarrow: pip install pandas pyarrow")
    os.makedirs(config.CATALOG_DIR, exist_ok=True)

    files = [
        os.path.join(config.DOCS_DIR, f)
        for f in os.listdir(config.DOCS_DIR)
        if f.lower().endswith((".md", ".txt", ".pdf"))
    ]
    df = pd.DataFrame([scan_file(f) for f in files])
    out = os.path.join(config.CATALOG_DIR, "policyqa_docs.parquet")
    df.to_parquet(out, index=False)
    print(f"Catalogo: {len(df)} docs -> {out}")
    return out


if __name__ == "__main__":
    build()
