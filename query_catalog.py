"""Consulta SQL sobre el catalogo Parquet (equivalente local a Athena).

Usa DuckDB para correr SQL sobre policyqa_docs.parquet. En AWS, athena_query.py
corre exactamente la misma logica sobre la tabla del Glue Data Catalog.

Uso (SQL libre):
    python query_catalog.py "SELECT source, n_chars FROM docs WHERE has_reembolso"
    python query_catalog.py                       # corre los ejemplos

Nota SQL: el catalogo se expone como vista `docs`.
"""
import os
import sys

import config

PARQUET = os.path.join(config.CATALOG_DIR, "policyqa_docs.parquet").replace("\\", "/")

EXAMPLES = [
    ("Docs que mencionan 'reembolso'",
     "SELECT source, n_chars FROM docs WHERE has_reembolso = true"),
    ("Resumen del catalogo",
     "SELECT count(*) AS n_docs, round(avg(n_chars)) AS avg_chars, "
     "CAST(sum(CASE WHEN has_prestamo THEN 1 ELSE 0 END) AS INT) AS n_hablan_prestamos FROM docs"),
    ("Docs por extension",
     "SELECT ext, count(*) AS n FROM docs GROUP BY ext ORDER BY n DESC"),
]


def run(sql, parquet=PARQUET):
    import duckdb

    con = duckdb.connect()
    con.execute(f"CREATE VIEW docs AS SELECT * FROM read_parquet('{parquet}')")
    return con.execute(sql).fetchdf()


def main():
    if len(sys.argv) > 1:
        sql = sys.argv[1]
        print(run(sql).to_string(index=False))
        return
    for title, sql in EXAMPLES:
        print(f"\n=== {title} ===\n{sql}")
        print(run(sql).to_string(index=False))


if __name__ == "__main__":
    main()
