"""Corre consultas Amazon Athena sobre la tabla del Glue Data Catalog.

Equivalente en AWS de query_catalog.py (DuckDB local). Requiere credenciales AWS
y una tabla creada (ver athena/table.ddl.sql).

Uso:
    export AWS_REGION=us-east-1
    python athena_query.py "SELECT source, n_chars FROM policyqa_docs WHERE has_reembolso"
"""
import argparse
import sys
import time

import config


def run_athena(sql, database, output_bucket, region=config.BEDROCK_REGION):
    import boto3

    client = boto3.client("athena", region_name=region)
    start = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": f"s3://{output_bucket}/athena-results/"},
    )
    qid = start["QueryExecutionId"]
    while True:
        status = client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state = status["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        raise RuntimeError(f"Athena {state}: {status.get('StateChangeReason')}")

    result = client.get_query_results(QueryExecutionId=qid)
    # cabecera + filas
    rows = result.get("ResultSet", {}).get("Rows", [])
    parsed = [[c.get("VarCharValue", "") for c in r.get("Data", [])] for r in rows]
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sql", nargs="?", default="SELECT source, n_chars FROM policyqa_docs WHERE has_reembolso = true")
    parser.add_argument("--database", default="policyqa")
    parser.add_argument("--output-bucket", default=config.S3_BUCKET or "MY-ATHENA-OUTPUT-BUCKET")
    args = parser.parse_args()

    table = [r for r in run_athena(args.sql, args.database, args.output_bucket) if r]
    for row in table:
        print(" | ".join(row))


if __name__ == "__main__":
    main()
