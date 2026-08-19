# glue/etl_job.py
"""AWS Glue ETL job (PySpark) - serverless para PolicyQA.

Lee metadatos/registros crudos de documentos desde S3, limpia, genera senales
(keyword flags) y escribe un catalogo Parquet en S3 que luego se consulta con
Amazon Athena (via la tabla del Glue Data Catalog).

Como registrarlo (AWS CLI):
  aws glue create-job \
    --name policyqa-etl \
    --role <GLUE_SERVICE_ROLE_ARN> \
    --command '{"Name":"pythonshell","PythonVersion":"3","ScriptLocation":"s3://<bucket>/glue/etl_job.py"}' \
    --default-arguments '{"--BUCKET":"<bucket>","--SRC_PREFIX":"policyqa/raw","--OUT_PREFIX":"policyqa-catalog"}'
  aws glue start-job-run --job-name policyqa-etl

Requiere bucket S3 con JSON/CSV crudo de docs (policyqa/raw/).
"""
import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

import pyspark.sql.functions as F

ARGS = ["JOB_NAME", "BUCKET", "SRC_PREFIX", "OUT_PREFIX"]
args = getResolvedOptions(sys.argv, ARGS)

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

SRC = f"s3://{args['BUCKET']}/{args['SRC_PREFIX']}/"
OUT = f"s3://{args['BUCKET']}/{args['OUT_PREFIX']}/"

KEYWORDS = ["reembolso", "cuenta", "prestamo", "horario", "tarjeta", "seguro"]


def add_flag(col):
    """Crea bandera booleana si el texto (lowercase) contiene el keyword."""
    return F.lower(F.col("doc_text")).contains(col)


def main():
    # Leer JSON crudo de documentos
    raw = spark.read.json(SRC)

    # Limpieza + senales
    df = (
        raw
        .filter(F.col("doc_text").isNotNull())
        .dropDuplicates(["doc_id"])
        .withColumn("n_chars", F.length("doc_text"))
        .withColumn("ext", F.lower(F.regexp_extract("source", r"\\.(\\w+)$", 1)))
        .withColumn("has_reembolso", F.lower(F.col("doc_text")).contains("reembolso"))
        .withColumn("has_cuenta", F.lower(F.col("doc_text")).contains("cuenta"))
        .withColumn("has_prestamo", F.lower(F.col("doc_text")).contains("prestamo"))
        .withColumn("has_horario", F.lower(F.col("doc_text")).contains("horario"))
        .withColumn("has_tarjeta", F.lower(F.col("doc_text")).contains("tarjeta"))
        .withColumn("has_seguro", F.lower(F.col("doc_text")).contains("seguro"))
        .drop("doc_text")  # texto no va al catalogo; se indexa aparte para RAG
    )

    # Escribir Parquet particionado y registrar en Data Catalog (tabla Athena)
    df.write.mode("overwrite").format("parquet").partitionBy("ext").save(OUT)

    # Registrar tabla en el Glue Data Catalog para consultas Athena
    df.createOrReplaceTempView("docs")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS policyqa")
    df.write.mode("overwrite").format("parquet").saveAsTable("policyqa.policyqa_docs")

    glue_context.commit()
    print(f"Glue ETL OK: {df.count()} docs -> {OUT}")
    job.commit()


if __name__ == "__main__":
    main()
