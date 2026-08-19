# athena/table.ddl.sql
-- DDL para crear la tabla externa en Amazon Athena sobre el catalogo Parquet
-- que genera el job de Glue (policyqa-catalog/). Corre en Athena > Query editor,
-- o con: aws athena start-query-execution --query-string "$(cat athena/table.ddl.sql)"
--
-- Nota: sustituye <BUCKET> por tu bucket real.

CREATE EXTERNAL TABLE IF NOT EXISTS policyqa_docs (
  doc_id             string,
  source             string,
  ext                string,
  size_bytes         bigint,
  n_chars            bigint,
  has_reembolso      boolean,
  has_cuenta         boolean,
  has_prestamo       boolean,
  has_horario        boolean,
  has_tarjeta        boolean,
  has_seguro         boolean
)
PARTITIONED BY (ext string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
STORED AS INPUTFORMAT 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat'
       OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
LOCATION 's3://<BUCKET>/policyqa-catalog/'
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- Habilitar particiones:
MSCK REPAIR TABLE policyqa_docs;
