"""Modulo de embeddings / LLM de PolicyQA.

Backend pluggable:
  - bedrock : Amazon Bedrock (Titan embeddings + Claude) -> produccion / CV.
  - local   : fastembed (ONNX, sin AWS) + respuesta extractiva -> dev/demo.
"""
import os

import config


def build_embeddings():
    """Devuelve el objeto embeddings segun EMBEDDINGS_BACKEND."""
    if config.EMBEDDINGS_BACKEND == "bedrock":
        import boto3
        from langchain_community.embeddings import BedrockEmbeddings

        client = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)
        return BedrockEmbeddings(model_id=config.TITAN_EMBED_MODEL, client=client)

    # local
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def has_bedrock():
    """True si hay credenciales AWS disponibles (para modo prod)."""
    if config.EMBEDDINGS_BACKEND != "bedrock":
        return False
    try:
        import boto3
        boto3.client("sts", region_name=config.BEDROCK_REGION).get_caller_identity()
        return True
    except Exception:
        return False


def build_llm():
    """LLM Bedrock (Nova via llamada directa, formato Messages). Accesible en esta cuenta."""
    import boto3

    client = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)
    return BedrockLLM(client=client)


class BedrockLLM:
    """Wrapper minimo: recibe un prompt y devuelve texto del LLM Bedrock (Nova)."""

    def __init__(self, client):
        self._client = client

    def invoke(self, prompt):
        return invoke_llm(prompt, client=self._client)


def invoke_llm(prompt, client=None):
    """Invoca el LLM de Bedrock (Nova Messages API) y devuelve texto."""
    import json

    if client is None:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)

    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {
            "max_new_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
        },
    }
    resp = client.invoke_model(
        modelId=config.LLM_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body).encode(),
    )
    out = json.loads(resp["body"].read())
    # Nova: output.message.content[].text
    try:
        return out["output"]["message"]["content"][0]["text"]
    except Exception:
        return str(out)


def build_prompt(system, context, question):
    """Arma el prompt final con sistema + contexto + pregunta."""
    return (
        f"{system}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}"
    )


def answer_from_sources(sources, question):
    """Modo LOCAL: respuesta extractiva honesta usando el top-k recuperado."""
    if not sources:
        return "No encuentro eso en los documentos cargados."
    parts = []
    for i, doc in enumerate(sources, 1):
        snippet = doc.page_content.strip().replace("\n", " ")
        parts.append(f"[{i}] {snippet[:400]}")
    return (
        "MODO LOCAL (sin Bedrock): no genero texto con LLM, recupero contexto.\n\n"
        + f"Pregunta: {question}\n\nContexto recuperado ({len(sources)}):\n"
        + "\n\n".join(parts)
    )
