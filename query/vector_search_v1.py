"""
Query Vertex AI Vector Search 1.0 (Matching Engine).

Generates a query embedding for each BEIR query using text-embedding-004,
then calls find_neighbors() on the deployed IndexEndpoint.
"""
import time

import vertexai
from google.cloud import aiplatform
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

import config
from utils.batching import dynamic_batches

EMBED_MODEL = "text-embedding-004"


def _init():
    vertexai.init(project=config.PROJECT_ID, location=config.VS1_LOCATION)
    aiplatform.init(project=config.PROJECT_ID, location=config.VS1_LOCATION)


def _get_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    _init()
    existing = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{config.VS1_ENDPOINT_DISPLAY_NAME}"'
    )
    if not existing:
        raise RuntimeError(f"No endpoint found with display_name={config.VS1_ENDPOINT_DISPLAY_NAME}")
    return existing[0]


def _embed_queries(query_texts: list[str]) -> list[list[float]]:
    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_QUERY") for t in query_texts]
    return [e.values for e in model.get_embeddings(inputs)]


def run_queries(queries: dict, top_k: int = 10) -> tuple:
    """
    Args:
        queries: {query_id: query_text}
        top_k:   number of results to retrieve

    Returns:
        results:   {query_id: {beir_doc_id: score}}
        latencies: {query_id: float}  — seconds
    """
    _init()
    endpoint = _get_endpoint()

    query_ids   = list(queries.keys())
    query_texts = list(queries.values())

    # Embed all queries using dynamic batching
    print("  Embedding queries ...")
    all_embeddings = [None] * len(query_texts)
    for idx_batch in tqdm(dynamic_batches(query_texts), desc="  Query embeddings"):
        batch_texts = [query_texts[i] for i in idx_batch]
        batch_embs  = _embed_queries(batch_texts)
        for i, emb in zip(idx_batch, batch_embs):
            all_embeddings[i] = emb

    # Query the index
    results   = {}
    latencies = {}

    print("  Querying index ...")
    for i, (query_id, emb) in enumerate(
        tqdm(zip(query_ids, all_embeddings), total=len(query_ids), desc="  VS1.0 queries")
    ):
        t0 = time.perf_counter()
        response = endpoint.find_neighbors(
            deployed_index_id=config.VS1_DEPLOYED_INDEX_ID,
            queries=[emb],
            num_neighbors=top_k,
        )
        latencies[query_id] = time.perf_counter() - t0

        ranked = {}
        for neighbor in response[0]:
            doc_id = neighbor.id
            # VS1.0 returns distance (cosine); convert to score
            score = 1.0 - float(neighbor.distance)
            ranked[doc_id] = score
        results[query_id] = ranked

    return results, latencies
