"""
Query Vector Search 1.0 with Gemini-style chunking.

Embeds queries with text-embedding-005 (RETRIEVAL_QUERY), calls find_neighbors()
to retrieve top chunks, then aggregates chunk-level scores back to document level
using max pooling (highest-scoring chunk score = document score).

Chunk ID format from ingest: "{doc_id}__{chunk_idx}"
"""
import concurrent.futures
import time

import vertexai
from google.cloud import aiplatform
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

import config
from utils.batching import dynamic_batches

EMBED_MODEL = "text-embedding-005"


def _init():
    vertexai.init(project=config.PROJECT_ID, location=config.VS1_LOCATION)
    aiplatform.init(project=config.PROJECT_ID, location=config.VS1_LOCATION)


def _get_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    _init()
    existing = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{config.VS1_GC_ENDPOINT_DISPLAY_NAME}"'
    )
    if not existing:
        raise RuntimeError(
            f"No endpoint found with display_name={config.VS1_GC_ENDPOINT_DISPLAY_NAME}"
        )
    return aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=existing[0].resource_name
    )


def _embed_queries(query_texts: list[str]) -> list[list[float]]:
    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_QUERY") for t in query_texts]
    return [e.values for e in model.get_embeddings(inputs)]


def run_queries(queries: dict, top_k: int = 10, max_workers: int = 32) -> tuple:
    """
    Args:
        queries:     {query_id: query_text}
        top_k:       number of documents to return
        max_workers: thread pool size for concurrent find_neighbors calls

    Returns:
        results:   {query_id: {beir_doc_id: score}}
        latencies: {query_id: float}  — seconds per query (individual API call time)

    Retrieval strategy:
        Fetches top_k * 5 chunks per query to ensure at least top_k unique
        documents after deduplication, then aggregates via max pooling.
    """
    _init()
    endpoint = _get_endpoint()

    query_ids   = list(queries.keys())
    query_texts = list(queries.values())

    # Batch embed all queries (sequential — already efficient)
    print("  Embedding queries ...")
    all_embeddings = [None] * len(query_texts)
    for idx_batch in tqdm(dynamic_batches(query_texts, max_tokens=15000), desc="  Query embeddings"):
        batch_texts = [query_texts[i] for i in idx_batch]
        batch_embs  = _embed_queries(batch_texts)
        for i, emb in zip(idx_batch, batch_embs):
            all_embeddings[i] = emb

    # Retrieve more chunks than top_k so dedup still yields top_k unique docs
    num_chunks = top_k * 5

    def _query_one(query_id, emb):
        t0 = time.perf_counter()
        response = endpoint.find_neighbors(
            deployed_index_id=config.VS1_GC_DEPLOYED_INDEX_ID,
            queries=[emb],
            num_neighbors=num_chunks,
        )
        latency = time.perf_counter() - t0

        # Aggregate chunk scores → doc scores via max pooling
        # Chunk ID format: "{doc_id}__{chunk_idx}"
        doc_scores: dict[str, float] = {}
        for neighbor in response[0]:
            score    = 1.0 - float(neighbor.distance)
            doc_id   = neighbor.id.rsplit("__", 1)[0]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        # Return top_k docs ranked by score
        ranked = dict(
            sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        return query_id, ranked, latency

    results   = {}
    latencies = {}

    print("  Querying index ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_query_one, qid, emb): qid
            for qid, emb in zip(query_ids, all_embeddings)
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="  VS1-GC queries",
        ):
            query_id, ranked, latency = future.result()
            results[query_id]   = ranked
            latencies[query_id] = latency

    return results, latencies
