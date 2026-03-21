"""
Query Vertex AI RAG Engine for each BEIR test query.

Uses id_map {rag_file_resource_name_or_gcs_uri: beir_doc_id} saved during ingest
to map retrieved contexts back to BEIR doc IDs for evaluation.
"""
import concurrent.futures
import time

import vertexai
import vertexai.preview.rag as rag
from tqdm import tqdm

import config


def _query_one(query_id, query_text, corpus_name, top_k, id_map):
    t0 = time.perf_counter()
    response = rag.retrieval_query(
        text=query_text,
        rag_corpora=[corpus_name],
        similarity_top_k=top_k,
    )
    latency = time.perf_counter() - t0

    ranked = {}
    for i, ctx in enumerate(response.contexts.contexts):
        source = getattr(ctx, "source_uri", None) or getattr(ctx, "rag_file_id", None)
        beir_id = id_map.get(source)
        if beir_id is None:
            if source:
                stem = source.rstrip("/").split("/")[-1].replace(".txt", "")
                beir_id = stem if stem in id_map.values() else f"unknown_{i}"
            else:
                beir_id = f"unknown_{i}"
        score = float(ctx.score) if getattr(ctx, "score", None) else 1.0 / (i + 1)
        ranked[beir_id] = score

    return query_id, ranked, latency


def run_queries(
    queries: dict,
    corpus_name: str,
    id_map: dict,
    top_k: int = 10,
    max_workers: int = 32,
) -> tuple:
    """
    Args:
        queries:     {query_id: query_text}
        corpus_name: RAG corpus resource name
        id_map:      {rag_file_resource_name_or_gcs_uri: beir_doc_id}
        top_k:       number of results to retrieve
        max_workers: thread pool size for concurrent queries

    Returns:
        results:   {query_id: {beir_doc_id: score}}
        latencies: {query_id: float}  — seconds per query (individual API call time)
    """
    vertexai.init(project=config.PROJECT_ID, location=config.RAG_LOCATION)
    results = {}
    latencies = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_query_one, qid, qtxt, corpus_name, top_k, id_map): qid
            for qid, qtxt in queries.items()
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="RAG Engine queries",
        ):
            query_id, ranked, latency = future.result()
            results[query_id] = ranked
            latencies[query_id] = latency

    return results, latencies
