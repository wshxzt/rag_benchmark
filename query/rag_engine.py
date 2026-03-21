"""
Query Vertex AI RAG Engine for each BEIR test query.

Uses id_map {rag_file_resource_name_or_gcs_uri: beir_doc_id} saved during ingest
to map retrieved contexts back to BEIR doc IDs for evaluation.
"""
import time
import vertexai
import vertexai.preview.rag as rag
from tqdm import tqdm
import config


def run_queries(
    queries: dict,
    corpus_name: str,
    id_map: dict,
    top_k: int = 10,
) -> tuple:
    """
    Args:
        queries:     {query_id: query_text}
        corpus_name: RAG corpus resource name
        id_map:      {rag_file_resource_name_or_gcs_uri: beir_doc_id}
        top_k:       number of results to retrieve

    Returns:
        results:   {query_id: {beir_doc_id: score}}
        latencies: {query_id: float}  — seconds
    """
    vertexai.init(project=config.PROJECT_ID, location=config.RAG_LOCATION)
    results = {}
    latencies = {}

    for query_id, query_text in tqdm(queries.items(), desc="RAG Engine queries"):
        t0 = time.perf_counter()
        response = rag.retrieval_query(
            text=query_text,
            rag_corpora=[corpus_name],
            similarity_top_k=top_k,
        )
        latencies[query_id] = time.perf_counter() - t0

        ranked = {}
        for i, ctx in enumerate(response.contexts.contexts):
            # source_uri may be a GCS URI or a RAG file resource name
            source = getattr(ctx, "source_uri", None) or getattr(ctx, "rag_file_id", None)
            beir_id = id_map.get(source)
            if beir_id is None:
                # Try matching by filename stem in GCS URIs
                if source:
                    stem = source.rstrip("/").split("/")[-1].replace(".txt", "")
                    beir_id = stem if stem in id_map.values() else f"unknown_{i}"
                else:
                    beir_id = f"unknown_{i}"
            score = float(ctx.score) if getattr(ctx, "score", None) else 1.0 / (i + 1)
            ranked[beir_id] = score

        results[query_id] = ranked

    return results, latencies
