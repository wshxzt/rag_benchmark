"""
Query Vertex AI Vector Search 2.0 using SemanticSearch.

The server auto-embeds query text using the model configured in the collection,
so no manual embedding generation is needed.
"""
import concurrent.futures
import time

from google.cloud import vectorsearch_v1beta as vs
from tqdm import tqdm

import config


def _search_client():
    return vs.DataObjectSearchServiceClient()


def _collection_path():
    return (
        f"projects/{config.PROJECT_ID}/locations/{config.VS2_LOCATION}"
        f"/collections/{config.VS2_COLLECTION_ID}"
    )


def run_queries(queries: dict, top_k: int = 10, max_workers: int = 32) -> tuple:
    """
    Args:
        queries:     {query_id: query_text}
        top_k:       number of results to retrieve
        max_workers: thread pool size for concurrent queries

    Returns:
        results:   {query_id: {beir_doc_id: score}}
        latencies: {query_id: float}  — seconds per query (individual API call time)
    """
    client = _search_client()
    collection_path = _collection_path()

    def _query_one(query_id, query_text):
        request = vs.SearchDataObjectsRequest(
            parent=collection_path,
            semantic_search=vs.SemanticSearch(
                search_text=query_text,
                search_field="embedding",
                task_type=vs.EmbeddingTaskType.RETRIEVAL_QUERY,
                top_k=top_k,
                output_fields=vs.OutputFields(data_fields=["*"]),
            ),
        )
        t0 = time.perf_counter()
        response = client.search_data_objects(request=request)
        latency = time.perf_counter() - t0

        ranked = {}
        for i, result in enumerate(response.results[:top_k]):
            doc_id = result.data_object.name.split("/")[-1]
            score = 1.0 - float(result.distance) if result.distance is not None else 1.0 / (i + 1)
            ranked[doc_id] = score
        return query_id, ranked, latency

    results = {}
    latencies = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_query_one, qid, qtxt): qid
            for qid, qtxt in queries.items()
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="VS2 queries",
        ):
            query_id, ranked, latency = future.result()
            results[query_id] = ranked
            latencies[query_id] = latency

    return results, latencies
