"""
Query Vertex AI Search for each BEIR test query.

Document IDs returned by the API are the original BEIR doc IDs (set at ingest time),
so no id_map is needed.
"""
import concurrent.futures
import time

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ResourceExhausted
from google.cloud import discoveryengine_v1 as discoveryengine
from tqdm import tqdm

import config


def _client_options():
    endpoint = (
        "discoveryengine.googleapis.com"
        if config.SEARCH_LOCATION == "global"
        else f"{config.SEARCH_LOCATION}-discoveryengine.googleapis.com"
    )
    return ClientOptions(api_endpoint=endpoint)


def run_queries(queries: dict, top_k: int = 10, max_workers: int = 8) -> tuple:
    """
    Args:
        queries:     {query_id: query_text}
        top_k:       number of results to retrieve
        max_workers: thread pool size for concurrent queries

    Returns:
        results:   {query_id: {beir_doc_id: score}}
        latencies: {query_id: float}  — seconds per query (individual API call time)
    """
    client = discoveryengine.SearchServiceClient(client_options=_client_options())
    serving_config = (
        f"projects/{config.PROJECT_ID}/locations/{config.SEARCH_LOCATION}"
        f"/collections/default_collection/engines/{config.SEARCH_ENGINE_ID}"
        f"/servingConfigs/default_config"
    )

    def _query_one(query_id, query_text):
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query_text,
            page_size=top_k,
        )
        # Retry on quota errors with exponential backoff
        delay = 2.0
        for attempt in range(6):
            try:
                t0 = time.perf_counter()
                search_results = list(client.search(request))
                latency = time.perf_counter() - t0
                break
            except ResourceExhausted:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay *= 2

        ranked = {}
        for i, result in enumerate(search_results[:top_k]):
            doc_id = result.document.id
            score = (
                float(result.relevance_score)
                if getattr(result, "relevance_score", None)
                else 1.0 / (i + 1)
            )
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
            desc="Vertex Search queries",
        ):
            query_id, ranked, latency = future.result()
            results[query_id] = ranked
            latencies[query_id] = latency

    return results, latencies
