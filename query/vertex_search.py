"""
Query Vertex AI Search for each BEIR test query.

Document IDs returned by the API are the original BEIR doc IDs (set at ingest time),
so no id_map is needed.
"""
import time
from google.api_core.client_options import ClientOptions
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


def run_queries(queries: dict, top_k: int = 10) -> tuple:
    """
    Args:
        queries: {query_id: query_text}
        top_k:   number of results to retrieve

    Returns:
        results:   {query_id: {beir_doc_id: score}}
        latencies: {query_id: float}  — seconds
    """
    client = discoveryengine.SearchServiceClient(client_options=_client_options())
    serving_config = (
        f"projects/{config.PROJECT_ID}/locations/{config.SEARCH_LOCATION}"
        f"/collections/default_collection/engines/{config.SEARCH_ENGINE_ID}"
        f"/servingConfigs/default_config"
    )

    results = {}
    latencies = {}

    for query_id, query_text in tqdm(queries.items(), desc="Vertex Search queries"):
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query_text,
            page_size=top_k,
        )
        t0 = time.perf_counter()
        page_result = client.search(request)
        search_results = list(page_result)
        latencies[query_id] = time.perf_counter() - t0

        ranked = {}
        for i, result in enumerate(search_results[:top_k]):
            doc_id = result.document.id
            score = (
                float(result.relevance_score)
                if getattr(result, "relevance_score", None)
                else 1.0 / (i + 1)
            )
            ranked[doc_id] = score

        results[query_id] = ranked

    return results, latencies
