from beir.retrieval.evaluation import EvaluateRetrieval


def compute_metrics(qrels: dict, results: dict, k_values: list = [10]) -> dict:
    """
    Args:
        qrels:    {query_id: {doc_id: relevance_int}}
        results:  {query_id: {doc_id: score_float}}
        k_values: list of cutoffs, e.g. [10]

    Returns:
        Flat dict of metrics, e.g. {"NDCG@10": 0.52, "Recall@10": 0.61, ...}
        Returns zeros for all metrics if results is empty (system failed).
    """
    if not results:
        zeros = {}
        for k in k_values:
            zeros[f"NDCG@{k}"] = 0.0
            zeros[f"MAP@{k}"]  = 0.0
            zeros[f"Recall@{k}"] = 0.0
            zeros[f"P@{k}"]    = 0.0
        return zeros

    ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(
        qrels=qrels,
        results=results,
        k_values=k_values,
        ignore_identical_ids=True,
    )
    return {**ndcg, **map_, **recall, **precision}


def compute_avg_latency(latencies: dict) -> float:
    """Returns mean latency in milliseconds."""
    vals = list(latencies.values())
    return (sum(vals) / len(vals)) * 1000 if vals else 0.0
