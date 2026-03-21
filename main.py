"""
RAG Benchmark: Vertex AI RAG Engine vs Vertex AI Search vs Vector Search 1.0 vs 2.0
Dataset: BEIR/scifact

Usage:
    python main.py
    python main.py --skip-ingest   # skip ingest if already done
"""
import argparse
import json
import os

import pandas as pd
from tabulate import tabulate

import config
from data.download import download_and_load
from evaluate.metrics import compute_avg_latency, compute_metrics
from ingest import rag_engine as rag_ingest
from ingest import vertex_search as vs_ingest
from ingest import vector_search_v1 as vs1_ingest
from ingest import vector_search_v2 as vs2_ingest
from query import rag_engine as rag_query
from query import vertex_search as vs_query
from query import vector_search_v1 as vs1_query
from query import vector_search_v2 as vs2_query

os.makedirs(config.RESULTS_DIR, exist_ok=True)
ID_MAP_PATH = os.path.join(config.RESULTS_DIR, "rag_engine_id_map.json")


def main(skip_ingest: bool = False):
    # ── 1. Download ───────────────────────────────────────────────────────────
    print("=== 1. Downloading BEIR/scifact ===")
    corpus, queries, qrels = download_and_load()
    print(f"  corpus: {len(corpus)} docs | queries: {len(queries)} | qrels: {len(qrels)}")

    # ── 2. Ingest ─────────────────────────────────────────────────────────────
    if not skip_ingest:
        print("\n=== 2a. Ingesting: Vertex AI RAG Engine ===")
        rag_corpus_name = rag_ingest.get_or_create_corpus()
        if not os.path.exists(ID_MAP_PATH):
            id_map = rag_ingest.ingest(rag_corpus_name, corpus)
            with open(ID_MAP_PATH, "w") as f:
                json.dump(id_map, f)
            print(f"  Saved id_map to {ID_MAP_PATH}")
        else:
            print(f"  id_map already exists at {ID_MAP_PATH}, skipping ingest.")

        print("\n=== 2b. Ingesting: Vertex AI Search ===")
        vs_ingest.get_or_create_data_store()
        vs_ingest.get_or_create_engine()
        vs_ingest.ingest(corpus)

        print("\n=== 2c. Ingesting: Vector Search 1.0 ===")
        vs1_ingest.ingest(corpus)

        print("\n=== 2d. Ingesting: Vector Search 2.0 ===")
        vs2_collection = vs2_ingest.get_or_create_collection()
        vs2_ingest.ingest(corpus, vs2_collection)
    else:
        print("\n=== Skipping ingest (--skip-ingest) ===")
        rag_corpus_name = rag_ingest.get_or_create_corpus()
        vs2_collection  = vs2_ingest.get_or_create_collection()

    # ── 3. Query ──────────────────────────────────────────────────────────────
    print("\n=== 3a. Querying: Vertex AI RAG Engine ===")
    id_map = json.load(open(ID_MAP_PATH))
    rag_results, rag_latencies = rag_query.run_queries(
        queries, rag_corpus_name, id_map, top_k=10
    )

    print("\n=== 3b. Querying: Vertex AI Search ===")
    vs_results, vs_latencies = vs_query.run_queries(queries, top_k=10)

    print("\n=== 3c. Querying: Vector Search 1.0 ===")
    vs1_results, vs1_latencies = vs1_query.run_queries(queries, top_k=10)

    print("\n=== 3d. Querying: Vector Search 2.0 ===")
    vs2_results, vs2_latencies = vs2_query.run_queries(queries, top_k=10)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n=== 4. Evaluating ===")
    rag_metrics  = compute_metrics(qrels, rag_results,  config.K_VALUES)
    vs_metrics   = compute_metrics(qrels, vs_results,   config.K_VALUES)
    vs1_metrics  = compute_metrics(qrels, vs1_results,  config.K_VALUES)
    vs2_metrics  = compute_metrics(qrels, vs2_results,  config.K_VALUES)

    rag_metrics["Avg Latency (ms)"]  = compute_avg_latency(rag_latencies)
    vs_metrics["Avg Latency (ms)"]   = compute_avg_latency(vs_latencies)
    vs1_metrics["Avg Latency (ms)"]  = compute_avg_latency(vs1_latencies)
    vs2_metrics["Avg Latency (ms)"]  = compute_avg_latency(vs2_latencies)

    # ── 5. Print & Save ───────────────────────────────────────────────────────
    rows = [
        {"System": "RAG Engine",        **rag_metrics},
        {"System": "Vertex Search",     **vs_metrics},
        {"System": "Vector Search 1.0", **vs1_metrics},
        {"System": "Vector Search 2.0", **vs2_metrics},
    ]
    df = pd.DataFrame(rows).set_index("System")

    print("\n" + tabulate(df.round(4), headers="keys", tablefmt="github"))

    csv_path = os.path.join(config.RESULTS_DIR, "comparison.csv")
    df.to_csv(csv_path)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingest steps (use if corpus/data store already populated)",
    )
    args = parser.parse_args()
    main(skip_ingest=args.skip_ingest)
