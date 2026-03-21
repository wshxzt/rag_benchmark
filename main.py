"""
RAG Benchmark: Vertex AI RAG Engine vs Vertex AI Search vs Vector Search 1.0 vs 2.0
Dataset: BEIR/scifact

Usage:
    python main.py
    python main.py --skip-ingest   # skip ingest if already done
"""
import argparse
import concurrent.futures
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

    # ── 3. Query (all 4 systems in parallel) ─────────────────────────────────
    print("\n=== 3. Querying all systems in parallel ===")
    id_map = json.load(open(ID_MAP_PATH))

    def _run_rag():
        return rag_query.run_queries(queries, rag_corpus_name, id_map, top_k=10)

    def _run_vs():
        return vs_query.run_queries(queries, top_k=10)

    def _run_vs1():
        return vs1_query.run_queries(queries, top_k=10)

    def _run_vs2():
        return vs2_query.run_queries(queries, top_k=10)

    system_fns = {
        "rag":  _run_rag,
        "vs":   _run_vs,
        "vs1":  _run_vs1,
        "vs2":  _run_vs2,
    }

    system_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn): name for name, fn in system_fns.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            results_i, latencies_i = future.result()
            system_results[name] = (results_i, latencies_i)
            print(f"  [{name}] done")

    rag_results,  rag_latencies  = system_results["rag"]
    vs_results,   vs_latencies   = system_results["vs"]
    vs1_results,  vs1_latencies  = system_results["vs1"]
    vs2_results,  vs2_latencies  = system_results["vs2"]

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
