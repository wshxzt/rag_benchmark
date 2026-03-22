"""
RAG Benchmark: Vertex AI RAG Engine vs Vertex AI Search vs Vector Search 1.0 vs 2.0
Dataset: BEIR/scifact

Usage:
    python main.py
    python main.py --skip-ingest          # skip ingest if already done
    python main.py --skip-ingest --skip-generate  # skip answer gen + autorater
"""
import argparse
import concurrent.futures
import json
import os

import pandas as pd
from tabulate import tabulate

import config
from data.download import download_and_load
from evaluate.autorater import avg_scores, rate_answers
from evaluate.metrics import compute_avg_latency, compute_metrics
from generate.answer import generate_answers
from ingest import rag_engine as rag_ingest
from ingest import vector_search_v1 as vs1_ingest
from ingest import vector_search_v1_gemini_chunking as vs1gc_ingest
from ingest import vector_search_v2 as vs2_ingest
from ingest import vertex_search as vs_ingest
from query import rag_engine as rag_query
from query import vector_search_v1 as vs1_query
from query import vector_search_v1_gemini_chunking as vs1gc_query
from query import vector_search_v2 as vs2_query
from query import vertex_search as vs_query

os.makedirs(config.RESULTS_DIR, exist_ok=True)
ID_MAP_PATH = os.path.join(config.RESULTS_DIR, "rag_engine_id_map.json")

# ── Per-engine answer generation prompts ──────────────────────────────────────
# Each prompt reflects how that engine retrieves and what kind of context it returns.

_RAG_PROMPT = """\
You are a scientific research assistant backed by a RAG (Retrieval-Augmented Generation) system.
The passages below were retrieved from a scientific corpus via semantic chunking.
Answer the question using only information present in these passages. Be concise.

Question: {query}

Retrieved passages:
{context}

Answer:\
"""

_VS_PROMPT = """\
You are a scientific expert assistant. The documents below were surfaced by a \
full-text search engine ranked by relevance.
Answer the question using only information present in these documents. Be concise.

Question: {query}

Search results:
{context}

Answer:\
"""

_VS1_PROMPT = """\
You are a scientific assistant. The documents below were retrieved by dense \
vector similarity search on full-document embeddings.
Answer the question using only information present in these documents. Be concise.

Question: {query}

Retrieved documents:
{context}

Answer:\
"""

_VS1GC_PROMPT = """\
You are a scientific assistant. The documents below were retrieved by dense vector \
search on 512-character text chunks; the best-matching chunk from each document \
was used to rank the documents.
Answer the question using only information present in these documents. Be concise.

Question: {query}

Retrieved documents (ranked by best chunk match):
{context}

Answer:\
"""

_VS2_PROMPT = """\
You are a scientific assistant. The documents below were retrieved by semantic \
vector search using an auto-embedding service.
Answer the question using only information present in these documents. Be concise.

Question: {query}

Semantically retrieved documents:
{context}

Answer:\
"""


def main(skip_ingest: bool = False, skip_generate: bool = False):
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

        print("\n=== 2d. Ingesting: Vector Search 1.0 + Gemini Chunking ===")
        vs1gc_ingest.ingest(corpus)

        print("\n=== 2e. Ingesting: Vector Search 2.0 ===")
        vs2_collection = vs2_ingest.get_or_create_collection()
        vs2_ingest.ingest(corpus, vs2_collection)
    else:
        print("\n=== Skipping ingest (--skip-ingest) ===")
        rag_corpus_name = rag_ingest.get_or_create_corpus()
        vs2_collection  = vs2_ingest.get_or_create_collection()

    # ── 3. Query (all 5 systems in parallel) ─────────────────────────────────
    print("\n=== 3. Querying all systems in parallel ===")
    id_map = json.load(open(ID_MAP_PATH))

    def _run_rag():
        return rag_query.run_queries(queries, rag_corpus_name, id_map, top_k=10)

    def _run_vs():
        return vs_query.run_queries(queries, top_k=10)

    def _run_vs1():
        return vs1_query.run_queries(queries, top_k=10)

    def _run_vs1gc():
        return vs1gc_query.run_queries(queries, top_k=10)

    def _run_vs2():
        return vs2_query.run_queries(queries, top_k=10)

    system_fns = {
        "rag":   _run_rag,
        "vs":    _run_vs,
        "vs1":   _run_vs1,
        "vs1gc": _run_vs1gc,
        "vs2":   _run_vs2,
    }

    system_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fn): name for name, fn in system_fns.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results_i, latencies_i = future.result()
                system_results[name] = (results_i, latencies_i)
                print(f"  [{name}] done")
            except Exception as e:
                print(f"  [{name}] FAILED: {e} — skipping")
                system_results[name] = ({}, {})

    rag_results,   rag_latencies   = system_results["rag"]
    vs_results,    vs_latencies    = system_results["vs"]
    vs1_results,   vs1_latencies   = system_results["vs1"]
    vs1gc_results, vs1gc_latencies = system_results["vs1gc"]
    vs2_results,   vs2_latencies   = system_results["vs2"]

    # ── 4. Retrieval evaluation ───────────────────────────────────────────────
    print("\n=== 4. Evaluating retrieval metrics ===")
    rag_metrics   = compute_metrics(qrels, rag_results,   config.K_VALUES)
    vs_metrics    = compute_metrics(qrels, vs_results,    config.K_VALUES)
    vs1_metrics   = compute_metrics(qrels, vs1_results,   config.K_VALUES)
    vs1gc_metrics = compute_metrics(qrels, vs1gc_results, config.K_VALUES)
    vs2_metrics   = compute_metrics(qrels, vs2_results,   config.K_VALUES)

    rag_metrics["Avg Latency (ms)"]   = compute_avg_latency(rag_latencies)
    vs_metrics["Avg Latency (ms)"]    = compute_avg_latency(vs_latencies)
    vs1_metrics["Avg Latency (ms)"]   = compute_avg_latency(vs1_latencies)
    vs1gc_metrics["Avg Latency (ms)"] = compute_avg_latency(vs1gc_latencies)
    vs2_metrics["Avg Latency (ms)"]   = compute_avg_latency(vs2_latencies)

    # ── 5. Answer generation + LLM autorater ─────────────────────────────────
    if not skip_generate:
        systems_for_gen = [
            ("rag",   rag_results,   _RAG_PROMPT,   rag_metrics),
            ("vs",    vs_results,    _VS_PROMPT,    vs_metrics),
            ("vs1",   vs1_results,   _VS1_PROMPT,   vs1_metrics),
            ("vs1gc", vs1gc_results, _VS1GC_PROMPT, vs1gc_metrics),
            ("vs2",   vs2_results,   _VS2_PROMPT,   vs2_metrics),
        ]

        for name, results, prompt_tmpl, metrics in systems_for_gen:
            if not results:
                print(f"\n=== 5. Skipping answer gen for {name} (no results) ===")
                continue
            print(f"\n=== 5. Answer generation + autorater: {name} ===")

            try:
                print("  Generating answers ...")
                answers = generate_answers(
                    queries=queries,
                    results=results,
                    corpus=corpus,
                    prompt_template=prompt_tmpl,
                )

                print("  Rating answers ...")
                ratings = rate_answers(
                    queries=queries,
                    answers=answers,
                    results=results,
                    corpus=corpus,
                )

                scores = avg_scores(ratings)
                metrics["Faithfulness (1-5)"] = scores["Faithfulness"]
                metrics["Relevance (1-5)"]    = scores["Relevance"]

                # Save answers for inspection
                answers_path = os.path.join(config.RESULTS_DIR, f"answers_{name}.json")
                with open(answers_path, "w") as f:
                    json.dump(answers, f, indent=2)
                print(f"  Answers saved to {answers_path}")
            except Exception as e:
                print(f"  [{name}] answer gen/rating FAILED: {e} — skipping")

    # ── 6. Print & Save ───────────────────────────────────────────────────────
    rows = [
        {"System": "RAG Engine",        **rag_metrics},
        {"System": "Vertex Search",     **vs_metrics},
        {"System": "Vector Search 1.0", **vs1_metrics},
        {"System": "Vector Search 1.0 GC", **vs1gc_metrics},
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
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip answer generation and LLM autorater",
    )
    args = parser.parse_args()
    main(skip_ingest=args.skip_ingest, skip_generate=args.skip_generate)
