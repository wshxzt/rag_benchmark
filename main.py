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
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd
from tabulate import tabulate

import config
from utils.checkpoint import Checkpoint
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
ID_MAP_PATH = os.path.join(config.RESULTS_DIR, f"rag_engine_id_map_{config.BEIR_DATASET}.json")

ENGINES = ["rag", "vs", "vs1", "vs1gc", "vs2"]


def _engine_dir(run_dir: str, engine: str) -> str:
    d = os.path.join(run_dir, engine)
    os.makedirs(d, exist_ok=True)
    return d


def _save_retrieval(run_dir: str, engine: str, results: dict, latencies: dict) -> None:
    d = _engine_dir(run_dir, engine)
    with open(os.path.join(d, "retrieval.json"), "w") as f:
        json.dump(results, f)
    with open(os.path.join(d, "latencies.json"), "w") as f:
        json.dump(latencies, f)


def _load_retrieval(run_dir: str, engine: str) -> Tuple[Optional[dict], Optional[dict]]:
    ret_path = os.path.join(run_dir, engine, "retrieval.json")
    lat_path = os.path.join(run_dir, engine, "latencies.json")
    if not os.path.exists(ret_path):
        return None, None
    with open(ret_path) as f:
        results = json.load(f)
    latencies = {}
    if os.path.exists(lat_path):
        with open(lat_path) as f:
            latencies = json.load(f)
    return results, latencies

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


def main(skip_ingest: bool = False, skip_generate: bool = False, retry_run: str = None, k: int = 10):
    # ── Run identity ──────────────────────────────────────────────────────────
    run_id  = retry_run if retry_run else str(uuid.uuid4())
    run_dir = os.path.join(config.RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    if retry_run:
        print(f"=== Retrying run {run_id} (failed queries only) ===")
    else:
        meta = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(), "k": k}
        with open(os.path.join(run_dir, "run_info.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"=== Run ID: {run_id} | K={k} ===")

    # ── 1. Download ───────────────────────────────────────────────────────────
    print("=== 1. Downloading BEIR/scifact ===")
    corpus, queries, qrels = download_and_load()
    print(f"  corpus: {len(corpus)} docs | queries: {len(queries)} | qrels: {len(qrels)}")

    # ── 2. Ingest ─────────────────────────────────────────────────────────────
    if not skip_ingest:
        print("\n=== 2. Ingesting all engines in parallel ===")

        def _ingest_rag():
            print("  [rag] starting ...")
            name = rag_ingest.get_or_create_corpus()
            if not os.path.exists(ID_MAP_PATH):
                id_map_data = rag_ingest.ingest(name, corpus)
                with open(ID_MAP_PATH, "w") as f:
                    json.dump(id_map_data, f)
                print(f"  [rag] id_map saved to {ID_MAP_PATH}")
            else:
                print(f"  [rag] id_map already exists, skipping ingest.")
            return name

        def _ingest_vs():
            print("  [vs] starting ...")
            vs_ingest.get_or_create_data_store()
            vs_ingest.get_or_create_engine()
            vs_ingest.ingest(corpus)
            print("  [vs] done")

        def _ingest_vs1():
            print("  [vs1] starting ...")
            vs1_ingest.ingest(corpus)
            print("  [vs1] done")

        def _ingest_vs1gc():
            print("  [vs1gc] starting ...")
            vs1gc_ingest.ingest(corpus)
            print("  [vs1gc] done")

        def _ingest_vs2():
            print("  [vs2] starting ...")
            col = vs2_ingest.get_or_create_collection()
            vs2_ingest.ingest(corpus, col)
            print("  [vs2] done")
            return col

        ingest_fns = {
            "rag": _ingest_rag, "vs": _ingest_vs, "vs1": _ingest_vs1,
            "vs1gc": _ingest_vs1gc, "vs2": _ingest_vs2,
        }
        ingest_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fn): name for name, fn in ingest_fns.items()}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    ingest_results[name] = future.result()
                    print(f"  [{name}] ingest complete")
                except Exception as e:
                    print(f"  [{name}] ingest FAILED: {e}")
                    ingest_results[name] = None

        rag_corpus_name = ingest_results["rag"]
        vs2_collection  = ingest_results["vs2"]
    else:
        print("\n=== Skipping ingest (--skip-ingest) ===")
        rag_corpus_name = rag_ingest.get_or_create_corpus()
        vs2_collection  = vs2_ingest.get_or_create_collection()

    # ── 3. Query (all 5 systems in parallel) ─────────────────────────────────
    print("\n=== 3. Querying all systems in parallel ===")
    id_map = json.load(open(ID_MAP_PATH))

    # Check which engines already have retrieval checkpoints
    pending_engines = {}
    system_results  = {}
    for name in ENGINES:
        cached_results, cached_latencies = _load_retrieval(run_dir, name)
        if cached_results is not None:
            print(f"  [{name}] retrieval loaded from checkpoint ({len(cached_results)} queries)")
            system_results[name] = (cached_results, cached_latencies)
        else:
            pending_engines[name] = None

    if pending_engines:
        def _run_rag():
            return rag_query.run_queries(queries, rag_corpus_name, id_map, top_k=k)

        def _run_vs():
            return vs_query.run_queries(queries, top_k=k)

        def _run_vs1():
            return vs1_query.run_queries(queries, top_k=k)

        def _run_vs1gc():
            return vs1gc_query.run_queries(queries, top_k=k)

        def _run_vs2():
            return vs2_query.run_queries(queries, top_k=k)

        all_fns = {
            "rag": _run_rag, "vs": _run_vs, "vs1": _run_vs1,
            "vs1gc": _run_vs1gc, "vs2": _run_vs2,
        }
        engine_fns = {name: fn for name, fn in all_fns.items() if name in pending_engines}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(engine_fns)) as executor:
            futures = {executor.submit(fn): name for name, fn in engine_fns.items()}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results_i, latencies_i = future.result()
                    system_results[name] = (results_i, latencies_i)
                    _save_retrieval(run_dir, name, results_i, latencies_i)
                    print(f"  [{name}] done — saved to {run_dir}/{name}/retrieval.json")
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
    rag_metrics   = compute_metrics(qrels, rag_results,   [k])
    vs_metrics    = compute_metrics(qrels, vs_results,    [k])
    vs1_metrics   = compute_metrics(qrels, vs1_results,   [k])
    vs1gc_metrics = compute_metrics(qrels, vs1gc_results, [k])
    vs2_metrics   = compute_metrics(qrels, vs2_results,   [k])

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

            eng_dir        = _engine_dir(run_dir, name)
            answers_ckpt   = Checkpoint(os.path.join(eng_dir, "answers.json"))
            ratings_ckpt   = Checkpoint(os.path.join(eng_dir, "ratings.json"))

            try:
                pending_ans = len(queries) - len(answers_ckpt)
                print(f"  Generating answers ... ({pending_ans} pending, {len(answers_ckpt)} cached)")
                answers = generate_answers(
                    queries=queries,
                    results=results,
                    corpus=corpus,
                    prompt_template=prompt_tmpl,
                    checkpoint=answers_ckpt,
                )

                pending_rat = len(queries) - len(ratings_ckpt)
                print(f"  Rating answers ... ({pending_rat} pending, {len(ratings_ckpt)} cached)")
                ratings = rate_answers(
                    queries=queries,
                    answers=answers,
                    results=results,
                    corpus=corpus,
                    checkpoint=ratings_ckpt,
                )

                scores = avg_scores(ratings)
                metrics["Faithfulness (1-5)"] = scores["Faithfulness"]
                metrics["Relevance (1-5)"]    = scores["Relevance"]

                print(f"  Answers/ratings saved to {eng_dir}/")
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

    csv_path = os.path.join(run_dir, "comparison.csv")
    df.to_csv(csv_path)
    print(f"\nRun ID: {run_id}")
    print(f"Results saved to {csv_path}")


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
    parser.add_argument(
        "--retry-run",
        metavar="UUID",
        default=None,
        help="Resume a previous run: load its retrieval checkpoints and retry only failed queries",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Retrieval cutoff: number of docs to retrieve and evaluate at (default: 10)",
    )
    args = parser.parse_args()
    main(skip_ingest=args.skip_ingest, skip_generate=args.skip_generate,
         retry_run=args.retry_run, k=args.k)
