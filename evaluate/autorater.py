"""
LLM autorater using Gemini 2.5 Flash.

Rates each generated answer on two dimensions:
  - faithfulness (1-5): every claim is supported by the retrieved context
  - relevance   (1-5): the answer directly addresses the question

Scores are parsed from structured JSON output and averaged across all queries
to produce per-engine scalars for the comparison table.
"""
import concurrent.futures
import json
import re

import vertexai
from tqdm import tqdm
from vertexai.generative_models import GenerationConfig, GenerativeModel

import config

GEMINI_MODEL = "gemini-2.5-flash"

RATING_PROMPT = """\
You are an expert evaluator for scientific question-answering systems.

Rate the answer below on two dimensions. Reply with ONLY a JSON object — \
no markdown, no extra text.

Question: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Scoring rubric (integer 1–5 each):
- faithfulness: Every factual claim in the answer is directly supported by \
the retrieved context. \
1 = the answer contradicts or ignores the context; \
5 = every claim is explicitly grounded in the context.
- relevance: The answer directly and completely addresses the question. \
1 = off-topic or empty; 5 = fully on-point and complete.

Respond with exactly this JSON (fill in the integers):
{{"faithfulness": <1-5>, "relevance": <1-5>}}\
"""

RATING_CONFIG = GenerationConfig(temperature=0.0, max_output_tokens=64)


def _build_context(results: dict, corpus: dict, max_docs: int) -> str:
    top   = sorted(results.items(), key=lambda x: x[1], reverse=True)[:max_docs]
    parts = []
    for i, (doc_id, _) in enumerate(top, 1):
        doc   = corpus.get(doc_id, {})
        title = doc.get("title", "").strip()
        text  = doc.get("text", "").strip()
        parts.append(f"[{i}] {title}\n{text}" if title else f"[{i}] {text}")
    return "\n\n".join(parts)


def _parse_scores(text: str) -> dict[str, float]:
    """Extract faithfulness + relevance from the model response."""
    # Strip markdown fences if the model adds them despite instructions
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
        return {
            "Faithfulness": max(1.0, min(5.0, float(data["faithfulness"]))),
            "Relevance":    max(1.0, min(5.0, float(data["relevance"])),   ),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"Faithfulness": 0.0, "Relevance": 0.0}


def rate_answers(
    queries: dict,
    answers: dict,
    results: dict,
    corpus: dict,
    max_docs: int = 5,
    max_workers: int = 16,
) -> dict:
    """
    Rate every answer on faithfulness and relevance.

    Args:
        queries:     {query_id: query_text}
        answers:     {query_id: answer_text}
        results:     {query_id: {doc_id: score}}
        corpus:      {doc_id: {"title": ..., "text": ...}}
        max_docs:    number of context docs to include in the rating prompt
        max_workers: thread pool size for concurrent Gemini calls

    Returns:
        {query_id: {"Faithfulness": float, "Relevance": float}}
    """
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    model = GenerativeModel(GEMINI_MODEL)

    def _rate_one(query_id: str) -> tuple[str, dict]:
        context  = _build_context(results.get(query_id, {}), corpus, max_docs)
        prompt   = RATING_PROMPT.format(
            query=queries[query_id],
            context=context,
            answer=answers.get(query_id, ""),
        )
        response = model.generate_content(prompt, generation_config=RATING_CONFIG)
        return query_id, _parse_scores(response.text)

    ratings: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_rate_one, qid): qid for qid in queries}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="  Rating answers",
        ):
            query_id, scores = future.result()
            ratings[query_id] = scores

    return ratings


def avg_scores(ratings: dict) -> dict[str, float]:
    """Average faithfulness and relevance across all queries."""
    if not ratings:
        return {"Faithfulness": 0.0, "Relevance": 0.0}
    keys = ["Faithfulness", "Relevance"]
    return {k: sum(r[k] for r in ratings.values()) / len(ratings) for k in keys}
