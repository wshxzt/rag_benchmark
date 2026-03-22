"""
Answer generation using Gemini 2.5 Flash.

For each query, builds a context string from the top retrieved documents,
renders an engine-specific prompt, and generates a grounded answer.
All queries are generated concurrently via a thread pool.
"""
import concurrent.futures
import time

import vertexai
from google.api_core.exceptions import ResourceExhausted
from tqdm import tqdm
from vertexai.generative_models import GenerationConfig, GenerativeModel

import config

GEMINI_MODEL = "gemini-2.5-flash"

# Default prompt — can be overridden per engine via the prompt_template argument
DEFAULT_PROMPT = """\
You are a scientific research assistant. Answer the following question \
based solely on the provided context. Be concise and factual.

Question: {query}

Context:
{context}

Answer:\
"""

GENERATION_CONFIG = GenerationConfig(temperature=0.0, max_output_tokens=512)


def _build_context(results: dict, corpus: dict, max_docs: int) -> str:
    """Return numbered context string from top-ranked doc IDs."""
    top = sorted(results.items(), key=lambda x: x[1], reverse=True)[:max_docs]
    parts = []
    for i, (doc_id, _) in enumerate(top, 1):
        doc   = corpus.get(doc_id, {})
        title = doc.get("title", "").strip()
        text  = doc.get("text", "").strip()
        parts.append(f"[{i}] {title}\n{text}" if title else f"[{i}] {text}")
    return "\n\n".join(parts)


def generate_answers(
    queries: dict,
    results: dict,
    corpus: dict,
    prompt_template: str = DEFAULT_PROMPT,
    max_docs: int = 5,
    max_workers: int = 16,
) -> dict:
    """
    Generate one answer per query using the retrieved context.

    Args:
        queries:         {query_id: query_text}
        results:         {query_id: {doc_id: score}}
        corpus:          {doc_id: {"title": ..., "text": ...}}
        prompt_template: f-string with {query} and {context} placeholders
        max_docs:        number of top retrieved docs to include in context
        max_workers:     thread pool size for concurrent Gemini calls

    Returns:
        {query_id: answer_text}
    """
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    model = GenerativeModel(GEMINI_MODEL)

    def _generate_one(query_id: str) -> tuple[str, str]:
        query_text = queries[query_id]
        context    = _build_context(results.get(query_id, {}), corpus, max_docs)
        prompt     = prompt_template.format(query=query_text, context=context)
        delay = 2.0
        for attempt in range(6):
            try:
                response = model.generate_content(prompt, generation_config=GENERATION_CONFIG)
                try:
                    return query_id, response.text.strip()
                except ValueError:
                    # finish_reason=MAX_TOKENS — extract partial text if available
                    parts = response.candidates[0].content.parts
                    text = parts[0].text.strip() if parts else ""
                    return query_id, text
            except ResourceExhausted:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay *= 2

    answers: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_generate_one, qid): qid for qid in queries}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="  Generating answers",
        ):
            query_id, answer = future.result()
            answers[query_id] = answer

    return answers
