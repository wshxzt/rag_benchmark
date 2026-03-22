"""
Ingest BEIR corpus into Vector Search 1.0 with Gemini-style chunking.

Differences from plain VS1.0:
  - Documents are chunked with RecursiveCharacterTextSplitter (512 chars, 50 overlap)
  - Embeddings use text-embedding-005 instead of text-embedding-004
  - Chunk IDs use "{doc_id}__{chunk_idx}" so the query layer can recover the
    original document ID by splitting on "__"

Steps:
  1. Chunk all docs → generate embeddings with text-embedding-005
  2. Upload JSONL to GCS
  3. Create brute-force index
  4. Create public IndexEndpoint and deploy
"""
import json
import os
import time

import vertexai
from google.cloud import aiplatform, storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

import config
from utils.batching import dynamic_batches

EMBED_MODEL = "text-embedding-005"
DIMENSIONS  = 768

RESULTS_DIR        = config.RESULTS_DIR
INDEX_NAME_FILE    = os.path.join(RESULTS_DIR, "vs1_gc_index_name.txt")
ENDPOINT_NAME_FILE = os.path.join(RESULTS_DIR, "vs1_gc_endpoint_name.txt")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)


def _init():
    vertexai.init(project=config.PROJECT_ID, location=config.VS1_LOCATION)
    aiplatform.init(project=config.PROJECT_ID, location=config.VS1_LOCATION)


def _embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
    return [e.values for e in model.get_embeddings(inputs)]


def generate_and_upload_embeddings(corpus: dict) -> str:
    """
    Chunk all docs, embed with text-embedding-005, upload JSONL to GCS.
    Returns the GCS directory URI. Skips if file already exists.
    """
    _init()
    gcs_dir  = f"gs://{config.GCS_STAGING_BUCKET}/{config.VS1_GC_GCS_PREFIX}"
    gcs_file = f"{gcs_dir}/embeddings.json"

    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(config.GCS_STAGING_BUCKET)
    blob   = bucket.blob(f"{config.VS1_GC_GCS_PREFIX}/embeddings.json")
    if blob.exists():
        print(f"  Embeddings already in GCS at {gcs_file}, skipping.")
        return gcs_dir

    # Build (chunk_id, chunk_text) pairs
    all_chunks = []  # list of (chunk_id, chunk_text)
    for doc_id, doc in corpus.items():
        text   = f"{doc.get('title', '')}\n\n{doc['text']}"
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append((f"{doc_id}__{i}", chunk))

    print(f"  {len(corpus)} docs → {len(all_chunks)} chunks")

    texts     = [chunk_text for _, chunk_text in all_chunks]
    all_lines = []

    # Use tighter token budget: short chunks have higher token density
    # (scientific jargon, numbers, punctuation → ~3.5 chars/token vs estimate of 4)
    for idx_batch in tqdm(dynamic_batches(texts, max_tokens=15000), desc="  Embedding chunks"):
        batch_texts = [texts[i] for i in idx_batch]
        embeddings  = _embed_texts(batch_texts, task_type="RETRIEVAL_DOCUMENT")
        for i, emb in zip(idx_batch, embeddings):
            chunk_id = all_chunks[i][0]
            all_lines.append(json.dumps({"id": chunk_id, "embedding": emb}))

    blob.upload_from_string("\n".join(all_lines), content_type="application/json")
    print(f"  Uploaded {len(all_lines)} chunk embeddings to {gcs_file}")
    return gcs_dir


def get_or_create_index(gcs_dir: str) -> aiplatform.MatchingEngineIndex:
    """Create brute-force index from GCS embeddings. Reuses if exists."""
    _init()
    existing = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{config.VS1_GC_INDEX_DISPLAY_NAME}"'
    )
    if existing:
        print(f"  Reusing existing index: {existing[0].resource_name}")
        return existing[0]

    print(f"  Creating brute-force index from {gcs_dir} ...")
    index = aiplatform.MatchingEngineIndex.create_brute_force_index(
        display_name=config.VS1_GC_INDEX_DISPLAY_NAME,
        contents_delta_uri=gcs_dir,
        dimensions=DIMENSIONS,
        distance_measure_type="COSINE_DISTANCE",
        description="BEIR scifact VS1.0 gemini-chunking benchmark index",
    )
    print(f"  Created index: {index.resource_name}")
    with open(INDEX_NAME_FILE, "w") as f:
        f.write(index.resource_name)
    return index


def get_or_create_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    """Create a public IndexEndpoint. Reuses if exists."""
    _init()
    existing = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{config.VS1_GC_ENDPOINT_DISPLAY_NAME}"'
    )
    if existing:
        print(f"  Reusing existing endpoint: {existing[0].resource_name}")
        return existing[0]

    print("  Creating IndexEndpoint ...")
    try:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=config.VS1_GC_ENDPOINT_DISPLAY_NAME,
            public_endpoint_enabled=True,
        )
    except Exception as e:
        print(f"  Endpoint create raised ({e}), waiting for propagation ...")
        time.sleep(30)
        existing = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{config.VS1_GC_ENDPOINT_DISPLAY_NAME}"'
        )
        if not existing:
            raise
        endpoint = existing[0]
    print(f"  Created endpoint: {endpoint.resource_name}")
    with open(ENDPOINT_NAME_FILE, "w") as f:
        f.write(endpoint.resource_name)
    return endpoint


def deploy_index(
    index: aiplatform.MatchingEngineIndex,
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
):
    """Deploy index to endpoint. No-ops if already deployed."""
    _init()
    if any(d.id == config.VS1_GC_DEPLOYED_INDEX_ID for d in endpoint.deployed_indexes):
        print(f"  Index already deployed as '{config.VS1_GC_DEPLOYED_INDEX_ID}'.")
        return

    print("  Deploying index to endpoint (this takes ~20-30 min) ...")
    try:
        endpoint.deploy_index(
            index=index,
            deployed_index_id=config.VS1_GC_DEPLOYED_INDEX_ID,
            machine_type="n1-standard-16",
            min_replica_count=1,
            max_replica_count=1,
        )
        print("  Deployment complete.")
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            print(f"  Index already deployed as '{config.VS1_GC_DEPLOYED_INDEX_ID}' (from previous run), reusing.")
        else:
            raise


def ingest(corpus: dict):
    """Full ingest pipeline: chunk → embed → GCS → index → endpoint → deploy."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gcs_dir  = generate_and_upload_embeddings(corpus)
    index    = get_or_create_index(gcs_dir)
    endpoint = get_or_create_endpoint()
    deploy_index(index, endpoint)
    return endpoint
