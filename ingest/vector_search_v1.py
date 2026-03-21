"""
Ingest BEIR corpus into Vertex AI Vector Search 1.0 (Matching Engine).

Steps:
  1. Generate embeddings for all docs using text-embedding-004
  2. Upload embeddings JSONL to GCS
  3. Create a brute-force index (exact, no build time) pointing to GCS
  4. Create a public IndexEndpoint and deploy the index

The index and endpoint resource names are saved to results/ for use by
query/vector_search_v1.py and for cleanup after benchmarking.
"""
import json
import os

import vertexai
from google.cloud import aiplatform, storage
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

import config
from utils.batching import dynamic_batches

# Paths for persisting resource names across runs
RESULTS_DIR       = config.RESULTS_DIR
INDEX_NAME_FILE   = os.path.join(RESULTS_DIR, "vs1_index_name.txt")
ENDPOINT_NAME_FILE = os.path.join(RESULTS_DIR, "vs1_endpoint_name.txt")

EMBED_MODEL  = "text-embedding-004"
DIMENSIONS   = 768


def _init():
    vertexai.init(project=config.PROJECT_ID, location=config.VS1_LOCATION)
    aiplatform.init(project=config.PROJECT_ID, location=config.VS1_LOCATION)


def _embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
    return [e.values for e in model.get_embeddings(inputs)]


def generate_and_upload_embeddings(corpus: dict) -> str:
    """
    Generate doc embeddings and upload as JSONL to GCS.
    Returns the GCS directory URI.
    Skips upload if the file already exists.
    """
    _init()
    gcs_dir  = f"gs://{config.GCS_STAGING_BUCKET}/{config.VS1_GCS_PREFIX}"
    gcs_file = f"{gcs_dir}/embeddings.json"

    # Check if already uploaded
    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(config.GCS_STAGING_BUCKET)
    blob   = bucket.blob(f"{config.VS1_GCS_PREFIX}/embeddings.json")
    if blob.exists():
        print(f"  Embeddings already in GCS at {gcs_file}, skipping.")
        return gcs_dir

    print(f"  Generating embeddings for {len(corpus)} docs ...")
    docs      = list(corpus.items())
    texts     = [f"{d.get('title', '')}\n\n{d['text']}" for _, d in docs]
    all_lines = []

    batches = dynamic_batches(texts)
    for idx_batch in tqdm(batches, desc="  Embedding docs"):
        batch_texts = [texts[i] for i in idx_batch]
        embeddings  = _embed_texts(batch_texts, task_type="RETRIEVAL_DOCUMENT")
        for i, emb in zip(idx_batch, embeddings):
            doc_id = docs[i][0]
            all_lines.append(json.dumps({"id": str(doc_id), "embedding": emb}))

    jsonl_content = "\n".join(all_lines)
    blob.upload_from_string(jsonl_content, content_type="application/json")
    print(f"  Uploaded {len(all_lines)} embeddings to {gcs_file}")
    return gcs_dir


def get_or_create_index(gcs_dir: str) -> aiplatform.MatchingEngineIndex:
    """Create brute-force index pointing to GCS embeddings. Reuses if exists."""
    _init()
    existing = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{config.VS1_INDEX_DISPLAY_NAME}"'
    )
    if existing:
        print(f"  Reusing existing index: {existing[0].resource_name}")
        return existing[0]

    print(f"  Creating brute-force index from {gcs_dir} ...")
    index = aiplatform.MatchingEngineIndex.create_brute_force_index(
        display_name=config.VS1_INDEX_DISPLAY_NAME,
        contents_delta_uri=gcs_dir,
        dimensions=DIMENSIONS,
        distance_measure_type="COSINE_DISTANCE",
        description="BEIR scifact VS1.0 benchmark index",
    )
    print(f"  Created index: {index.resource_name}")
    with open(INDEX_NAME_FILE, "w") as f:
        f.write(index.resource_name)
    return index


def get_or_create_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    """Create a public IndexEndpoint. Reuses if exists."""
    _init()
    existing = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{config.VS1_ENDPOINT_DISPLAY_NAME}"'
    )
    if existing:
        print(f"  Reusing existing endpoint: {existing[0].resource_name}")
        return existing[0]

    import time
    print("  Creating IndexEndpoint ...")
    try:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=config.VS1_ENDPOINT_DISPLAY_NAME,
            public_endpoint_enabled=True,
        )
    except Exception as e:
        # SDK sometimes fails fetching the resource immediately after creation.
        # Wait for propagation and find it by display name.
        print(f"  Endpoint create raised ({e}), waiting for propagation ...")
        time.sleep(30)
        existing = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{config.VS1_ENDPOINT_DISPLAY_NAME}"'
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
    deployed = endpoint.deployed_indexes
    if any(d.id == config.VS1_DEPLOYED_INDEX_ID for d in deployed):
        print(f"  Index already deployed as '{config.VS1_DEPLOYED_INDEX_ID}'.")
        return

    print("  Deploying index to endpoint (this takes ~20-30 min) ...")
    endpoint.deploy_index(
        index=index,
        deployed_index_id=config.VS1_DEPLOYED_INDEX_ID,
        machine_type="n1-standard-16",  # required for SHARD_SIZE_MEDIUM (brute-force default)
        min_replica_count=1,
        max_replica_count=1,
    )
    print("  Deployment complete.")


def ingest(corpus: dict):
    """Full ingest pipeline: embed → GCS → index → endpoint → deploy."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gcs_dir  = generate_and_upload_embeddings(corpus)
    index    = get_or_create_index(gcs_dir)
    endpoint = get_or_create_endpoint()
    deploy_index(index, endpoint)
    return endpoint
