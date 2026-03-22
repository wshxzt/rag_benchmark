"""
Ingest BEIR corpus into Vertex AI RAG Engine.

Two strategies:
  A) GCS staging (fast): upload docs to GCS, then call import_files once as an LRO.
     Requires GCS_STAGING_BUCKET to be set in config.
  B) Direct upload (slow): upload_file() per document — no GCS bucket needed.
     ~5k docs will take a long time; use only for small subsets.

The returned id_map {rag_file_resource_name: beir_doc_id} must be saved to disk
and loaded by query/rag_engine.py to map retrieved contexts back to BEIR doc IDs.
"""
import concurrent.futures
import os
import tempfile
import vertexai
import vertexai.preview.rag as rag
from tqdm import tqdm
import config


def get_or_create_corpus() -> str:
    """Return the resource name of the RAG corpus, creating it if needed."""
    vertexai.init(project=config.PROJECT_ID, location=config.RAG_LOCATION)
    for corpus in rag.list_corpora():
        if corpus.display_name == config.RAG_CORPUS_DISPLAY_NAME:
            print(f"  Reusing existing corpus: {corpus.name}")
            return corpus.name
    corpus = rag.create_corpus(
        display_name=config.RAG_CORPUS_DISPLAY_NAME,
        description="BEIR scifact benchmark corpus",
    )
    print(f"  Created corpus: {corpus.name}")
    return corpus.name


def ingest(corpus_name: str, corpus: dict) -> dict:
    """
    Upload all docs to the RAG corpus.
    Returns id_map: {rag_file_resource_name: beir_doc_id}
    """
    vertexai.init(project=config.PROJECT_ID, location=config.RAG_LOCATION)
    if config.GCS_STAGING_BUCKET:
        return _ingest_via_gcs(corpus_name, corpus, config.GCS_STAGING_BUCKET)
    else:
        return _ingest_via_upload(corpus_name, corpus)


def _ingest_via_gcs(corpus_name: str, corpus: dict, bucket_name: str) -> dict:
    """Stage docs to GCS, then import_files in one LRO. Fast for large corpora."""
    from google.cloud import storage

    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(bucket_name)
    gcs_prefix = f"rag_benchmark/{config.BEIR_DATASET}"

    # Check how many files already exist in GCS to skip re-uploading
    existing = {b.name for b in bucket.list_blobs(prefix=gcs_prefix)}
    to_upload = {
        doc_id: doc for doc_id, doc in corpus.items()
        if f"{gcs_prefix}/{doc_id}.txt" not in existing
    }
    print(f"  Staging {len(to_upload)}/{len(corpus)} docs to gs://{bucket_name}/{gcs_prefix}/ ({len(existing)} already exist)")

    def _upload_one(item):
        doc_id, doc = item
        content = f"{doc.get('title', '')}\n\n{doc['text']}"
        bucket.blob(f"{gcs_prefix}/{doc_id}.txt").upload_from_string(
            content, content_type="text/plain"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(_upload_one, item) for item in to_upload.items()]
        for _ in tqdm(concurrent.futures.as_completed(futures),
                      total=len(futures), desc="  GCS upload"):
            pass

    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}"
    print(f"  Calling import_files from {gcs_uri} ...")
    for attempt in range(3):
        try:
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=[gcs_uri],
                chunk_size=512,
                chunk_overlap=50,
                timeout=600,
            )
            print(f"  Import complete: {response}")
            break
        except Exception as e:
            if attempt < 2:
                print(f"  import_files attempt {attempt+1} failed: {e}. Retrying...")
                import time; time.sleep(10)
            else:
                raise

    # Build id_map from GCS URI -> doc_id (doc_id is the filename stem)
    id_map = {}
    for doc_id in corpus:
        uri = f"gs://{bucket_name}/{gcs_prefix}/{doc_id}.txt"
        id_map[uri] = doc_id
    return id_map


def _ingest_via_upload(corpus_name: str, corpus: dict) -> dict:
    """Fallback: upload_file() per doc. Slow but requires no GCS bucket."""
    id_map = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for doc_id, doc in tqdm(corpus.items(), desc="  Uploading docs"):
            content = f"{doc.get('title', '')}\n\n{doc['text']}"
            fpath = os.path.join(tmpdir, f"{doc_id}.txt")
            with open(fpath, "w") as f:
                f.write(content)
            rag_file = rag.upload_file(
                corpus_name=corpus_name,
                path=fpath,
                display_name=doc_id,
                description=doc.get("title", ""),
            )
            # rag_file.name is the resource name; display_name is the BEIR doc_id
            id_map[rag_file.name] = doc_id
    return id_map
