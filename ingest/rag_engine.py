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
    """Stage docs to GCS subdirectories (≤9000 each), then import_files per subdirectory.

    Constraints:
      - import_files directory import: ≤10,000 files per directory
      - import_files individual URIs: ≤25 URIs per call
    Solution: upload to batch_N/ subdirs, import each dir with one call.
    """
    import time
    from google.cloud import storage

    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(bucket_name)
    base_prefix = f"rag_benchmark/{config.BEIR_DATASET}"

    BATCH_SIZE = 9000
    docs_list = list(corpus.items())
    n_batches = (len(docs_list) - 1) // BATCH_SIZE + 1
    print(f"  Uploading {len(docs_list)} docs in {n_batches} subdirectory batch(es) of ≤{BATCH_SIZE} ...")

    id_map = {}
    for batch_idx in range(n_batches):
        batch_docs = docs_list[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch_prefix = f"{base_prefix}/batch_{batch_idx}"
        batch_gcs_uri = f"gs://{bucket_name}/{batch_prefix}"

        existing = {b.name for b in bucket.list_blobs(prefix=batch_prefix)}
        to_upload = {
            doc_id: doc for doc_id, doc in batch_docs
            if f"{batch_prefix}/{doc_id}.txt" not in existing
        }
        print(f"  Batch {batch_idx+1}/{n_batches}: {len(to_upload)} to upload, {len(existing)} already exist")

        def _upload_one(item, _prefix=batch_prefix):
            doc_id, doc = item
            content = f"{doc.get('title', '')}\n\n{doc['text']}"
            bucket.blob(f"{_prefix}/{doc_id}.txt").upload_from_string(
                content, content_type="text/plain"
            )

        if to_upload:
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(_upload_one, item) for item in to_upload.items()]
                for _ in tqdm(concurrent.futures.as_completed(futures),
                              total=len(futures), desc=f"  Batch {batch_idx+1} upload"):
                    pass

        # Import this subdirectory (single directory URI, under the 10k-file limit)
        print(f"  Importing batch {batch_idx+1}/{n_batches} from {batch_gcs_uri} ...")
        for attempt in range(3):
            try:
                response = rag.import_files(
                    corpus_name=corpus_name,
                    paths=[batch_gcs_uri],
                    chunk_size=512,
                    chunk_overlap=50,
                    timeout=600,
                )
                print(f"  Batch {batch_idx+1} imported: {response}")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Batch {batch_idx+1} attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(10)
                else:
                    raise

        for doc_id, _ in batch_docs:
            id_map[f"{batch_gcs_uri}/{doc_id}.txt"] = doc_id

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
