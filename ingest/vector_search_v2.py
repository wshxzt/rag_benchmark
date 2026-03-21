"""
Ingest BEIR corpus into Vertex AI Vector Search 2.0.

Creates a Collection with auto-embedding via text-embedding-004, then batch-inserts
all documents. No manual embedding generation or endpoint deployment required.
"""
from tqdm import tqdm
from google.cloud import vectorsearch_v1beta as vs
import config
from utils.batching import dynamic_batches


def _vs_client():
    return vs.VectorSearchServiceClient()


def _do_client():
    return vs.DataObjectServiceClient()


def _parent():
    return f"projects/{config.PROJECT_ID}/locations/{config.VS2_LOCATION}"


def _collection_path():
    return f"{_parent()}/collections/{config.VS2_COLLECTION_ID}"


def get_or_create_collection() -> str:
    """Return collection path, creating it if it doesn't exist."""
    client = _vs_client()
    collection_path = _collection_path()

    try:
        collection = client.get_collection(
            request=vs.GetCollectionRequest(name=collection_path)
        )
        print(f"  Reusing existing collection: {collection.name}")
        return collection_path
    except Exception as e:
        if "not found" not in str(e).lower() and "404" not in str(e):
            raise

    print(f"  Creating collection {config.VS2_COLLECTION_ID} in {config.VS2_LOCATION} ...")
    collection = vs.Collection(
        display_name="BEIR SciFact VS2 Collection",
        description="BEIR scifact benchmark corpus for Vector Search 2.0",
        vector_schema={
            "embedding": vs.VectorField(
                dense_vector=vs.DenseVectorField(
                    dimensions=768,
                    vertex_embedding_config=vs.VertexEmbeddingConfig(
                        model_id="text-embedding-004",
                        text_template="{title}\n\n{text}",
                        task_type=vs.EmbeddingTaskType.RETRIEVAL_DOCUMENT,
                    ),
                )
            )
        },
    )
    operation = client.create_collection(
        request=vs.CreateCollectionRequest(
            parent=_parent(),
            collection_id=config.VS2_COLLECTION_ID,
            collection=collection,
        )
    )
    result = operation.result(timeout=120)
    print(f"  Created collection: {result.name}")
    return collection_path


def ingest(corpus: dict, collection_path: str):
    """Batch-insert all docs using dynamic token-aware batching.
    Server auto-generates embeddings via text-embedding-004."""
    client = _do_client()
    docs   = list(corpus.items())
    texts  = [f"{d.get('title', '')}\n\n{d['text']}" for _, d in docs]
    batches = dynamic_batches(texts)

    for idx_batch in tqdm(batches, desc="  Inserting data objects"):
        batch = [docs[i] for i in idx_batch]
        create_requests = [
            vs.CreateDataObjectRequest(
                parent=collection_path,
                data_object_id=str(doc_id),
                data_object=vs.DataObject(
                    data={"title": doc.get("title", ""), "text": doc["text"]}
                ),
            )
            for doc_id, doc in batch  # type: ignore[assignment]
        ]
        try:
            client.batch_create_data_objects(
                request=vs.BatchCreateDataObjectsRequest(
                    parent=collection_path,
                    requests=create_requests,
                )
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
            # Fall back to update for each doc individually
            for doc_id, doc in batch:
                data_object = vs.DataObject(
                    name=f"{collection_path}/dataObjects/{doc_id}",
                    data={"title": doc.get("title", ""), "text": doc["text"]},
                )
                client.update_data_object(
                    request=vs.UpdateDataObjectRequest(data_object=data_object)
                )
