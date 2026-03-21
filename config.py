import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID   = os.environ.get("GCP_PROJECT_ID",    "zhiting-personal")
LOCATION        = os.environ.get("GCP_LOCATION",        "us-central1")
RAG_LOCATION    = os.environ.get("GCP_RAG_LOCATION",    "us-west1")     # RAG Engine region (us-central1 restricted for new projects)
SEARCH_LOCATION = os.environ.get("GCP_SEARCH_LOCATION", "global")       # Vertex AI Search uses global location
VS1_LOCATION              = os.environ.get("GCP_VS1_LOCATION", "us-west1")  # Vector Search 1.0 region (must match GCS bucket region)
VS1_INDEX_DISPLAY_NAME    = "beir-scifact-vs1"
VS1_ENDPOINT_DISPLAY_NAME = "beir-scifact-vs1-endpoint"
VS1_DEPLOYED_INDEX_ID     = "beir_scifact_vs1"
VS1_GCS_PREFIX            = "vs1_embeddings/scifact"

# Vector Search 1.0 with Gemini-style chunking (text-embedding-005, 512-char chunks)
VS1_GC_INDEX_DISPLAY_NAME    = "beir-scifact-vs1-gc"
VS1_GC_ENDPOINT_DISPLAY_NAME = "beir-scifact-vs1-gc-endpoint"
VS1_GC_DEPLOYED_INDEX_ID     = "beir_scifact_vs1_gc"
VS1_GC_GCS_PREFIX            = "vs1_gc_embeddings/scifact"

VS2_LOCATION      = os.environ.get("GCP_VS2_LOCATION", "us-central1")  # Vector Search 2.0 region
VS2_COLLECTION_ID = "beir-scifact-vs2"

# Vertex AI RAG Engine
RAG_CORPUS_DISPLAY_NAME = "beir-scifact-corpus"

# Vertex AI Search
SEARCH_DATA_STORE_ID = "beir-scifact-store-v2"
SEARCH_ENGINE_ID     = "beir-scifact-engine-v2"

# GCS staging bucket for RAG Engine ingest (optional but recommended for speed)
# If set, docs are staged to GCS and imported in one LRO instead of one API call per doc.
GCS_STAGING_BUCKET = os.environ.get("GCS_STAGING_BUCKET", "")

# BEIR
BEIR_DATASET  = "scifact"
BEIR_DATA_DIR = "data/datasets"
BEIR_SPLIT    = "test"

# Evaluation
K_VALUES = [10]

# Results
RESULTS_DIR = "results"
