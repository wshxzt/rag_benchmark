import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID   = os.environ.get("GCP_PROJECT_ID",    "zhiting-personal")
LOCATION        = os.environ.get("GCP_LOCATION",        "us-central1")
RAG_LOCATION    = os.environ.get("GCP_RAG_LOCATION",    "us-west1")     # RAG Engine region (us-central1 restricted for new projects)
SEARCH_LOCATION = os.environ.get("GCP_SEARCH_LOCATION", "global")       # Vertex AI Search uses global location

# Vertex AI RAG Engine
RAG_CORPUS_DISPLAY_NAME = "beir-scifact-corpus"

# Vertex AI Search
SEARCH_DATA_STORE_ID = "beir-scifact-store"
SEARCH_ENGINE_ID     = "beir-scifact-engine"

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
