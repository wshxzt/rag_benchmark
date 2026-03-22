import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID      = os.environ.get("GCP_PROJECT_ID",      "zhiting-personal")
LOCATION        = os.environ.get("GCP_LOCATION",         "us-central1")
RAG_LOCATION    = os.environ.get("GCP_RAG_LOCATION",    "us-west1")      # RAG Engine region
SEARCH_LOCATION = os.environ.get("GCP_SEARCH_LOCATION", "global")         # Vertex AI Search uses global location
VS1_LOCATION    = os.environ.get("GCP_VS1_LOCATION",    "us-west1")       # Vector Search 1.0 region (must match GCS bucket region)
VS2_LOCATION    = os.environ.get("GCP_VS2_LOCATION",    "us-central1")    # Vector Search 2.0 region

# GCS staging bucket for RAG Engine ingest (optional but recommended for speed)
GCS_STAGING_BUCKET = os.environ.get("GCS_STAGING_BUCKET", "")

# BEIR — change BEIR_DATASET to switch datasets; all resource names derive from it
BEIR_DATASET  = "fiqa"
BEIR_DATA_DIR = "data/datasets"
BEIR_SPLIT    = "test"

# Resource names — all dataset-scoped so multiple datasets coexist without conflict
_D = BEIR_DATASET  # shorthand

RAG_CORPUS_DISPLAY_NAME = f"beir-{_D}-corpus"

SEARCH_DATA_STORE_ID = f"beir-{_D}-store"
SEARCH_ENGINE_ID     = f"beir-{_D}-engine"

VS1_INDEX_DISPLAY_NAME    = f"beir-{_D}-vs1"
VS1_ENDPOINT_DISPLAY_NAME = f"beir-{_D}-vs1-endpoint"
VS1_DEPLOYED_INDEX_ID     = f"beir_{_D}_vs1"
VS1_GCS_PREFIX            = f"vs1_embeddings/{_D}"

VS1_GC_INDEX_DISPLAY_NAME    = f"beir-{_D}-vs1-gc"
VS1_GC_ENDPOINT_DISPLAY_NAME = f"beir-{_D}-vs1-gc-endpoint"
VS1_GC_DEPLOYED_INDEX_ID     = f"beir_{_D}_vs1_gc"
VS1_GC_GCS_PREFIX            = f"vs1_gc_embeddings/{_D}"

VS2_COLLECTION_ID = f"beir-{_D}-vs2"

# Evaluation
K_VALUES = [10]

# Results
RESULTS_DIR = "results"
