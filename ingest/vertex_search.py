"""
Ingest BEIR corpus into Vertex AI Search.

Document IDs are set to the BEIR doc_id and are preserved verbatim,
so no id_map is needed at query time.
"""
import json
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from tqdm import tqdm
import config


def _client_options():
    endpoint = (
        "discoveryengine.googleapis.com"
        if config.SEARCH_LOCATION == "global"
        else f"{config.SEARCH_LOCATION}-discoveryengine.googleapis.com"
    )
    return ClientOptions(api_endpoint=endpoint)


def get_or_create_data_store() -> str:
    """Create data store and return its resource name. No-ops if already exists."""
    client = discoveryengine.DataStoreServiceClient(client_options=_client_options())
    parent = client.collection_path(
        project=config.PROJECT_ID,
        location=config.SEARCH_LOCATION,
        collection="default_collection",
    )
    data_store = discoveryengine.DataStore(
        display_name="BEIR SciFact Store",
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
        content_config=discoveryengine.DataStore.ContentConfig.NO_CONTENT,
    )
    try:
        operation = client.create_data_store(
            request=discoveryengine.CreateDataStoreRequest(
                parent=parent,
                data_store_id=config.SEARCH_DATA_STORE_ID,
                data_store=data_store,
            )
        )
        result = operation.result(timeout=300)
        print(f"  Created data store: {result.name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Data store already exists, reusing.")
        else:
            raise
    return f"{parent}/dataStores/{config.SEARCH_DATA_STORE_ID}"


def get_or_create_engine():
    """Create the search engine linked to the data store. No-ops if already exists."""
    client = discoveryengine.EngineServiceClient(client_options=_client_options())
    parent = client.collection_path(
        project=config.PROJECT_ID,
        location=config.SEARCH_LOCATION,
        collection="default_collection",
    )
    engine = discoveryengine.Engine(
        display_name="BEIR SciFact Engine",
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
        search_engine_config=discoveryengine.Engine.SearchEngineConfig(
            search_tier=discoveryengine.SearchTier.SEARCH_TIER_STANDARD,
        ),
        data_store_ids=[config.SEARCH_DATA_STORE_ID],
    )
    try:
        operation = client.create_engine(
            request=discoveryengine.CreateEngineRequest(
                parent=parent,
                engine=engine,
                engine_id=config.SEARCH_ENGINE_ID,
            )
        )
        result = operation.result(timeout=300)
        print(f"  Created engine: {result.name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  Engine already exists, reusing.")
        else:
            raise


def ingest(corpus: dict):
    """Import all docs using inline batches of 100."""
    client = discoveryengine.DocumentServiceClient(client_options=_client_options())
    parent = client.branch_path(
        project=config.PROJECT_ID,
        location=config.SEARCH_LOCATION,
        data_store=config.SEARCH_DATA_STORE_ID,
        branch="default_branch",
    )

    docs = list(corpus.items())
    batch_size = 100
    batches = range(0, len(docs), batch_size)

    for i in tqdm(batches, desc="  Importing batches"):
        batch = docs[i : i + batch_size]
        inline_documents = [
            discoveryengine.Document(
                id=doc_id,
                json_data=json.dumps(
                    {"title": doc.get("title", ""), "text": doc["text"]}
                ),
            )
            for doc_id, doc in batch
        ]
        operation = client.import_documents(
            request=discoveryengine.ImportDocumentsRequest(
                parent=parent,
                inline_source=discoveryengine.ImportDocumentsRequest.InlineSource(
                    documents=inline_documents,
                ),
                reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
            )
        )
        operation.result(timeout=120)
