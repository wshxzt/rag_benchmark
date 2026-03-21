import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import config


def download_and_load():
    """Download BEIR scifact dataset and return (corpus, queries, qrels)."""
    url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
        f"{config.BEIR_DATASET}.zip"
    )
    os.makedirs(config.BEIR_DATA_DIR, exist_ok=True)
    data_path = util.download_and_unzip(url, config.BEIR_DATA_DIR)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split=config.BEIR_SPLIT
    )
    return corpus, queries, qrels
