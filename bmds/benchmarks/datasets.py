from enum import Enum
from typing import NamedTuple

import pandas as pd

from .config import Config
from .db import Session, transaction
from .models import Dataset


class BenchmarkDatasetMetadata(NamedTuple):
    filename: str
    dataset_key: str


class BenchmarkDataset(Enum):
    """A benchmark dataset that can be used in to run an analysis"""

    TOXREFDB_V2 = (
        "TOXREFDB_V2",
        BenchmarkDatasetMetadata("toxrefdb_v2.csv.zip", "toxrefdb_v2"),
    )

    def __new__(cls, value: str, metadata: BenchmarkDatasetMetadata):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.metadata = metadata
        return obj

    def load_data(self):
        df = (
            pd.read_csv(Config.DATA / self.metadata.filename)
            .rename(columns={"id": "dataset_id", "meta": "dataset_metadata"})
            .fillna("")
        )

        datasets = []
        key = self.metadata.dataset_key
        for record in df.to_dict(orient="records"):
            ds = Dataset(**record)
            ds.dataset = key
            datasets.append(ds)

        with transaction() as sess:
            sess.bulk_save_objects(datasets)

    def data_loaded(self) -> bool:
        key = self.metadata.dataset_key
        return Session().query(Dataset).filter(Dataset.dataset == key).count() > 0
