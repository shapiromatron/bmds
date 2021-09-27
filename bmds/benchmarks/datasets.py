from enum import Enum
from typing import Any, NamedTuple

import pandas as pd

from .config import Config
from .db import Session
from .models import ContinuousDataset, DichotomousDataset


class BenchmarkDatasetMetadata(NamedTuple):
    filename: str
    db_model: Any
    dataset_key: str


class BenchmarkDataset(Enum):
    TOXREFDB_CONT = (
        "TOXREFDB_CONT",
        BenchmarkDatasetMetadata("toxrefdbv2_continuous.csv.zip", ContinuousDataset, "toxrefdbv2"),
    )
    TOXREFDB_DICH = (
        "TOXREFDB_DICH",
        BenchmarkDatasetMetadata(
            "toxrefdbv2_dichotomous.csv.zip", DichotomousDataset, "toxrefdbv2"
        ),
    )

    def __new__(cls, value: str, metadata: BenchmarkDatasetMetadata):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.metadata = metadata
        return obj

    def load_data(self):
        df = pd.read_csv(Config.DATA / self.metadata.filename).rename(
            columns={"id": "dataset_id", "meta": "dataset_metadata"}
        )
        datasets = [self.metadata.db_model(**record) for record in df.to_dict(orient="records")]
        with Session() as sess:
            sess.bulk_save_objects(datasets)

    def data_loaded(self) -> bool:
        Model = self.metadata.db_model
        return Session().query(Model).filter(Model.dataset == self.metadata.dataset_key).count() > 0
