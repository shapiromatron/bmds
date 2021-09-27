from enum import Enum

import pandas as pd

from .config import Config
from .db import Session
from .db.models import ContinuousDataset, DichotomousDataset


class Dataset(str, Enum):
    TOXREFDB_DICH = "TOXREFDB_DICH"
    TOXREFDB_CONT = "TOXREFDB_CONT"

    @classmethod
    def _fetch_data(cls, dataset):
        if dataset is cls.TOXREFDB_CONT:
            path = Config.DATA / "toxrefdbv2_continuous.csv.zip"
            Model = ContinuousDataset
        elif dataset is cls.TOXREFDB_DICH:
            path = Config.DATA / "toxrefdbv2_dichotomous.csv.zip"
            Model = DichotomousDataset
        else:
            raise ValueError("Unknown dataset")
        return path, Model


def load_dataset(dataset: Dataset):
    path, Model = Dataset._fetch_data(dataset)
    df = pd.read_csv(path).rename(columns={"id": "dataset_id", "meta": "dataset_metadata"})
    datasets = [Model(**record) for record in df.to_dict(orient="records")]
    with Session() as sess:
        sess.bulk_save_objects(datasets)
