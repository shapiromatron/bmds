from enum import Enum
from zipfile import ZipFile

import pandas as pd

from bmds.benchmarks.toxrefdb.continuous import runContinuousModels, save_continuous_datasets
from bmds.benchmarks.toxrefdb.dichotomous import runDichotomousModels, save_dichotomous_datasets
from bmds.benchmarks.toxrefdb.models import (
    ContinuousDataset,
    ContinuousResult,
    DichotomousDataset,
    DichotomousResult,
)

from .toxrefdb import db

filePath = "../data/toxrefdb/datasets.zip"


class Dataset(str, Enum):
    TOXREFDB_DICH = "TOXREFDB_DICH"
    TOXREFDB_CONT = "TOXREFDB_CONT"


class Versions(str, Enum):
    BMDS270 = "bmds270"
    BMDS330 = "bmds330"


class Benchmark:
    def run_analysis(dataset: Dataset, versions: list[Versions], clear_existing: bool):

        if dataset.value is Dataset.TOXREFDB_DICH.value:
            try:
                result = DichotomousDataset.query.all()
                if not result:
                    return "Dataset does not exist.Load dataset using Benchmark.load_dataset(Dataset.TOXREFDB_DICH, 'filename.csv') "
            except Exception:
                return "Table does not exist. load db tables using Benchmark.create_db()"
        if dataset.value is Dataset.TOXREFDB_CONT.value:
            try:
                result = ContinuousDataset.query.all()
                if not result:
                    return "Dataset does not exist.Load dataset using Benchmark.load_dataset(Dataset.TOXREFDB_DICH, 'filename.csv') "
            except Exception:
                return "Table does not exist. load db tables using Benchmark.create_db()"

        for version in versions:
            print(len(DichotomousResult.query.all()))
            if clear_existing:
                if dataset.value is Dataset.TOXREFDB_CONT.value:
                    ContinuousResult.query.filter(ContinuousResult.bmds_version == version).delete()
                if dataset.value is Dataset.TOXREFDB_DICH.value:
                    DichotomousResult.query.filter(
                        DichotomousResult.bmds_version == version
                    ).delete()
                    print(len(DichotomousResult.query.all()))
            if dataset.value is Dataset.TOXREFDB_CONT.value:
                runContinuousModels(version.value)
            if dataset.value is Dataset.TOXREFDB_DICH.value:
                runDichotomousModels(version.value)
                print(len(DichotomousResult.query.all()))
        return "Success"

    def load_dataset(dataset: Dataset, filename):
        zf = zf = ZipFile(filePath)
        df = pd.read_csv(zf.open(filename), index_col=0)
        if dataset.value is Dataset.TOXREFDB_DICH.value:
            save_dichotomous_datasets(df.to_dict("records"))
        if dataset.value is Dataset.TOXREFDB_CONT.value:
            save_continuous_datasets(df.to_dict("records"))
        return "Dataset Loaded"

    def create_db():
        db.create_all()
        return "Database created."
