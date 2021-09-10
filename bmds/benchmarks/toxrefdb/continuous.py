import os
from concurrent.futures import ProcessPoolExecutor
from enum import Enum

import pandas as pd
from tqdm.auto import tqdm

from bmds.bmds2.models.continuous import (
    Exponential_M3_111,
    Exponential_M5_111,
    Hill_218,
    Polynomial_221,
    Power_219,
)
from bmds.bmds3.models.continuous import ExponentialM3, ExponentialM5, Hill, Polynomial, Power

from . import db, models, schemas
from .shared import _execute_bmds2_model, _execute_bmds3_model, session_scope


class ContinuousModel(Enum):
    Power = "Power"
    Hill = "Hill"
    Polynomial = "Polynomial"
    ExponentialM3 = "ExponentialM3"
    ExponentialM5 = "ExponentialM5"


model_dict = {
    "bmds2": [
        (Power_219, ContinuousModel.Power.value),
        (Hill_218, ContinuousModel.Hill.value),
        (Polynomial_221, ContinuousModel.Polynomial.value),
        (Exponential_M3_111, ContinuousModel.ExponentialM3.value),
        (Exponential_M5_111, ContinuousModel.ExponentialM5.value),
    ],
    "bmds3": [
        (Power, ContinuousModel.Power.value),
        (Hill, ContinuousModel.Hill.value),
        (Polynomial, ContinuousModel.Polynomial.value),
        (ExponentialM3, ContinuousModel.ExponentialM3.value),
        (ExponentialM5, ContinuousModel.ExponentialM5.value),
    ],
}
execute_dict = {"bmds2": _execute_bmds2_model, "bmds3": _execute_bmds3_model}


def getModels(version):
    return model_dict[version]


def _clean_dataset(ds):
    return schemas.ContinuousDatasetSchema(**ds).dict()


def bulk_save_datasets(datasets: "list[dict]"):
    cleaned_datasets = map(_clean_dataset, datasets)
    objects = map(lambda ds: models.ContinuousDataset(**ds), cleaned_datasets)
    with session_scope() as session:
        session.bulk_save_objects(objects)


def get_datasets():
    return list(map(lambda obj: obj.to_bmds(), models.ContinuousDataset.query.all()))


# _models, version, execute
def runContinuousModels(version):
    datasets = get_datasets()[:10]
    results = []
    nprocs = max(os.cpu_count() - 1, 1)
    for Model, model_name in tqdm(model_dict[version]):
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            with tqdm(datasets, leave=False) as progress:
                futures = []
                for ds in datasets:
                    m = Model(ds)
                    future = executor.submit(execute_dict[version], m, model_name, ds, version)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)
                for future in futures:
                    results.append(future.result())
    objects = map(lambda res: models.ContinuousResult(**res.dict()), results)
    with session_scope() as session:
        session.bulk_save_objects(objects)


def compare_versions(ver1, ver2, threshold):
    raw_sql = f"""
        select {ver1}.dataset_id, {ver1}.model,
        {ver1}.bmd as {ver1}_bmd, {ver1}.bmdl as {ver1}_bmdl, {ver1}.bmdu as {ver1}_bmdu, {ver1}.aic as {ver1}_aic,
        {ver2}.bmd as {ver2}_bmd, {ver2}.bmdl as {ver2}_bmdl, {ver2}.bmdu as {ver2}_bmdu, {ver2}.aic as {ver2}_aic
        from
        (select * from continuous_results where bmds_version = '{ver1}')
        as {ver1},
        (select * from continuous_results where bmds_version = '{ver2}')
        as {ver2}
        where {ver1}.model = {ver2}.model
        and {ver1}.dataset_id = {ver2}.dataset_id;
    """
    df = pd.read_sql(raw_sql, db.session.bind)
    df["bmd_diff"] = ((df[f"{ver1}_bmd"] - df[f"{ver2}_bmd"]) / df[f"{ver1}_bmd"]).abs()
    df["bmdl_diff"] = ((df[f"{ver1}_bmdl"] - df[f"{ver2}_bmdl"]) / df[f"{ver1}_bmdl"]).abs()
    df["bmdu_diff"] = ((df[f"{ver1}_bmdu"] - df[f"{ver2}_bmdu"]) / df[f"{ver1}_bmdu"]).abs()
    df["aic_diff"] = ((df[f"{ver1}_aic"] - df[f"{ver2}_aic"]) / df[f"{ver1}_aic"]).abs()
    return df[
        (df["bmd_diff"] > threshold)
        | (df["bmdl_diff"] > threshold)
        | (df["bmdu_diff"] > threshold)
        | (df["aic_diff"] > threshold)
    ]
