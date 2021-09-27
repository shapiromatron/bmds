import os
from concurrent.futures import ProcessPoolExecutor

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

from ... import constants
from ..db import get_session, models, schemas
from .shared import _execute_bmds270_model, _execute_bmds330_model, session_scope

model_dict = {
    "bmds270": [
        (Power_219, constants.M_Power),
        (Hill_218, constants.M_Hill),
        (Polynomial_221, constants.M_Polynomial),
        (Exponential_M3_111, constants.M_ExponentialM3),
        (Exponential_M5_111, constants.M_ExponentialM5),
    ],
    "bmds330": [
        (Power, constants.M_Power),
        (Hill, constants.M_Hill),
        (Polynomial, constants.M_Polynomial),
        (ExponentialM3, constants.M_ExponentialM3),
        (ExponentialM5, constants.M_ExponentialM5),
    ],
}
execute_dict = {"bmds270": _execute_bmds270_model, "bmds330": _execute_bmds330_model}


def _clean_dataset(ds):
    return schemas.ContinuousDatasetSchema(**ds).dict()


def save_continuous_datasets(datasets: "list[dict]"):
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
