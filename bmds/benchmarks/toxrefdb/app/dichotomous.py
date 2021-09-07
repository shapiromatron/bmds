from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from enum import Enum

from bmds.bmds2.models.dichotomous import DichotomousHill_13, Gamma_217, Logistic_215, LogLogistic_215, LogProbit_34, Multistage_34, Probit_34, QuantalLinear_34, Weibull_217


from bmds.bmds3.models.dichotomous import DichotomousHill, Gamma, Logistic, LogLogistic, LogProbit, Multistage, Probit, QuantalLinear, Weibull


from . import db
from . import schemas, models
from .shared import session_scope, _execute_bmds2_model, _execute_bmds3_model


class DichotomousModel(Enum):
    DichotomousHill = "DichotomousHill"
    Gamma = "Gamma"
    Logistic = "Logistic"
    LogLogistic = "LogLogistic"
    LogProbit = "LogProbit"
    Probit = "Probit"
    Weibull = "Weibull"


bmds2_models = [(DichotomousHill_13, DichotomousModel.DichotomousHill.value), (Gamma_217, DichotomousModel.Gamma.value), (Logistic_215, DichotomousModel.Logistic.value), (LogLogistic_215,
                                                                                                                                                                           DichotomousModel.LogLogistic.value), (LogProbit_34, DichotomousModel.LogProbit.value), (Probit_34, DichotomousModel.Probit.value), (Weibull_217, DichotomousModel.Weibull.value)]


bmds3_models = [(DichotomousHill, DichotomousModel.DichotomousHill.value), (Gamma, DichotomousModel.Gamma.value), (Logistic, DichotomousModel.Logistic.value), (LogLogistic,
                                                                                                                                                                DichotomousModel.LogLogistic.value), (LogProbit, DichotomousModel.LogProbit.value), (Probit, DichotomousModel.Probit.value), (Weibull, DichotomousModel.Weibull.value)]


def _clean_dataset(ds):
    return schemas.DichotomousDatasetSchema(**ds).dict()


def bulk_save_datasets(datasets: 'list[dict]'):
    cleaned_datasets = map(_clean_dataset, datasets)
    objects = map(lambda ds: models.DichotomousDataset(**ds), cleaned_datasets)
    with session_scope() as session:
        session.bulk_save_objects(objects)


def get_datasets():
    return list(map(lambda obj: obj.to_bmds(), models.DichotomousDataset.query.all()))


def _run_model(Model, model_name, datasets, version, execute):
    nprocs = max(os.cpu_count() - 1, 1)
    results = []
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        with tqdm(datasets, leave=False) as progress:
            futures = []
            for ds in datasets:
                m = Model(ds)
                future = executor.submit(execute, m, model_name, ds, version)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            for future in futures:
                results.append(future.result())
    return results


def _run_models(mods, version, execute):
    datasets = get_datasets()[:10]
    results = []
    for Model, model_name in tqdm(mods):
        results.extend(_run_model(Model, model_name,
                                  datasets, version, execute))
    objects = map(lambda res: models.DichotomousResult(**res.dict()), results)
    with session_scope() as session:
        session.bulk_save_objects(objects)


def run_bmds2_model(Model, model_name):
    datasets = get_datasets()[:10]
    results = _run_model(Model, model_name, datasets,
                         "bmds2", _execute_bmds2_model)
    objects = map(lambda res: models.DichotomousResult(**res.dict()), results)
    with session_scope() as session:
        session.bulk_save_objects(objects)


def run_bmds3_model(Model, model_name, version: str):
    datasets = get_datasets()[:10]
    results = _run_model(Model, model_name, datasets,
                         version, _execute_bmds3_model)
    objects = map(lambda res: models.DichotomousResult(**res.dict()), results)
    with session_scope() as session:
        session.bulk_save_objects(objects)


def run_bmds2_models():
    _run_models(bmds2_models, "bmds2", _execute_bmds2_model)


def run_bmds3_models(version: str):
    _run_models(bmds3_models, version, _execute_bmds3_model)


def compare_versions(ver1, ver2, threshold):
    raw_sql = f"""
        select {ver1}.dataset_id, {ver1}.model,
        {ver1}.bmd as {ver1}_bmd, {ver1}.bmdl as {ver1}_bmdl, {ver1}.bmdu as {ver1}_bmdu, {ver1}.aic as {ver1}_aic,
        {ver2}.bmd as {ver2}_bmd, {ver2}.bmdl as {ver2}_bmdl, {ver2}.bmdu as {ver2}_bmdu, {ver2}.aic as {ver2}_aic
        from
        (select * from dichotomous_results where bmds_version = '{ver1}')
        as {ver1},
        (select * from dichotomous_results where bmds_version = '{ver2}')
        as {ver2}
        where {ver1}.model = {ver2}.model
        and {ver1}.dataset_id = {ver2}.dataset_id;
    """
    df = pd.read_sql(raw_sql, db.session.bind)
    df["bmd_diff"] = (
        (df[f"{ver1}_bmd"]-df[f"{ver2}_bmd"])/df[f"{ver1}_bmd"]).abs()
    df["bmdl_diff"] = (
        (df[f"{ver1}_bmdl"]-df[f"{ver2}_bmdl"])/df[f"{ver1}_bmdl"]).abs()
    df["bmdu_diff"] = (
        (df[f"{ver1}_bmdu"]-df[f"{ver2}_bmdu"])/df[f"{ver1}_bmdu"]).abs()
    df["aic_diff"] = (
        (df[f"{ver1}_aic"]-df[f"{ver2}_aic"])/df[f"{ver1}_aic"]).abs()
    return df[(df["bmd_diff"] > threshold) | (df["bmdl_diff"] > threshold) | (df["bmdu_diff"] > threshold) | (df["aic_diff"] > threshold)]
