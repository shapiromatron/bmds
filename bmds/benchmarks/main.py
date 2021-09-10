from enum import Enum

from bmds.benchmarks.toxrefdb.continuous import runContinuousModels
from bmds.benchmarks.toxrefdb.dichotomous import runDichotomousModels
from bmds.benchmarks.toxrefdb.models import ContinuousResult, DichotomousResult


class Dataset(Enum):
    TOXREFDB_DICH = "TOXREFDB_DICH"
    TOXREFDB_CONT = "TOXREFDB_CONT"


class Versions(Enum):
    BMDS270 = "bmds2"
    BMDS330 = "bmds3"


def run_analysis(dataset: Dataset, versions: list[Versions], clear_existing: bool):
    for version in versions:
        if clear_existing:
            if dataset == Dataset.TOXREFDB_CONT.value:
                ContinuousResult.query.filter(ContinuousResult.bmds_version == version).delete()
            if dataset == Dataset.TOXREFDB_DICH.value:
                DichotomousResult.query.filter(DichotomousResult.bmds_version == version).delete()
        if dataset == Dataset.TOXREFDB_CONT.value:
            runContinuousModels(version)
        if dataset == Dataset.TOXREFDB_DICH.value:
            runDichotomousModels(version)
    return "Success"
