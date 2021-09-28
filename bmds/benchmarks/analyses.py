import abc
from enum import Enum

from ..constants import Dtype, Version
from .datasets import BenchmarkDataset
from .db import Session, transaction
from .executors import model_fit
from .executors.shared import bulk_save_models, multiprocess
from .models import Dataset, ModelResult


class Analysis(abc.ABC):
    def __init__(self, key: int):
        self.key = key

    @abc.abstractmethod
    def clear_results(self, dataset: BenchmarkDataset, version: Version):
        """Delete results from database."""
        ...

    @abc.abstractmethod
    def execute(self, dataset: BenchmarkDataset, version: Version):
        """Execute analysis if possible with the specific version."""
        ...


class FitDichotomousAnalysis(Analysis):
    def datasets(self, dataset: BenchmarkDataset):
        qs = Session().query(Dataset).filter(Dataset.dtype == "D")
        return [ds.to_bmds() for ds in qs]

    def clear_results(self, dataset: BenchmarkDataset, version: Version):
        """
        Delete results from database.
        """
        with transaction() as sess:
            ids = (
                sess.query(ModelResult.id)
                .join("dataset")
                .filter(
                    ModelResult.analysis == self.key,
                    ModelResult.bmds_version == self.version,
                    Dataset.dtype == Dtype.DICHOTOMOUS,
                )
            )
            sess.query(ModelResult).filter(ModelResult.id.in_(ids)).delete(
                synchronize_session=False
            )

    def execute(self, dataset: BenchmarkDataset, version: Version):
        """Execute analysis if possible with the specific version."""
        datasets = self.datasets(dataset)
        runner = model_fit.executor[version]
        jobs = model_fit.build_jobs(datasets, version, Dtype.DICHOTOMOUS)
        results = multiprocess(jobs, runner)
        bulk_save_models(self.key, results)


class FitContinuousAnalysis(Analysis):
    def datasets(self, dataset: BenchmarkDataset):
        qs = Session().query(Dataset).filter(Dataset.dtype == "C")
        return [ds.to_bmds() for ds in qs]

    def clear_results(self, dataset: BenchmarkDataset, version: Version):
        """
        Delete results from database.
        """
        with transaction() as sess:
            ids = (
                sess.query(ModelResult.id)
                .join("dataset")
                .filter(
                    ModelResult.analysis == self.key,
                    ModelResult.bmds_version == self.version,
                    Dataset.dtype == Dtype.CONTINUOUS,
                )
            )
            sess.query(ModelResult).filter(ModelResult.id.in_(ids)).delete(
                synchronize_session=False
            )

    def execute(self, dataset: BenchmarkDataset, version: Version):
        """Execute analysis if possible with the specific version."""
        datasets = self.datasets(dataset)
        runner = model_fit.executor[version]
        jobs = model_fit.build_jobs(datasets, version, Dtype.CONTINUOUS)
        results = multiprocess(jobs, runner)
        bulk_save_models(self.key, results)


class BenchmarkAnalyses(Enum):
    """
    A specific type of BMDS analysis that can be conducted
    """

    FIT_DICHOTOMOUS = 0, FitDichotomousAnalysis
    FIT_CONTINUOUS = 1, FitContinuousAnalysis

    def __new__(cls, value: str, Executor: Analysis):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.executor = Executor(value)
        return obj
