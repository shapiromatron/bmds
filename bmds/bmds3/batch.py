import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, NamedTuple, Optional, Union

import pandas as pd
from tqdm import tqdm

from ..datasets import DatasetBase
from ..reporting.styling import Report
from .reporting import write_citation
from .sessions import BmdsSession


class ExecutionResponse(NamedTuple):
    success: bool
    content: Union[dict, list[dict]]


class BmdsSessionBatch:
    def __init__(self, sessions: Optional[List[BmdsSession]] = None):
        if sessions is None:
            sessions = []
        self.sessions: List[BmdsSession] = sessions
        self.errors = []

    def to_df(self) -> pd.DataFrame:
        dfs = [session.to_df() for session in self.sessions]
        return pd.concat(dfs).dropna(axis=1, how="all").fillna("")

    def to_docx(
        self,
        report: Optional[Report] = None,
        header_level: int = 1,
        citation: bool = True,
        dataset_format_long: bool = True,
        all_models: bool = False,
        bmd_cdf_table: bool = False,
    ):
        """Append each session to a single document

        Args:
            report (Report, optional): A Report object, or None to use default.
            header_level (int, optional): Starting header level. Defaults to 1.
            citation (bool, default True): Include citation
            dataset_format_long (bool, default True): long or wide dataset table format
            all_models (bool, default False):  Show all models, not just selected
            bmd_cdf_table (bool, default False): Export BMD CDF table

        Returns:
            A python docx.Document object with content added.
        """
        if report is None:
            report = Report.build_default()

        for session in self.sessions:
            session.to_docx(
                report,
                header_level=header_level,
                citation=False,
                dataset_format_long=dataset_format_long,
                all_models=all_models,
                bmd_cdf_table=bmd_cdf_table,
            )

        if citation and len(self.sessions) > 0:
            write_citation(report, self.sessions[0], header_level=header_level)

        return report.document

    def serialize(self) -> str:
        """Export BmdsSession into a json format which can be saved and loaded.

        Returns:
            str: A JSON string
        """
        return json.dumps([session.to_dict() for session in self.sessions])

    @classmethod
    def execute(
        cls, datasets: List[DatasetBase], runner: Callable, nprocs: Optional[int] = None
    ) -> "BmdsSessionBatch":
        """Execute sessions using multiple processors.

        Args:
            datasets (List[DatasetBase]): The datasets to execute
            runner (Callable[dataset] -> ExecutionResponse): The method which executes a session
            nprocs (Optional[int]): the number of processors to use; defaults to N-1. If 1 is
                specified; the batch session is called linearly without a process pool

        Returns:
            A BmdsSessionBatch with sessions executed.
        """
        if nprocs is None:
            nprocs = max(os.cpu_count() - 1, 1)

        results: List[ExecutionResponse] = []
        if nprocs == 1:
            # run without a ProcessPoolExecutor; useful for debugging
            for dataset in tqdm(datasets, desc="Executing..."):
                results.append(runner(dataset))
        else:
            # adapted from https://gist.github.com/alexeygrigorev/79c97c1e9dd854562df9bbeea76fc5de
            with ProcessPoolExecutor(max_workers=nprocs) as executor:
                with tqdm(total=len(datasets), desc="Executing...") as progress:

                    futures = []
                    for dataset in datasets:
                        future = executor.submit(runner, dataset)
                        future.add_done_callback(lambda p: progress.update())
                        futures.append(future)

                    for future in futures:
                        results.append(future.result())

        batch = cls()
        for result in tqdm(results, desc="Building batch..."):
            if result.success:
                if isinstance(result.content, list):
                    for item in result.content:
                        batch.sessions.append(BmdsSession.from_serialized(item))
                else:
                    batch.sessions.append(BmdsSession.from_serialized(result.content))
            else:
                batch.errors.append(result.content)

        return batch

    @classmethod
    def deserialize(cls, data: str) -> "BmdsSessionBatch":
        """Load serialized batch export into a batch session.

        Args:
            data (str): A JSON export generated from the `BmdsSessionBatch.serialize` method.
        """
        sessions_data = json.loads(data)
        sessions = [BmdsSession.from_serialized(session) for session in sessions_data]
        return BmdsSessionBatch(sessions=sessions)
