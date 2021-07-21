import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, NamedTuple, Optional

import pandas as pd
from tqdm import tqdm

from ..datasets import DatasetBase
from ..reporting.styling import Report
from .sessions import BmdsSession


class ExecutionResponse(NamedTuple):
    success: bool
    content: Dict


class BmdsSessionBatch:
    def __init__(self, sessions: List[BmdsSession] = None):
        if sessions is None:
            sessions = []
        self.sessions: List[BmdsSession] = sessions
        self.errors = []

    def to_df(self) -> pd.DataFrame:
        dfs = [session.to_df(dropna=False) for session in self.sessions]
        return pd.concat(dfs).dropna(axis=1, how="all").fillna("")

    def to_docx(self, report: Report = None, header_level: int = 1):
        """Append each session to a single document

        Args:
            report (Report, optional): A Report object, or None to use default.
            header_level (int, optional): Starting header level. Defaults to 1.

        Returns:
            A python docx.Document object with content added.
        """
        if report is None:
            report = Report.build_default()

        for session in self.sessions:
            session.to_docx(report, header_level=header_level)

        return report.document

    def serialize(self) -> str:
        """Export BmdsSession into a json format which can be saved and loaded.

        Returns:
            str: A JSON string
        """
        return json.dumps([session.to_dict() for session in self.sessions])

    @classmethod
    def multiprocess_execute(
        cls, datasets: List[DatasetBase], runner: Callable, nprocs: Optional[int] = None
    ) -> "BmdsSessionBatch":
        """Execute sessions using multiple cores.

        Args:
            datasets (List[DatasetBase]): The datasets to execute
            runner (Callable[dataset] -> ExecutionResponse): The method which executes a session
            nprocs (Optional[int]): the number of processors to use; defaults to N-1

        Returns:
            A BmdsSessionBatch with sessions executed.
        """
        if nprocs is None:
            nprocs = max(os.cpu_count() - 1, 1)

        # adapted from https://gist.github.com/alexeygrigorev/79c97c1e9dd854562df9bbeea76fc5de
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            with tqdm(total=len(datasets), desc="Executing...") as progress:

                futures = []
                for dataset in datasets:
                    future = executor.submit(runner, dataset)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results: List[ExecutionResponse] = []
                for future in futures:
                    results.append(future.result())

        batch = cls()
        for result in tqdm(results, desc="Building batch..."):
            if result.success:
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
