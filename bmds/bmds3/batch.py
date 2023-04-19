import json
import os
import zipfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import NamedTuple, Self

import pandas as pd
from tqdm import tqdm

from ..datasets import DatasetBase
from ..reporting.styling import Report
from .reporting import write_citation
from .sessions import BmdsSession


class ExecutionResponse(NamedTuple):
    success: bool
    content: dict | list[dict]


class BmdsSessionBatch:
    def __init__(self, sessions: list[BmdsSession] | None = None):
        if sessions is None:
            sessions = []
        self.sessions: list[BmdsSession] = sessions
        self.errors = []

    def df_summary(self) -> pd.DataFrame:
        dfs = [
            session.to_df(extras=dict(session_index=idx))
            for idx, session in enumerate(self.sessions)
        ]
        return pd.concat(dfs).dropna(axis=1, how="all").fillna("")

    def df_dataset(self) -> pd.DataFrame:
        data: list[dict] = []
        for idx, session in enumerate(self.sessions):
            data.extend(session.dataset.rows(extras=dict(session_index=idx)))
        return pd.DataFrame(data=data)

    def df_params(self) -> pd.DataFrame:
        data: list[dict] = []
        for idx, session in enumerate(self.sessions):
            for model_index, model in enumerate(session.models):
                if model.has_results:
                    data.extend(
                        model.results.parameters.rows(
                            extras=dict(
                                session_index=idx,
                                model_index=model_index,
                                model_name=model.name(),
                            )
                        )
                    )
        return pd.DataFrame(data=data)

    def to_excel(self, path: Path | None = None) -> Path | BytesIO:
        f: Path | BytesIO = path or BytesIO()
        with pd.ExcelWriter(f) as writer:
            data = {
                "summary": self.df_summary(),
                "datasets": self.df_dataset(),
                "parameters": self.df_params(),
            }
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name, index=False)
        return f

    def to_docx(
        self,
        report: Report | None = None,
        header_level: int = 1,
        citation: bool = True,
        dataset_format_long: bool = True,
        all_models: bool = False,
        bmd_cdf_table: bool = False,
        session_inputs_table: bool = False,
    ):
        """Append each session to a single document

        Args:
            report (Report, optional): A Report object, or None to use default.
            header_level (int, optional): Starting header level. Defaults to 1.
            citation (bool, default True): Include citation
            dataset_format_long (bool, default True): long or wide dataset table format
            all_models (bool, default False):  Show all models, not just selected
            bmd_cdf_table (bool, default False): Export BMD CDF table
            session_inputs_table (bool, default False): Write an inputs table for a session,
                assuming a single model's input settings are representative of all models in a
                session, which may not always be true

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
                session_inputs_table=session_inputs_table,
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
        cls, datasets: list[DatasetBase], runner: Callable, nprocs: int | None = None
    ) -> Self:
        """Execute sessions using multiple processors.

        Args:
            datasets (list[DatasetBase]): The datasets to execute
            runner (Callable[dataset] -> ExecutionResponse): The method which executes a session
            nprocs (Optional[int]): the number of processors to use; defaults to N-1. If 1 is
                specified; the batch session is called linearly without a process pool

        Returns:
            A BmdsSessionBatch with sessions executed.
        """
        if nprocs is None:
            nprocs = max(os.cpu_count() - 1, 1)

        results: list[ExecutionResponse] = []
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
    def deserialize(cls, data: str) -> Self:
        """Load serialized batch export into a batch session.

        Args:
            data (str): A JSON export generated from the `BmdsSessionBatch.serialize` method.
        """
        sessions_data = json.loads(data)
        sessions = [BmdsSession.from_serialized(session) for session in sessions_data]
        return BmdsSessionBatch(sessions=sessions)

    @classmethod
    def load(cls, archive: Path) -> Self:
        """Load a BmdsSession from a compressed zipfile

        Args:
            fn (Path): The zipfile path

        Returns:
            BmdsSessionBatch: An instance of this session
        """
        with zipfile.ZipFile(archive) as zf:
            with zf.open("data.json") as f:
                data = f.read()
        return BmdsSessionBatch.deserialize(data)

    def save(self, archive: Path):
        """Save BmdsSession to a compressed zipfile

        Args:
            fn (Path): The zipfile path
        """
        with zipfile.ZipFile(
            archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zf:
            zf.writestr("data.json", data=self.serialize())
