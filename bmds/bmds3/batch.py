import json
from typing import List

import pandas as pd

from ..reporting.styling import Report
from .sessions import BmdsSession


class BmdsSessionBatch:
    def __init__(self, sessions: List[BmdsSession] = None):
        if sessions is None:
            sessions = []
        self.sessions: List[BmdsSession] = sessions

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
        # TODO - update to use header styles
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
    def deserialize(cls, data: str) -> "BmdsSessionBatch":
        """Load serialized batch export into a batch session.

        Args:
            data (str): A JSON export generated from the `BmdsSessionBatch.serialize` method.
        """
        sessions_data = json.loads(data)
        sessions = [BmdsSession.from_serialized(session) for session in sessions_data]
        return BmdsSessionBatch(sessions=sessions)
