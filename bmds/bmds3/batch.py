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
