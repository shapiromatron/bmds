import json
import os

import pandas as pd

from . import exports
from .reporter import Reporter


class SessionBatch(list):
    """
    Export utilities for exporting a collection of multiple BMD sessions.

    Example
    -------
    >>> datasets = [
            bmds.ContinuousDataset(
                doses=[0, 10, 50, 150, 400],
                ns=[111, 142, 143, 93, 42],
                means=[2.112, 2.095, 1.956, 1.587, 1.254],
                stdevs=[0.235, 0.209, 0.231, 0.263, 0.159]),
            bmds.ContinuousDataset(
                doses=[0, 10, 50, 150, 400],
                ns=[111, 142, 143, 93, 42],
                means=[2.112, 2.095, 1.956, 1.587, 1.254],
                stdevs=[0.235, 0.209, 0.231, 0.263, 0.159])
        ]
    >>> batch = bmds.SessionBatch()
        for dataset in datasets:
            session = bmds.BMDS.latest_version(
                bmds.constants.CONTINUOUS,
                dataset=dataset)
            session.add_default_models()
            session.execute()
            session.recommend()
            batch.append(session)
    >>> df = batch.to_df()
    >>> batch.to_csv('~/Desktop/outputs.csv')
    >>> batch.save_plots('~/Desktop', recommended_only=True)
    """

    def to_dicts(self):
        """
        Return a list of dictionaries of all model inputs and outputs.

        Parameters
        ----------
        filename : str or file
            Either the file name (string) or an open file (file-like object)
            where the data will be saved.

        Returns
        -------
        out : list
            List of output dictionaries.
        """
        dicts = []
        for idx, session in enumerate(self):
            d = session.to_dict()
            d.update(dataset_index=idx)
            dicts.append(d)
        return dicts

    def to_json(self, filename, indent=2):
        """
        Return a JSON string of all model inputs and outputs.

        Parameters
        ----------
        filename : str or file
            Either the file name (string) or an open file (file-like object)
            where the data will be saved.
        indent : int, optional
            Indentation level for JSON output.

        Returns
        -------
        out : str
            JSON formatted output string.

        """
        d = self.to_dicts()
        if hasattr(filename, "write"):
            json.dump(d, filename, indent=indent)
        elif isinstance(filename, str):
            with open(os.path.expanduser(filename), "w") as f:
                json.dump(d, f, indent=indent)
        else:
            raise ValueError("Unknown filename or file-object")

    def to_df(self, recommended_only=False, include_io=True):
        """
        Return a pandas DataFrame for each model and dataset.

        Parameters
        ----------
        recommended_only : bool, optional
            If True, only recommended models for each session are included. If
            no model is recommended, then a row with it's ID will be included,
            but all fields will be null.
        include_io :  bool, optional
            If True, then the input/output files from BMDS will also be
            included, specifically the (d) input file and the out file.

        Returns
        -------
        out : pandas.DataFrame
            Data frame containing models and outputs

        """
        od = exports.df_ordered_dict(include_io)
        [session._add_to_to_ordered_dict(od, i, recommended_only) for i, session in enumerate(self)]
        return pd.DataFrame(od)

    def to_csv(self, filename, delimiter=",", recommended_only=False, include_io=True):
        """
        Return a CSV for each model and dataset.

        Parameters
        ----------
        filename : str or file
            Either the file name (string) or an open file (file-like object)
            where the data will be saved.
        delimiter : str, optional
            Delimiter used in CSV file between fields.
        recommended_only : bool, optional
            If True, only recommended models for each session are included. If
            no model is recommended, then a row with it's ID will be included,
            but all fields will be null.
        include_io :  bool, optional
            If True, then the input/output files from BMDS will also be
            included, specifically the (d) input file and the out file.

        Returns
        -------
        None

        """
        df = self.to_df(recommended_only, include_io)
        df.to_csv(filename, index=False, sep=delimiter)

    def to_excel(self, filename, recommended_only=False, include_io=True):
        """
        Return an Excel file for each model and dataset.

        Parameters
        ----------
        filename : str or ExcelWriter object
            Either the file name (string) or an ExcelWriter object.
        recommended_only : bool, optional
            If True, only recommended models for each session are included. If
            no model is recommended, then a row with it's ID will be included,
            but all fields will be null.
        include_io :  bool, optional
            If True, then the input/output files from BMDS will also be
            included, specifically the (d) input file and the out file.

        Returns
        -------
        None

        """
        df = self.to_df(recommended_only, include_io)
        if isinstance(filename, str):
            filename = os.path.expanduser(filename)
        df.to_excel(filename, index=False)

    def to_docx(
        self,
        filename=None,
        input_dataset=True,
        summary_table=True,
        recommendation_details=True,
        recommended_model=True,
        all_models=False,
    ):
        """
        Write batch sessions to a Word file.

        Parameters
        ----------
        filename : str or None
            If provided, the file is saved to this location, otherwise this
            method returns a docx.Document
        input_dataset : bool
            Include input dataset data table
        summary_table : bool
            Include model summary table
        recommendation_details : bool
            Include model recommendation details table
        recommended_model : bool
            Include the recommended model output and dose-response plot, if
            one exists
        all_models : bool
            Include all models output and dose-response plots

        Returns
        -------
        bmds.Reporter
            The bmds.Reporter object.

        """
        rep = Reporter()
        for model in self:
            rep.add_session(
                model,
                input_dataset,
                summary_table,
                recommendation_details,
                recommended_model,
                all_models,
            )

        if filename:
            rep.save(filename)

        return rep

    def save_plots(self, directory, format="png", recommended_only=False):
        """
        Save images of dose-response curve-fits for each model.

        Parameters
        ----------
        directory : str
            Directory where the PNG files will be saved.
        format : str, optional
            Image output format. Valid options include: png, pdf, svg, ps, eps
        recommended_only : bool, optional
            If True, only recommended models for each session are included. If
            no model is recommended, then a row with it's ID will be included,
            but all fields will be null.

        Returns
        -------
        None

        """
        for i, session in enumerate(self):
            session.save_plots(
                directory, prefix=str(i), format=format, recommended_only=recommended_only
            )
