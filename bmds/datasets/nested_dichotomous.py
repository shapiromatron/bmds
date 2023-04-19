import math
from typing import ClassVar

import numpy as np
from pydantic import root_validator

from .. import constants, plotting
from ..utils import str_list
from .base import DatasetBase, DatasetMetadata, DatasetSchemaBase


class NestedDichotomousDataset(DatasetBase):
    _BMDS_DATASET_TYPE = 1  # group data
    MINIMUM_DOSE_GROUPS = 3
    dtype = constants.Dtype.NESTED_DICHOTOMOUS

    DEFAULT_YLABEL = "Fraction affected"

    def __init__(
        self,
        doses: list[float],
        litter_ns: list[int],
        incidences: list[float],
        litter_covariates: list[int],
        **metadata,
    ):
        self.doses = doses
        self.litter_ns = litter_ns
        self.incidences = incidences
        self.litter_covariates = litter_covariates
        self.metadata = DatasetMetadata.parse_obj(metadata)
        self._sort_by_dose_group()
        self._validate()

    def _sort_by_dose_group(self):
        # use mergesort since it's a stable-sorting algorithm in numpy
        indexes = np.array(self.doses).argsort(kind="mergesort")
        for fld in ("doses", "litter_ns", "incidences", "litter_covariates"):
            arr = getattr(self, fld)
            setattr(self, fld, np.array(arr)[indexes].tolist())
        self._validate()

    def _validate(self):
        length = len(self.doses)
        if not all(
            len(lst) == length
            for lst in [self.doses, self.litter_ns, self.incidences, self.litter_covariates]
        ):
            raise ValueError("All input lists must be same length")

        if self.num_dose_groups < self.MINIMUM_DOSE_GROUPS:
            raise ValueError(
                f"Must have {self.MINIMUM_DOSE_GROUPS} or more dose groups after dropping doses"
            )

    def drop_dose(self):
        raise NotImplementedError("")

    @property
    def dataset_length(self):
        """
        Return the length of the vector of doses-used.
        """
        return self.num_dose_groups

    def serialize(self) -> "NestedDichotomousDatasetSchema":
        return NestedDichotomousDatasetSchema(
            dtype=self.dtype,
            doses=self.doses,
            litter_ns=self.litter_ns,
            incidences=self.incidences,
            litter_covariates=self.litter_covariates,
            metadata=self.metadata,
        )

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        super().update_record(d)
        d.update(
            dataset_doses=str_list(self.doses),
            dataset_litter_ns=str_list(self.litter_ns),
            dataset_incidences=str_list(self.incidences),
            dataset_litter_covariates=str_list(self.litter_covariates),
        )

    def as_dfile(self):
        raise ValueError("N/A; requires BMDS3+ which doesn't use dfiles")

    def plot(self):
        """
        Return a matplotlib figure of the dose-response dataset.

        Examples
        --------
        >>> fig = dataset.plot()
        >>> fig.show()
        >>> fig.clear()

        .. image:: ../tests/data/mpl/test_ddataset_plot.png
           :align: center
           :alt: Example generated BMD plot

        Returns
        -------
        out : matplotlib.figure.Figure
            A matplotlib figure representation of the dataset.
        """
        fig = plotting.create_empty_figure()
        ax = fig.gca()
        ax.set_xlabel(self.get_xlabel())
        ax.set_ylabel(self.get_ylabel())
        # TODO - replace in BMDS 3.4
        ax.margins(plotting.PLOT_MARGINS)
        ax.set_title(self._get_dataset_name())
        ax.legend(**plotting.LEGEND_OPTS)
        return fig

    def rows(self, extras: dict) -> list[dict]:
        """Return a list of rows; one for each item in a dataset"""
        raise NotImplementedError("TODO")


class NestedDichotomousDatasetSchema(DatasetSchemaBase):
    dtype: constants.Dtype = constants.Dtype.NESTED_DICHOTOMOUS
    metadata: DatasetMetadata
    doses: list[float]
    litter_ns: list[int]
    incidences: list[int]
    litter_covariates: list[float]

    MIN_N: ClassVar = 3
    MAX_N: ClassVar = math.inf

    def deserialize(self) -> NestedDichotomousDataset:
        ds = NestedDichotomousDataset(
            doses=self.doses,
            litter_ns=self.litter_ns,
            incidences=self.incidences,
            litter_covariates=self.litter_covariates,
            **self.metadata.dict(),
        )
        return ds

    @root_validator(skip_on_failure=True)
    def num_groups(cls, values):
        n_doses = len(values["doses"])
        n_litter_ns = len(values["litter_ns"])
        n_incidences = len(values["incidences"])
        n_litter_covariates = len(values["litter_covariates"])
        if len(set([n_doses, n_litter_ns, n_incidences, n_litter_covariates])) > 1:
            raise ValueError("Length of dose, litter, incidence, and covariate are not the same")
        if n_doses < cls.MIN_N:
            raise ValueError(f"At least {cls.MIN_N} groups are required")
        if n_doses > cls.MAX_N:
            raise ValueError(f"A maximum of {cls.MAX_N} groups are allowed")
        for incidence, n in zip(values["incidences"], values["litter_ns"], strict=True):
            if incidence > n:
                raise ValueError(f"Incidence cannot be greater than N ({incidence} > {n})")
        return values
