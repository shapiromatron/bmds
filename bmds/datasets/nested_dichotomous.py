from typing import List

import numpy as np

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
        doses: List[float],
        litter_ns: List[int],
        incidences: List[float],
        litter_covariates: List[int],
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


class NestedDichotomousDatasetSchema(DatasetSchemaBase):
    dtype: constants.Dtype
    metadata: DatasetMetadata
    doses: List[float]
    litter_ns: List[int]
    incidences: List[int]
    litter_covariates: List[float]

    def deserialize(self) -> NestedDichotomousDataset:
        ds = NestedDichotomousDataset(
            doses=self.doses,
            litter_ns=self.litter_ns,
            incidences=self.incidences,
            litter_covariates=self.litter_covariates,
            **self.metadata.dict(),
        )
        return ds
