import math
from typing import ClassVar

import numpy as np
from matplotlib.figure import Figure
from pydantic import confloat, root_validator
from scipy import stats

from .. import constants, plotting
from ..utils import str_list
from .base import DatasetBase, DatasetMetadata, DatasetPlottingSchema, DatasetSchemaBase


class DichotomousDataset(DatasetBase):
    """
    Dataset object for dichotomous datasets.

    A dichotomous dataset contains a list of 3 identically sized arrays of
    input values, for the dose, number of subjects, and incidences (subjects
    with a positive response).

    Example
    -------
    >>> dataset = bmds.DichotomousDataset(
            doses=[0, 1.96, 5.69, 29.75],
            ns=[75, 49, 50, 49],
            incidences=[5, 1, 3, 14]
        )
    """

    _BMDS_DATASET_TYPE = 1  # group data
    MINIMUM_DOSE_GROUPS = 3
    dtype = constants.Dtype.DICHOTOMOUS

    DEFAULT_YLABEL = "Fraction affected"

    def __init__(self, doses: list[float], ns: list[float], incidences: list[float], **metadata):
        self.doses = doses
        self.ns = ns
        self.incidences = incidences
        self.remainings = [n - p for n, p in zip(ns, incidences, strict=True)]
        self.metadata = DatasetMetadata.parse_obj(metadata)
        self._sort_by_dose_group()
        self._validate()

    def _sort_by_dose_group(self):
        # use mergesort since it's a stable-sorting algorithm in numpy
        indexes = np.array(self.doses).argsort(kind="mergesort")
        for fld in ("doses", "ns", "incidences", "remainings"):
            arr = getattr(self, fld)
            setattr(self, fld, np.array(arr)[indexes].tolist())
        self._validate()

    def _validate(self):
        length = len(self.doses)
        if not all(len(lst) == length for lst in [self.doses, self.ns, self.incidences]):
            raise ValueError("All input lists must be same length")

        if length != len(set(self.doses)):
            raise ValueError("Doses are not unique")

        if self.num_dose_groups < self.MINIMUM_DOSE_GROUPS:
            raise ValueError(
                f"Must have {self.MINIMUM_DOSE_GROUPS} or more dose groups after dropping doses"
            )

    def drop_dose(self):
        """
        Drop the maximum dose and related response values.
        """
        for fld in ("doses", "ns", "incidences", "remainings"):
            arr = getattr(self, fld)[:-1]
            setattr(self, fld, arr)
        self._validate()

    def as_dfile(self):
        """
        Return the dataset representation in BMDS .(d) file.

        Example
        -------
        >>> print(dataset.as_dfile())
        Dose Incidence NEGATIVE_RESPONSE
        0.000000 5 70
        1.960000 1 48
        5.690000 3 47
        29.750000 14 35
        """
        rows = ["Dose Incidence NEGATIVE_RESPONSE"]
        for i, v in enumerate(self.doses):
            if i >= self.num_dose_groups:
                continue
            rows.append("%f %d %d" % (self.doses[i], self.incidences[i], self.remainings[i]))
        return "\n".join(rows)

    @property
    def dataset_length(self):
        """
        Return the length of the vector of doses-used.
        """
        return self.num_dose_groups

    @staticmethod
    def _calculate_plotting(n, incidence):
        """
        Add confidence intervals to dichotomous datasets.
        https://www.epa.gov/sites/production/files/2020-09/documents/bmds_3.2_user_guide.pdf

        The error bars shown in BMDS plots use alpha = 0.05 and so
        represent the 95% confidence intervals on the observed
        proportions (independent of model).
        """
        p = incidence / float(n)
        z = stats.norm.ppf(1 - 0.05 / 2)
        z2 = z * z
        q = 1.0 - p
        tmp1 = 2 * n * p + z2
        ll = ((tmp1 - 1) - z * np.sqrt(z2 - (2 + 1 / n) + 4 * p * (n * q + 1))) / (2 * (n + z2))
        ul = ((tmp1 + 1) + z * np.sqrt(z2 + (2 + 1 / n) + 4 * p * (n * q - 1))) / (2 * (n + z2))
        return p, ll, ul

    def plot_data(self) -> DatasetPlottingSchema:
        if not getattr(self, "_plot_data", None):
            means, lls, uls = zip(
                *[
                    self._calculate_plotting(i, j)
                    for i, j in zip(self.ns, self.incidences, strict=True)
                ],
                strict=True,
            )
            self._plot_data = DatasetPlottingSchema(
                mean=means,
                ll=(np.array(means) - np.array(lls)).clip(0).tolist(),
                ul=(np.array(uls) - np.array(means)).clip(0).tolist(),
            )
        return self._plot_data

    def plot(self) -> Figure:
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
        ax = self.setup_plot()
        plot_data = self.plot_data()
        ax.errorbar(
            self.doses,
            plot_data.mean,
            yerr=[plot_data.ll, plot_data.ul],
            label="Fraction affected ± 95% CI",
            **plotting.DATASET_POINT_FORMAT,
        )
        ax.legend(**plotting.LEGEND_OPTS)
        return ax.get_figure()

    def serialize(self) -> "DichotomousDatasetSchema":
        return DichotomousDatasetSchema(
            dtype=self.dtype,
            doses=self.doses,
            ns=self.ns,
            incidences=self.incidences,
            plotting=self.plot_data(),
            metadata=self.metadata,
        )

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        super().update_record(d)
        d.update(
            dataset_doses=str_list(self.doses),
            dataset_ns=str_list(self.ns),
            dataset_incidences=str_list(self.incidences),
        )

    def rows(self, extras: dict) -> list[dict]:
        """Return a list of rows; one for each item in a dataset"""
        metadata = self.metadata.dict()
        rows = []
        for dose, n, incidence in zip(self.doses, self.ns, self.incidences, strict=True):
            rows.append({**extras, **metadata, **dict(dose=dose, n=n, incidence=incidence)})
        return rows


class DichotomousDatasetSchema(DatasetSchemaBase):
    dtype: constants.Dtype
    metadata: DatasetMetadata
    doses: list[confloat(ge=0)]
    ns: list[confloat(gt=0)]
    incidences: list[confloat(ge=0)]
    plotting: DatasetPlottingSchema | None

    MIN_N: ClassVar = 3
    MAX_N: ClassVar = math.inf

    @root_validator(skip_on_failure=True)
    def num_groups(cls, values):
        n_doses = len(values["doses"])
        n_ns = len(values["ns"])
        n_incidences = len(values["incidences"])
        if len(set([n_doses, n_ns, n_incidences])) > 1:
            raise ValueError("Length of doses, ns, and incidences are not the same")
        if n_doses < cls.MIN_N:
            raise ValueError(f"At least {cls.MIN_N} groups are required")
        if n_doses > cls.MAX_N:
            raise ValueError(f"A maximum of {cls.MAX_N} groups are allowed")
        for incidence, n in zip(values["incidences"], values["ns"], strict=True):
            if incidence > n:
                raise ValueError(f"Incidence cannot be greater than N ({incidence} > {n})")
        return values

    def deserialize(self) -> DichotomousDataset:
        ds = DichotomousDataset(
            doses=self.doses, ns=self.ns, incidences=self.incidences, **self.metadata.dict()
        )
        ds._plot_data = self.plotting
        return ds


class DichotomousCancerDataset(DichotomousDataset):
    """
    Dataset object for dichotomous cancer datasets.

    A dichotomous cancer dataset contains a list of 3 identically sized arrays of
    input values, for the dose, number of subjects, and incidences (subjects
    with a positive response).

    Example
    -------
    >>> dataset = bmds.DichotomousCancerDataset(
            doses=[0, 1.96, 5.69, 29.75],
            ns=[75, 49, 50, 49],
            incidences=[5, 1, 3, 14]
        )
    """

    MINIMUM_DOSE_GROUPS = 2
    dtype = constants.Dtype.DICHOTOMOUS_CANCER

    def _validate(self):
        length = len(self.doses)
        if not all(len(lst) == length for lst in [self.doses, self.ns, self.incidences]):
            raise ValueError("All input lists must be same length")

        if length != len(set(self.doses)):
            raise ValueError("Doses are not unique")

        if self.num_dose_groups < self.MINIMUM_DOSE_GROUPS:
            raise ValueError(
                f"Must have {self.MINIMUM_DOSE_GROUPS} or more dose groups after dropping doses"
            )

    def serialize(self) -> "DichotomousCancerDatasetSchema":
        return DichotomousCancerDatasetSchema(
            dtype=self.dtype,
            doses=self.doses,
            ns=self.ns,
            incidences=self.incidences,
            plotting=self.plot_data(),
            metadata=self.metadata,
        )


class DichotomousCancerDatasetSchema(DichotomousDatasetSchema):
    MIN_N: ClassVar = 2

    def deserialize(self) -> DichotomousCancerDataset:
        ds = DichotomousCancerDataset(
            doses=self.doses, ns=self.ns, incidences=self.incidences, **self.metadata.dict()
        )
        ds._plot_data = self.plotting
        return ds
