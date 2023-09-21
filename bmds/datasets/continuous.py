import math
from typing import Annotated, ClassVar

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pydantic import Field, model_validator
from scipy import stats

from .. import constants, plotting
from ..stats.anova import AnovaTests
from ..utils import str_list
from .base import DatasetBase, DatasetMetadata, DatasetPlottingSchema, DatasetSchemaBase


class ContinuousSummaryDataMixin:
    def _validate_summary_data(self):
        length = len(self.doses)
        if not all(len(lst) == length for lst in [self.doses, self.ns, self.means, self.stdevs]):
            raise ValueError("All input lists must be same length")

        if length != len(set(self.doses)):
            raise ValueError("Doses are not unique")

        if self.num_dose_groups < self.MINIMUM_DOSE_GROUPS:
            raise ValueError(
                f"Must have {self.MINIMUM_DOSE_GROUPS} or more dose groups after dropping doses"
            )

    @property
    def is_increasing(self):
        # increasing or decreasing with respect to control?
        change = 0.0
        for i in range(1, len(self.means)):
            change += self.means[i] - self.means[0]
        return change >= 0

    @property
    def variances(self):
        if not hasattr(self, "_variances"):
            stds = np.array(self.stdevs)
            self._variances = np.power(stds, 2).tolist()
        return self._variances

    def anova(self) -> AnovaTests | None:
        if not hasattr(self, "_anova") and not hasattr(self, "_anova_attempted"):
            # Returns either a tuple of 3 Test objects, or None if anova failed
            try:
                num_params = 3  # assume linear model
                (A1, A2, A3, AR) = AnovaTests.compute_likelihoods(
                    self.num_dose_groups, self.ns, self.means, self.variances
                )
                self._anova = AnovaTests.get_anova_c3_tests(
                    num_params, self.num_dose_groups, A1, A2, A3, AR
                )
            except ValueError:
                self._anova = None
                self._anova_attempted = True
        return self._anova

    def get_anova_report(self):
        return AnovaTests.output_3tests(self.anova())

    @property
    def dataset_length(self):
        return self.num_dose_groups


class ContinuousDataset(ContinuousSummaryDataMixin, DatasetBase):
    """
    Dataset object for continuous datasets.

    A continuous dataset contains a list of 4 identically sized arrays of
    input values, for the dose, number of subjects, mean of response values for
    dose group, and standard-deviation of response for that dose group.

    Example
    -------
    >>> dataset = bmds.ContinuousDataset(
            doses=[0, 10, 50, 150, 400],
            ns=[25, 25, 24, 24, 24],
            means=[2.61, 2.81, 2.96, 4.66, 11.23],
            stdevs=[0.81, 1.19, 1.37, 1.72, 2.84]
        )
    """

    _BMDS_DATASET_TYPE = 1  # group data
    MINIMUM_DOSE_GROUPS = 3
    dtype = constants.Dtype.CONTINUOUS

    def __init__(
        self,
        doses: list[float],
        ns: list[float],
        means: list[float],
        stdevs: list[float],
        **metadata,
    ):
        self.doses = doses
        self.ns = ns
        self.means = means
        self.stdevs = stdevs
        self.metadata = DatasetMetadata.model_validate(metadata)
        self._sort_by_dose_group()
        self._validate()

    def _validate(self):
        self._validate_summary_data()

    def _sort_by_dose_group(self):
        # use mergesort since it's a stable-sorting algorithm in numpy
        indexes = np.array(self.doses).argsort(kind="mergesort")
        for fld in ("doses", "ns", "means", "stdevs"):
            arr = getattr(self, fld)
            setattr(self, fld, np.array(arr)[indexes].tolist())
        self._validate()

    def drop_dose(self):
        """
        Drop the maximum dose and related response values.
        """
        for fld in ("doses", "ns", "means", "stdevs"):
            arr = getattr(self, fld)[:-1]
            setattr(self, fld, arr)
        self._validate()

    def as_dfile(self):
        """
        Return the dataset representation in BMDS .(d) file.
        """
        rows = ["Dose NumAnimals Response Stdev"]
        for i, v in enumerate(self.doses):
            if i >= self.num_dose_groups:
                continue
            rows.append("%f %d %f %f" % (self.doses[i], self.ns[i], self.means[i], self.stdevs[i]))
        return "\n".join(rows)

    def errorbars(self):
        # 95% confidence interval
        if not hasattr(self, "_errorbars"):
            self._errorbars = [
                stats.t.ppf(0.975, max(n - 1, 1)) * stdev / np.sqrt(float(n))
                for stdev, n in zip(self.stdevs, self.ns, strict=True)
            ]
        return self._errorbars

    def get_plotting(self):
        errorbars = self.errorbars()
        return DatasetPlottingSchema(
            mean=self.means,
            ll=[mean - err for mean, err in zip(self.means, errorbars, strict=True)],
            ul=[mean + err for mean, err in zip(self.means, errorbars, strict=True)],
        )

    def plot(self) -> Figure:
        """
        Return a matplotlib figure of the dose-response dataset.

        Examples
        --------
        >>> fig = dataset.plot()
        >>> fig.show()

        .. image:: ../tests/data/mpl/test_cdataset_plot.png
           :align: center
           :alt: Example generated BMD plot

        Returns
        -------
        out : matplotlib.figure.Figure
            A matplotlib figure representation of the dataset.
        """
        ax = self.setup_plot()
        ax.errorbar(
            self.doses,
            self.means,
            yerr=self.errorbars(),
            label="Mean ± 95% CI",
            **plotting.DATASET_POINT_FORMAT,
        )
        ax.legend(**plotting.LEGEND_OPTS)
        return ax.get_figure()

    def serialize(self) -> "ContinuousDatasetSchema":
        anova = self.anova()
        return ContinuousDatasetSchema(
            dtype=self.dtype,
            doses=self.doses,
            ns=self.ns,
            means=self.means,
            stdevs=self.stdevs,
            anova=anova,
            plotting=self.get_plotting(),
            metadata=self.metadata,
        )

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        super().update_record(d)
        d.update(
            dataset_doses=str_list(self.doses),
            dataset_ns=str_list(self.ns),
            dataset_stdevs=str_list(self.stdevs),
            dataset_means=str_list(self.means),
        )

    def rows(self, extras: dict) -> list[dict]:
        """Return a list of rows; one for each item in a dataset"""
        metadata = self.metadata.model_dump()
        return [
            {**extras, **metadata, **dict(dose=dose, n=n, mean=mean, stdev=stdev)}
            for dose, n, mean, stdev in zip(
                self.doses, self.ns, self.means, self.stdevs, strict=True
            )
        ]


class ContinuousDatasetSchema(DatasetSchemaBase):
    dtype: constants.Dtype
    metadata: DatasetMetadata
    doses: list[Annotated[float, Field(ge=0)]]
    ns: list[Annotated[float, Field(gt=0)]]
    means: list[float]
    stdevs: list[Annotated[float, Field(ge=0)]]
    anova: AnovaTests | None = None
    plotting: DatasetPlottingSchema | None = None

    MIN_N: ClassVar = 3
    MAX_N: ClassVar = math.inf

    @model_validator(mode="after")
    def num_groups(self):
        n_doses = len(self.doses)
        n_ns = len(self.ns)
        n_means = len(self.means)
        n_stdevs = len(self.stdevs)
        if len(set([n_doses, n_ns, n_means, n_stdevs])) > 1:
            raise ValueError("Length of doses, ns, means, and stdevs are not the same")
        if n_doses < self.MIN_N:
            raise ValueError(f"At least {self.MIN_N} groups are required")
        if n_doses > self.MAX_N:
            raise ValueError(f"A maximum of {self.MAX_N} groups are allowed")
        return self

    def deserialize(self) -> ContinuousDataset:
        ds = ContinuousDataset(
            doses=self.doses,
            ns=self.ns,
            means=self.means,
            stdevs=self.stdevs,
            **self.metadata.model_dump(),
        )
        ds._anova = self.anova
        return ds


class ContinuousIndividualDataset(ContinuousSummaryDataMixin, DatasetBase):
    """
    Dataset object for continuous individual datasets.

    A continuous individual dataset contains a list of 2 identically sized
    arrays of input values, one for the dose and one for the response of an
    individual test-subject.

    Example
    -------
    >>> dataset = bmds.ContinuousIndividualDataset(
            doses=[
                0, 0, 0, 0, 0, 0, 0, 0,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                1, 1, 1, 1, 1, 1,
                10, 10, 10, 10, 10, 10,
                100, 100, 100, 100, 100, 100,
                300, 300, 300, 300, 300, 300,
                500, 500, 500, 500, 500, 500,
            ],
            responses=[
                8.1079, 9.3063, 9.7431, 9.781, 10.052, 10.613, 10.751, 11.057,
                9.1556, 9.6821, 9.8256, 10.2095, 10.2222, 12.0382,
                9.5661, 9.7059, 9.9905, 10.2716, 10.471, 11.0602,
                8.8514, 10.0107, 10.0854, 10.5683, 11.1394, 11.4875,
                9.5427, 9.7211, 9.8267, 10.0231, 10.1833, 10.8685,
                10.368, 10.5176, 11.3168, 12.002, 12.1186, 12.6368,
                9.9572, 10.1347, 10.7743, 11.0571, 11.1564, 12.0368
            ]
        )
    """

    _BMDS_DATASET_TYPE = 0  # individual data
    MINIMUM_DOSE_GROUPS = 3
    dtype = constants.Dtype.CONTINUOUS_INDIVIDUAL

    def __init__(self, doses: list[float], responses: list[float], **metadata):
        data = self._prepare_summary_data(doses, responses)
        for key, value in data.items():
            setattr(self, key, value)
        self.metadata = DatasetMetadata.model_validate(metadata)
        self._validate()

    def _prepare_summary_data(self, individual_doses, responses):
        data = pd.DataFrame(
            data=dict(individual_doses=individual_doses, responses=responses)
        ).groupby("individual_doses")
        return dict(
            individual_doses=individual_doses,
            responses=responses,
            doses=list(data.groups.keys()),
            ns=data.responses.count().tolist(),
            means=data.responses.mean().tolist(),
            stdevs=data.responses.std(ddof=0).fillna(0).tolist(),
        )

    def _validate(self):
        length = len(self.individual_doses)
        if not all(len(lst) == length for lst in [self.individual_doses, self.responses]):
            raise ValueError("All input lists must be same length")

        if self.num_dose_groups < self.MINIMUM_DOSE_GROUPS:
            raise ValueError(
                f"Must have {self.MINIMUM_DOSE_GROUPS} or more dose groups after dropping doses"
            )

        self._validate_summary_data()

    def drop_dose(self):
        """
        Drop the maximum dose and related response values.
        """
        individual_doses = np.array(self.individual_doses)
        responses = np.array(self.responses)

        mask = individual_doses != individual_doses.max()
        doses = individual_doses[mask].tolist()
        responses = responses[mask].tolist()

        summary_data = self._prepare_summary_data(doses, responses)
        for key, value in summary_data.items():
            setattr(self, key, value)

        self._validate()

    def as_dfile(self):
        """
        Return the dataset representation in BMDS .(d) file.
        """
        rows = ["Dose Response"]
        for dose, response in zip(self.individual_doses, self.responses, strict=True):
            dose_idx = self.doses.index(dose)
            if dose_idx >= self.num_dose_groups:
                continue
            rows.append(f"{dose:f} {response:f}")
        return "\n".join(rows)

    def get_responses_by_dose(self):
        doses = np.array(self.individual_doses)
        resps = np.array(self.responses)
        return sorted([resps[doses == dose].tolist() for dose in self.doses])

    @property
    def dataset_length(self):
        return len(self.individual_doses)

    def plot(self) -> Figure:
        """
        Return a matplotlib figure of the dose-response dataset.

        Examples
        --------
        >>> fig = dataset.plot()
        >>> fig.show()
        >>> fig.clear()

        .. image:: ../tests/data/mpl/test_cidataset_plot.png
           :align: center
           :alt: Example generated BMD plot

        Returns
        -------
        out : matplotlib.figure.Figure
            A matplotlib figure representation of the dataset.
        """
        ax = self.setup_plot()
        ax.scatter(
            self.individual_doses,
            self.responses,
            label="Data",
            **plotting.DATASET_INDIVIDUAL_FORMAT,
        )
        ax.legend(**plotting.LEGEND_OPTS)
        return ax.get_figure()

    def serialize(self) -> "ContinuousIndividualDatasetSchema":
        anova = self.anova()
        return ContinuousIndividualDatasetSchema(
            dtype=self.dtype,
            doses=self.individual_doses,
            responses=self.responses,
            anova=anova,
            metadata=self.metadata,
        )

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        super().update_record(d)
        d.update(
            dataset_individual_doses=str_list(self.individual_doses),
            dataset_responses=str_list(self.responses),
            dataset_doses=str_list(self.doses),
            dataset_ns=str_list(self.ns),
            dataset_stdevs=str_list(self.stdevs),
            dataset_means=str_list(self.means),
        )

    def rows(self, extras: dict) -> list[dict]:
        """Return a list of rows; one for each item in a dataset"""
        metadata = self.metadata.model_dump()
        return [
            {**extras, **metadata, **dict(dose=dose, response=response)}
            for dose, response in zip(self.individual_doses, self.responses, strict=True)
        ]


class ContinuousIndividualDatasetSchema(DatasetSchemaBase):
    dtype: constants.Dtype
    metadata: DatasetMetadata
    doses: list[Annotated[float, Field(ge=0)]]
    responses: list[float]
    anova: AnovaTests | None = None

    MIN_N: ClassVar = 3
    MAX_N: ClassVar = math.inf

    @model_validator(mode="after")
    def num_groups(self):
        n_doses = len(self.doses)
        n_responses = len(self.responses)
        if len(set([n_doses, n_responses])) > 1:
            raise ValueError("Length of doses and responses are not the same")
        if n_doses < self.MIN_N:
            raise ValueError(f"At least {self.MIN_N} groups are required")
        if n_doses > self.MAX_N:
            raise ValueError(f"A maximum of {self.MAX_N} groups are allowed")
        # may throw ValueErrors; caught in validator
        ContinuousIndividualDataset(doses=self.doses, responses=self.responses)
        return self

    def deserialize(self) -> ContinuousIndividualDataset:
        ds = ContinuousIndividualDataset(
            doses=self.doses,
            responses=self.responses,
            **self.metadata.model_dump(),
        )
        ds._anova = self.anova
        return ds


ContinuousDatasets = ContinuousDataset | ContinuousIndividualDataset
ContinuousDatasetSchemas = ContinuousDatasetSchema | ContinuousIndividualDatasetSchema
