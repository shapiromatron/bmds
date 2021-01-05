from typing import List

import numpy as np
from scipy import stats
from simple_settings import settings

from .. import constants, plotting
from .base import DatasetBase


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

    doses: List[float]
    ns: List[float]
    incidences: List[float]

    _BMDS_DATASET_TYPE = 1  # group data
    MINIMUM_DOSE_GROUPS = 3
    dtype = constants.Dtype.DICHOTOMOUS

    def __init__(self, doses, ns, incidences, **kwargs):
        super().__init__(doses=doses, ns=ns, incidences=incidences)
        self.remainings = [n - p for n, p in zip(ns, incidences)]
        self.kwargs = kwargs
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
        Add confidence intervals to dichotomous datasets. From bmds231_manual.pdf, pg 124-5.

        LL = {(2np + z2 - 1) - z*sqrt[z2 - (2+1/n) + 4p(nq+1)]}/[2*(n+z2)]
        UL = {(2np + z2 + 1) + z*sqrt[z2 + (2-1/n) + 4p(nq-1)]}/[2*(n+z2)]

        - p = the observed proportion
        - n = the total number in the group in question
        - z = Z(1-alpha/2) is the inverse standard normal cumulative
              distribution function evaluated at 1-alpha/2
        - q = 1-p.

        The error bars shown in BMDS plots use alpha = 0.05 and so
        represent the 95% confidence intervals on the observed
        proportions (independent of model).
        """
        p = incidence / float(n)
        z = stats.norm.ppf(0.975)
        q = 1.0 - p
        ll = ((2 * n * p + 2 * z - 1) - z * np.sqrt(2 * z - (2 + 1 / n) + 4 * p * (n * q + 1))) / (
            2 * (n + 2 * z)
        )
        ul = ((2 * n * p + 2 * z + 1) + z * np.sqrt(2 * z + (2 + 1 / n) + 4 * p * (n * q - 1))) / (
            2 * (n + 2 * z)
        )
        return p, ll, ul

    def _set_plot_data(self):
        if hasattr(self, "_means"):
            return
        self._means, self._lls, self._uls = zip(
            *[self._calculate_plotting(i, j) for i, j in zip(self.ns, self.incidences)]
        )

    def plot(self):
        """
        Return a matplotlib figure of the dose-response dataset.

        Examples
        --------
        >>> fig = dataset.plot()
        >>> fig.show()
        >>> fig.clear()

        .. image:: ../tests/resources/test_ddataset_plot.png
           :align: center
           :alt: Example generated BMD plot

        Returns
        -------
        out : matplotlib.figure.Figure
            A matplotlib figure representation of the dataset.
        """
        self._set_plot_data()
        fig = plotting.create_empty_figure()
        ax = fig.gca()
        xlabel = self.kwargs.get("xlabel", "Dose")
        ylabel = self.kwargs.get("ylabel", "Fraction affected")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.errorbar(
            self.doses,
            self._means,
            yerr=[self._lls, self._uls],
            label="Fraction affected ± 95% CI",
            **plotting.DATASET_POINT_FORMAT,
        )
        ax.margins(plotting.PLOT_MARGINS)
        ax.set_title(self._get_dataset_name())
        ax.legend(**settings.LEGEND_OPTS)
        return fig


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
    dtype = constants.Dtype.CONTINUOUS_INDIVIDUAL

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
