from collections import defaultdict
import numpy as np
from scipy import stats

from . import plotting
from .anova import AnovaTests

__all__ = [
    'DichotomousDataset',
    'ContinuousDataset',
    'ContinuousIndividualDataset',
]


class Dataset(object):
    # Abstract parent-class for dataset-types.

    def _validate(self):
        raise NotImplemented('Abstract method; requires implementation')

    def as_dfile(self):
        raise NotImplemented('Abstract method; requires implementation')

    def to_dict(self):
        raise NotImplemented('Abstract method; requires implementation')

    def plot(self):
        raise NotImplemented('Abstract method; requires implementation')

    def drop_dose(self):
        raise NotImplemented('Abstract method; requires implementation')

    @property
    def num_dose_groups(self):
        return len(set(self.doses))


class DichotomousDataset(Dataset):
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

    def __init__(self, doses, ns, incidences):
        self.doses = doses
        self.ns = ns
        self.incidences = incidences
        self.remainings = [n - p for n, p in zip(ns, incidences)]
        self._sort_by_dose_group()
        self._validate()

    def _sort_by_dose_group(self):
        # use mergesort since it's a stable-sorting algorithm in numpy
        indexes = np.array(self.doses).argsort(kind='mergesort')
        for fld in ('doses', 'ns', 'incidences', 'remainings'):
            arr = getattr(self, fld)
            setattr(self, fld, np.array(arr)[indexes].tolist())
        self._validate()

    def _validate(self):
        length = len(self.doses)
        if not all(
                len(lst) == length for lst in
                [self.doses, self.ns, self.incidences]):
            raise ValueError('All input lists must be same length')

        if length != len(set(self.doses)):
            raise ValueError('Doses are not unique')

        if self.num_dose_groups < 3:
            raise ValueError('Must have 3 or more dose groups after dropping doses')

    def drop_dose(self):
        """
        Drop the maximum dose and related response values.
        """
        for fld in ('doses', 'ns', 'incidences', 'remainings'):
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
        rows = ['Dose Incidence NEGATIVE_RESPONSE']
        for i, v in enumerate(self.doses):
            if i >= self.num_dose_groups:
                continue
            rows.append('%f %d %d' % (
                self.doses[i], self.incidences[i], self.remainings[i]))
        return '\n'.join(rows)

    @property
    def dataset_length(self):
        """
        Return the length of the vector of doses-used.
        """
        return self.num_dose_groups

    def to_dict(self):
        """
        Returns a dictionary representation of the dataset.
        """
        return dict(
            doses=self.doses,
            ns=self.ns,
            incidences=self.incidences,
        )

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
        q = 1. - p
        ll = ((2 * n * p + 2 * z - 1) - z *
              np.sqrt(2 * z - (2 + 1 / n) + 4 * p * (n * q + 1))) / (2 * (n + 2 * z))
        ul = ((2 * n * p + 2 * z + 1) + z *
              np.sqrt(2 * z + (2 + 1 / n) + 4 * p * (n * q - 1))) / (2 * (n + 2 * z))
        return p, ll, ul

    def _set_plot_data(self):
        if hasattr(self, '_means'):
            return
        self._means, self._lls, self._uls = zip(*[
            self._calculate_plotting(i, j)
            for i, j in zip(self.ns, self.incidences)
        ])

    def plot(self):
        """
        Return a matplotlib figure of the dose-response dataset.

        Examples
        --------
        >>> fig = dataset.plot()
        >>> fig.show()

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
        ax.set_xlabel('Dose')
        ax.set_ylabel('Response')
        ax.errorbar(
            self.doses, self._means, yerr=[self._lls, self._uls],
            **plotting.DATASET_POINT_FORMAT)
        ax.margins(plotting.PLOT_MARGINS)
        return fig


class ContinuousDataset(Dataset):
    """
    Dataset object for continuous datasets.

    A continuous dataset contains a list of 4 identically sized arrays of
    input values, for the dose, number of subjects, mean of response values for
    dose group, and standard-devation of response for that dose group.

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

    def __init__(self, doses, ns, means, stdevs):
        self.doses = doses
        self.ns = ns
        self.means = means
        self.stdevs = stdevs
        self._sort_by_dose_group()
        self._validate()

    def _sort_by_dose_group(self):
        # use mergesort since it's a stable-sorting algorithm in numpy
        indexes = np.array(self.doses).argsort(kind='mergesort')
        for fld in ('doses', 'ns', 'means', 'stdevs'):
            arr = getattr(self, fld)
            setattr(self, fld, np.array(arr)[indexes].tolist())
        self._validate()

    def _validate(self):
        length = len(self.doses)
        if not all(
                len(lst) == length for lst in
                [self.doses, self.ns, self.means, self.stdevs]):
            raise ValueError('All input lists must be same length')

        if length != len(set(self.doses)):
            raise ValueError('Doses are not unique')

        if self.num_dose_groups < 3:
            raise ValueError('Must have 3 or more dose groups after dropping doses')

    @property
    def is_increasing(self):
        inc = 0
        for i in range(len(self.means) - 1):
            if self.means[i + 1] > self.means[i]:
                inc += 1
            else:
                inc -= 1
        return inc >= 0

    def drop_dose(self):
        """
        Drop the maximum dose and related response values.
        """
        for fld in ('doses', 'ns', 'means', 'stdevs'):
            arr = getattr(self, fld)[:-1]
            setattr(self, fld, arr)
        self._validate()

    def as_dfile(self):
        """
        Return the dataset representation in BMDS .(d) file.
        """
        rows = ['Dose NumAnimals Response Stdev']
        for i, v in enumerate(self.doses):
            if i >= self.num_dose_groups:
                continue
            rows.append('%f %d %f %f' % (
                self.doses[i], self.ns[i], self.means[i], self.stdevs[i]))
        return '\n'.join(rows)

    @property
    def variances(self):
        if not hasattr(self, '_variances'):
            stds = np.array(self.stdevs)
            self._variances = np.power(stds, 2).tolist()
        return self._variances

    @property
    def anova(self):
        # Either be a tuple of 3 Test objects, or None if anova failed
        if not hasattr(self, '_anova'):
            try:
                num_params = 3  # assume linear model
                (A1, A2, A3, AR) = AnovaTests.compute_likelihoods(
                    self.num_dose_groups, self.ns, self.means, self.variances)
                tests = AnovaTests.get_anova_c3_tests(
                    num_params, self.num_dose_groups, A1, A2, A3, AR)
            except ValueError:
                tests = None
            self._anova = tests
        return self._anova

    @property
    def dataset_length(self):
        return self.num_dose_groups

    def get_anova_report(self):
        return AnovaTests.output_3tests(self.anova)

    def to_dict(self):
        """
        Return a dictionary representation of the dataset.
        """
        return dict(
            doses=self.doses,
            ns=self.ns,
            means=self.means,
            stdevs=self.stdevs,
        )

    @property
    def errorbars(self):
        if not hasattr(self, '_errorbars'):
            self._errorbars = [
                stats.t.ppf(0.975, max(n - 1, 1)) * stdev / np.sqrt(float(n))
                for stdev, n in zip(self.stdevs, self.ns)
            ]
        return self._errorbars

    def plot(self):
        """
        Return a matplotlib figure of the dose-response dataset.

        Examples
        --------
        >>> fig = dataset.plot()
        >>> fig.show()

        .. image:: ../tests/resources/test_cdataset_plot.png
           :align: center
           :alt: Example generated BMD plot

        Returns
        -------
        out : matplotlib.figure.Figure
            A matplotlib figure representation of the dataset.
        """
        fig = plotting.create_empty_figure()
        ax = fig.gca()
        ax.set_xlabel('Dose')
        ax.set_ylabel('Response')
        ax.errorbar(
            self.doses, self.means, yerr=self.errorbars,
            **plotting.DATASET_POINT_FORMAT)
        ax.margins(plotting.PLOT_MARGINS)
        return fig


class ContinuousIndividualDataset(ContinuousDataset):
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

    def __init__(self, doses, responses):
        self.individual_doses = doses
        self.responses = responses
        self._sort_by_dose_group()
        self.set_summary_data()
        self._validate()

    def _sort_by_dose_group(self):
        # use mergesort since it's a stable-sorting algorithm in numpy
        indexes = np.array(self.individual_doses).argsort(kind='mergesort')
        for fld in ('individual_doses', 'responses'):
            arr = getattr(self, fld)
            setattr(self, fld, np.array(arr)[indexes].tolist())

    def _validate(self):
        length = len(self.individual_doses)
        if not all(
                len(lst) == length for lst in
                [self.individual_doses, self.responses]):
            raise ValueError('All input lists must be same length')

        if self.num_dose_groups < 3:
            raise ValueError('Must have 3 or more doses after dropping doses')

    def set_summary_data(self):
        doses = list(set(self.individual_doses))
        doses.sort()

        dd = defaultdict(list)
        for d, r in zip(self.individual_doses, self.responses):
            dd[d].append(r)

        def _get_stats(lst):
            arr = np.array(lst, dtype=np.float64)
            return [arr.size, arr.mean(), arr.std()]

        vals = [_get_stats(dd[dose]) for dose in doses]
        self.ns, self.means, self.stdevs = zip(*vals)
        self.doses = doses

    def drop_dose(self):
        """
        Drop the maximum dose and related response values.
        """
        doses = np.array(self.individual_doses)
        responses = np.array(self.responses)
        mask = (doses != doses.max())
        self.individual_doses = doses[mask].tolist()
        self.responses = responses[mask].tolist()
        self.set_summary_data()
        self._validate()

    def as_dfile(self):
        """
        Return the dataset representation in BMDS .(d) file.
        """
        rows = ['Dose Response']
        for dose, response in zip(self.individual_doses, self.responses):
            dose_idx = self.doses.index(dose)
            if dose_idx >= self.num_dose_groups:
                continue
            rows.append('%f %f' % (dose, response))
        return '\n'.join(rows)

    @property
    def dataset_length(self):
        return len(self.individual_doses)

    def to_dict(self):
        """
        Return a dictionary representation of the dataset.
        """
        return dict(
            individual_doses=self.individual_doses,
            responses=self.responses,
        )

    def plot(self):
        """
        Return a matplotlib figure of the dose-response dataset.

        Examples
        --------
        >>> fig = dataset.plot()
        >>> fig.show()

        .. image:: ../tests/resources/test_cidataset_plot.png
           :align: center
           :alt: Example generated BMD plot

        Returns
        -------
        out : matplotlib.figure.Figure
            A matplotlib figure representation of the dataset.
        """
        fig = plotting.create_empty_figure()
        ax = fig.gca()
        ax.set_xlabel('Dose')
        ax.set_ylabel('Response')
        ax.scatter(
            self.individual_doses, self.responses,
            **plotting.DATASET_INDIVIDUAL_FORMAT)
        ax.margins(plotting.PLOT_MARGINS)
        return fig
