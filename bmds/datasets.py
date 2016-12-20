from collections import defaultdict
import numpy as np
from scipy import stats

import matplotlib as mpl
mpl.use('agg')  # noqa (required before the pyplot import)
import matplotlib.pyplot as plt

from .anova import AnovaTests


# GLOBAL plot overrides
plt.style.use('grayscale')
mpl.rcParams.update({'font.size': 10})
PLOT_FIGSIZE = (8, 5)
PLOT_MARGINS = 0.05


class Dataset(object):

    def validate(self):
        raise NotImplemented('Abstract method; Requires implementation')

    def as_dfile(self):
        raise NotImplemented('Abstract method; Requires implementation')

    def _to_dict(self):
        raise NotImplemented('Abstract method; Requires implementation')

    def plot(self):
        raise NotImplemented('Abstract method; Requires implementation')


class DichotomousDataset(Dataset):

    BMDS_DATASET_TYPE = 1  # group data

    def __init__(self, doses, ns, incidences, doses_dropped=0):
        self.doses = doses
        self.ns = ns
        self.incidences = incidences
        self.doses_dropped = doses_dropped
        self.num_doses = len(doses)
        self.doses_used = self.num_doses - self.doses_dropped
        self.remainings = [n - p for n, p in zip(ns, incidences)]
        self.validate()

    def validate(self):
        length = len(self.doses)
        if not all(
                len(lst) == length for lst in
                [self.doses, self.ns, self.incidences]):
            raise ValueError('All input lists must be same length')

        if length != len(set(self.doses)):
            raise ValueError('Doses are not unique')

        if self.doses_used < 3:
            raise ValueError('Must have 3 or more doses after dropping doses')

    def as_dfile(self):
        rows = ['Dose Incidence NEGATIVE_RESPONSE']
        for i, v in enumerate(self.doses):
            if i >= self.doses_used:
                continue
            rows.append('%f %d %d' % (
                self.doses[i], self.incidences[i], self.remainings[i]))
        return '\n'.join(rows)

    @property
    def dataset_length(self):
        return self.doses_used

    def _to_dict(self):
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

    def set_plot_data(self):
        if hasattr(self, '_means'):
            return
        self._means, self._lls, self._uls = zip(*[
            self._calculate_plotting(i, j)
            for i, j in zip(self.ns, self.incidences)
        ])

    def plot(self):
        self.set_plot_data()
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.set_xlabel('Dose')
        ax.set_ylabel('Response')
        plt.errorbar(
            self.doses, self._means, yerr=[self._lls, self._uls],
            fmt='o')
        plt.margins(PLOT_MARGINS)
        return plt


class ContinuousDataset(Dataset):

    BMDS_DATASET_TYPE = 1  # group data

    def __init__(self, doses, ns, means, stdevs, doses_dropped=0):
        self.doses = doses
        self.ns = ns
        self.means = means
        self.stdevs = stdevs
        self.doses_dropped = doses_dropped
        self.num_doses = len(doses)
        self.doses_used = self.num_doses - self.doses_dropped
        self.validate()

    def validate(self):
        length = len(self.doses)
        if not all(
                len(lst) == length for lst in
                [self.doses, self.ns, self.means, self.stdevs]):
            raise ValueError('All input lists must be same length')

        if length != len(set(self.doses)):
            raise ValueError('Doses are not unique')

        if self.doses_used < 3:
            raise ValueError('Must have 3 or more doses after dropping doses')

    @property
    def is_increasing(self):
        inc = 0
        for i in range(len(self.means) - 1):
            if self.means[i + 1] > self.means[i]:
                inc += 1
            else:
                inc -= 1
        return inc >= 0

    def as_dfile(self):
        rows = ['Dose NumAnimals Response Stdev']
        for i, v in enumerate(self.doses):
            if i >= self.doses_used:
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
        if not hasattr(self, '_anova'):
            num_params = 3  # assume linear model
            (A1, A2, A3, AR) = AnovaTests.compute_likelihoods(
                self.doses_used, self.ns, self.means, self.variances)
            tests = AnovaTests.get_anova_c3_tests(
                num_params, self.doses_used, A1, A2, A3, AR)
            self._anova = tests
        return self._anova

    @property
    def dataset_length(self):
        return self.doses_used

    def get_anova_report(self):
        return AnovaTests.output_3tests(self.anova)

    def _to_dict(self):
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
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.set_xlabel('Dose')
        ax.set_ylabel('Response')
        plt.errorbar(
            self.doses, self.means, yerr=self.errorbars,
            fmt='o')
        plt.margins(PLOT_MARGINS)
        return plt


class ContinuousIndividualDataset(ContinuousDataset):

    BMDS_DATASET_TYPE = 0  # individual data

    def __init__(self, doses, responses, doses_dropped=0):
        self.individual_doses = doses
        self.responses = responses
        self.doses_dropped = doses_dropped
        self.set_summary_data()
        self.num_doses = len(self.doses)
        self.doses_used = self.num_doses - self.doses_dropped
        self.validate()

    def validate(self):
        length = len(self.individual_doses)
        if not all(
                len(lst) == length for lst in
                [self.individual_doses, self.responses]):
            raise ValueError('All input lists must be same length')

        if self.doses_used < 3:
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

    def as_dfile(self):
        rows = ['Dose Response']
        for dose, response in zip(self.individual_doses, self.responses):
            dose_idx = self.doses.index(dose)
            if dose_idx >= self.doses_used:
                continue
            rows.append('%f %f' % (dose, response))
        return '\n'.join(rows)

    @property
    def dataset_length(self):
        return len(self.individual_doses)

    def _to_dict(self):
        return dict(
            individual_doses=self.individual_doses,
            responses=self.responses,
        )

    def plot(self):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.set_xlabel('Dose')
        ax.set_ylabel('Response')
        plt.scatter(
            self.individual_doses, self.responses,
            s=35, alpha=0.60, c='k')
        plt.margins(PLOT_MARGINS)
        return plt
