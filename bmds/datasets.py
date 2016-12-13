from collections import defaultdict
import numpy as np

from .anova import AnovaTests


class Dataset(object):

    def validate(self):
        raise NotImplemented('Abstract method; Requires implementation')

    def as_dfile(self):
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
