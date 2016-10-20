class DichotomousDataset(object):

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


class ContinuousDataset(object):

    def __init__(self, doses, ns, responses, stdevs, doses_dropped=0):
        self.doses = doses
        self.ns = ns
        self.responses = responses
        self.stdevs = stdevs
        self.doses_dropped = doses_dropped
        self.num_doses = len(doses)
        self.doses_used = self.num_doses - self.doses_dropped
        self.validate()

    def validate(self):
        length = len(self.doses)
        if not all(
                len(lst) == length for lst in
                [self.doses, self.ns, self.responses, self.stdevs]):
            raise ValueError('All input lists must be same length')

        if self.doses_used < 3:
            raise ValueError('Must have 3 or more doses after dropping doses')

    def as_dfile(self):
        rows = ['Dose NumAnimals Response Stdev']
        for i, v in enumerate(self.doses):
            if i >= self.doses_used:
                continue
            rows.append('%f %d %f %f' % (
                self.doses[i], self.ns[i], self.responses[i], self.stdevs[i]))
        return '\n'.join(rows)
