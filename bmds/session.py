from . import datasets, constants, logic


class Session(object):

    bmr_options = {
        constants.DICHOTOMOUS: constants.DICHOTOMOUS_BMRS,
        constants.DICHOTOMOUS_CANCER: constants.DICHOTOMOUS_BMRS,
        constants.CONTINUOUS: constants.CONTINUOUS_BMRS
    }

    def __init__(self, dtype, dataset=None):
        self.dtype = dtype
        if self.dtype not in constants.DTYPES:
            raise ValueError('Invalid data type')
        self._models = []
        self.dataset = dataset

    @property
    def model_options(self):
        raise NotImplementedError('Abstract method requires implementation')

    def get_bmr_options(self):
        return self.bmr_options[self.dtype]

    def get_model_options(self):
        return [
            model.get_default()
            for model in self.model_options[self.dtype].values()
        ]

    def add_dataset(self, **kwargs):
        if self.dtype == constants.CONTINUOUS:
            ds = datasets.ContinuousDataset(**kwargs)
        elif self.dtype in constants.DICH_DTYPES:
            ds = datasets.DichotomousDataset(**kwargs)
        else:
            raise ValueError('Invalid dtype')
        self.dataset = ds

    @property
    def has_models(self):
        return len(self._models) > 0

    def add_default_models(self):
        for name in self.model_options[self.dtype].keys():
            self.add_model(name)

    def add_model(self, name, overrides=None, id=None):
        if self.dataset is None:
            raise ValueError('Add dataset to session before adding models')
        Model = self.model_options[self.dtype][name]
        instance = Model(
            dataset=self.dataset,
            overrides=overrides,
            id=id,
        )
        self._models.append(instance)

    def execute(self):
        for model in self._models:
            model.execute()

    def add_recommender(self, overrides=None):
        self.recommender = logic.Recommender(self.dtype, overrides)

    def recommend(self):
        if not hasattr(self, 'recommender'):
            self.add_recommender()
        self.recommended_model = self.recommender.recommend(self.dataset, self._models)
        return self.recommended_model
