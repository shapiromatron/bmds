import platform

from bmds.bmds2.sessions import BMDS_v270 as BmdsSession2
from bmds.bmds3.sessions import Bmds330 as BmdsSession3
from bmds.constants import Version

from .shared import multiprocess


class BmdsSessionWrapper:
    # TODO - wrapper shouldn't modify session object
    def __init__(self, bmds_version: Version, dataset, session_name: str = ""):
        if bmds_version == Version.BMDS270:
            self.session = BmdsSession2(dataset.dtype, dataset)
        elif bmds_version == Version.BMDS330:
            self.session = BmdsSession3(dataset)
        else:
            raise ValueError()
        self.session._bmds_version = bmds_version
        self.session._os = platform.system()
        self.session._session_name = session_name

    def add_model(self, model_name: str, settings=dict):
        # TODO - remove settings name; add to settings
        self.session.add_model(model_name, settings)
        # set attributes on model
        model = self.session.models[-1]
        model._model_name = model_name

    def execute_and_recommend(self) -> "BmdsSessionWrapper":
        self.session.execute_and_recommend()
        return self

    def add_models(self, model_names: list[str], settings=None, settings_name: str = ""):
        for model_name in model_names:
            self.add_model(model_name, settings, settings_name)

    @classmethod
    def bulk_execute_and_recommend(
        cls, session_wrappers: list["BmdsSessionWrapper"]
    ) -> list["BmdsSessionWrapper"]:
        return multiprocess(session_wrappers, cls.execute_and_recommend)
