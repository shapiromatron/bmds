from typing import Any, ClassVar

from . import constants
from .bmds2.sessions import BMDS_v270
from .bmds3.sessions import Bmds330


class BMDS:
    """
    A single dataset, related models, outputs, and model recommendations.
    """

    _BMDS_VERSIONS: ClassVar[dict[str, Any]] = {
        constants.BMDS270: BMDS_v270,
        constants.BMDS330: Bmds330,
    }

    @classmethod
    def get_model(cls, version: str, model_name: str):
        """
        Return BMDS model class given BMDS version and model-name.
        """
        models = cls._BMDS_VERSIONS[version].model_options
        for keystore in models.values():
            if model_name in keystore:
                return keystore[model_name]
        raise ValueError("Unknown model name")

    @classmethod
    def get_versions(cls):
        return cls._BMDS_VERSIONS.keys()

    @classmethod
    def version(cls, version: str, *args, **kwargs):
        """
        Return a BMDS session of the specified version. If additional
        arguments are provided, an instance of this class is generated.
        """
        cls = cls._BMDS_VERSIONS[version]
        if len(args) > 0 or len(kwargs) > 0:
            return cls(*args, **kwargs)
        return cls

    @classmethod
    def latest_version(cls, *args, **kwargs):
        """
        Return the class of the latest version of the BMDS. If additional
        arguments are provided, an instance of this class is generated.
        """
        latest = list(cls._BMDS_VERSIONS.keys())[-1]
        return cls.version(latest, *args, **kwargs)
