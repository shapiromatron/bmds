"""bmds
    Python BMDS library

isort:skip_file
"""
__version__ = "1.0.0.dev"

import os

os.environ.setdefault("SIMPLE_SETTINGS", "bmds.settings")  # noqa

from . import bmds2, constants  # noqa
from .datasets import *  # noqa
from .plotting import *  # noqa
from .session import *  # noqa
from .utils import citation  # noqa
