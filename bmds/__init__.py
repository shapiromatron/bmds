import os

os.environ.setdefault("SIMPLE_SETTINGS", "bmds.settings")

from . import bmds2, constants  # noqa
from .datasets import *  # noqa
from .plotting import *  # noqa
from .session import *  # noqa
from .utils import citation  # noqa
from .version import __version__  # noqa
