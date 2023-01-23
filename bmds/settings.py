import os
import platform

# How long should the system wait for a BMDS model execution to complete
# before closing (in seconds)
BMDS_MODEL_TIMEOUT_SECONDS = 30

# Maximum polynomial used when setting default models; generally only used when
# the number of dose groups is large in a dataset. This is so that if you have
# 15 dose-groups you won't try to model using a 14 degree polynomial model.
# maximum_poly =  min(dataset dose groups - 1, MAXIMUM_POLYNOMIAL_ORDER)
MAXIMUM_POLYNOMIAL_ORDER = 8

# In model recommendation; if range of BMDLs is less than 3x different, the
# model with the smallest AIC will be recommended, otherwise the model with the
# smallest BMDL will be recommended.
SUFFICIENTLY_CLOSE_BMDL = 3

# Only required if running BMDS on a non-Windows computer
if platform.system() != "Windows":
    BMDS_REQUEST_URL = os.environ.get("BMDS_REQUEST_URL")
    BMDS_TOKEN = os.environ.get("BMDS_TOKEN")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
        "simple": {"format": "%(levelname)s %(message)s"},
    },
    "handlers": {
        "logfile": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "bmds.log",
            "maxBytes": 50 * 1024 * 1024,
            "backupCount": 10,
            "formatter": "default",
        },
        "console": {"level": "DEBUG", "class": "logging.StreamHandler", "formatter": "simple"},
    },
    "loggers": {
        "": {"handlers": ["logfile"], "level": "INFO"},
    },
}

SIMPLE_SETTINGS = {"CONFIGURE_LOGGING": True}
