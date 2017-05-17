import os
import platform

# How long should the system wait for a BMDS model execution to complete
# before closing (in seconds)
BMDS_MODEL_TIMEOUT_SECONDS = 10

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
if platform.system() != 'Windows':

    # The URL for the BMDS webserver
    BMDS_HOST = os.environ.get('BMDS_HOST')

    # The admin username for the BMDS webserver.
    BMDS_USERNAME = os.environ.get('BMDS_USERNAME')

    # The admin password for the BMDS webserver.
    BMDS_PASSWORD = os.environ.get('BMDS_PASSWORD')

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        },
    },
    'handlers': {
        'logfile': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'bmds.log',
            'maxBytes': 50 * 1024 * 1024,
            'backupCount': 10,
            'formatter': 'default'
        },
    },
    'loggers': {
        '': {
            'handlers': ['logfile'],
            'level': 'INFO',
        },
        'requests.packages.urllib3': {
            'handlers': ['logfile'],
            'level': 'INFO',
        }
    }
}

SIMPLE_SETTINGS = {
    'CONFIGURE_LOGGING': True,
}
