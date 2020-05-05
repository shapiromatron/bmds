class BMDSException(Exception):
    """An exception occurred in the BMDS module."""

    pass


class RemoteBMDSExecutionException(BMDSException):
    """There was an error in executing BMDS on the BMDS server"""

    pass
