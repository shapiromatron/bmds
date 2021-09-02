import os


# TODO remove when dll released
class RunBmds3:
    should_run = os.getenv("CI") is None
    skip_reason = "DLLs not present on CI"
