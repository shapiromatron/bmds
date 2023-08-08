import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from ...bmds3.constants import BMDS_BLANK_VALUE


def nan_to_default(val) -> float:
    # wrangle malformed floats
    return BMDS_BLANK_VALUE if isinstance(val, str) or np.isnan(val) else val


def multiprocess(jobs: list, run: Callable) -> list:
    """Run a list of jobs in multiprocessing

    Args:
        jobs (list): A list of jobs
        run (Callable): A callable executor

    Returns:
        list: A list of results
    """
    nprocs = max((os.cpu_count() or 2) - 1, 1)
    results = []
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        with tqdm(jobs, leave=False) as progress:
            futures = []
            for job in jobs:
                future = executor.submit(run, job)
                future.add_done_callback(lambda _: progress.update())
                futures.append(future)
            for future in futures:
                results.append(future.result())
    return results
