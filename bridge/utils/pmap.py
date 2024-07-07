"""
pmap: Parallel Mapping Utility

This module provides a utility function, pmap, to apply a given function to a list of arguments in parallel.
It supports multiple backends for parallel processing, including 'concurrent', 'joblib', and 'dataloader' (PyTorch).

Usage:
    To use this utility, simply call the pmap function with the desired method, arguments list, and optional parameters.

Example:

    from pmap import pmap

    def square(x):
        return x * x

    results = pmap(square, [1, 2, 3, 4, 5])
    print(results)  # Output: [1, 4, 9, 16, 25]


Functions:
    - pmap: Parallel map function to apply a method to a list of arguments in parallel.
    - _pmap_dataloader: Internal function for parallel mapping using PyTorch DataLoader.
    - _pmap_concurrent_futures: Internal function for parallel mapping using concurrent.futures.
    - _pmap_joblib: Internal function for parallel mapping using Joblib.

"""

import concurrent.futures
import functools
import os
import pickle
from typing import Callable, Sequence

from tqdm import tqdm


def pmap(
    function: Callable,
    iterable: Sequence,
    progress_bar: bool = True,
    n_jobs: int = os.cpu_count(),
    backend: str = "concurrent",
) -> Sequence:
    """
    Apply a given method to a list of arguments in parallel, return outputs in the order of the input.

    Args:
        function (Callable): Method to apply.
        iterable (Sequence): Sequence of arguments.
        progress_bar (bool, optional): Show progress bar. Defaults to True.
        n_jobs (int, optional): Number of parallel jobs. Defaults to CPU count.
        backend (str, optional): Parallel backend ('concurrent', 'joblib', or 'dataloader' (PyTorch)).
        Defaults to 'concurrent'.

    Returns:
        Sequence: Sequence of results.
    """
    if n_jobs == 0:
        return [function(params) for params in (tqdm(iterable) if progress_bar else iterable)]
    else:
        if backend == "joblib":
            return _pmap_joblib(function, iterable, n_jobs, progress_bar)
        elif backend == "concurrent":
            return _pmap_concurrent_futures(function, iterable, n_jobs, progress_bar)
        elif backend == "dataloader":
            return _pmap_dataloader(function, iterable, n_jobs, progress_bar)


def _pmap_dataloader(function: Callable, iterable: Sequence, n_jobs: int, progress_bar: bool):
    """
    Internal function for parallel mapping using PyTorch DataLoader.

    Args:
        function (Callable): Method to apply.
        iterable (Sequence): Sequence of arguments.
        n_jobs (int): Number of parallel jobs.
        progress_bar (bool): Show progress bar.

    Returns:
        Sequence: Sequence of results.
    """
    from torch.utils.data import DataLoader, Dataset

    class DummyDataset(Dataset):
        def __init__(self, fn: Callable, args: Sequence):
            self._method = fn
            self._args_list = args

        def __getitem__(self, item):
            args = self._args_list[item]
            return self._method(args)

        def __len__(self):
            return len(self._args_list)

    def collate(data):
        return data

    ds = DummyDataset(function, iterable)
    dataloader = DataLoader(
        ds, num_workers=n_jobs, collate_fn=collate, batch_size=None, batch_sampler=None, prefetch_factor=10
    )
    if progress_bar:
        return list(tqdm(dataloader, total=len(ds), position=0, leave=True))
    else:
        return list(dataloader)


def _pmap_concurrent_futures(function: Callable, iterable: Sequence, n_jobs, progress_bar: bool):
    """
    Internal function for parallel mapping using concurrent.futures.

    Args:
        function (Callable): Method to apply.
        iterable (Sequence): Sequence of arguments.
        n_jobs (int): Number of parallel jobs.
        progress_bar (bool): Show progress bar.

    Returns:
        Sequence: Sequence of results.
    """
    try:
        pickle.dumps(function)
        helper_instance = function
    except pickle.PickleError:
        import dill

        function = dill.dumps(function)
        helper_instance = functools.partial(_helper, function)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as p:
        if progress_bar:
            return p.map(helper_instance, tqdm(iterable, total=len(iterable)))
        else:
            return p.map(helper_instance, iterable)


def _helper(function, iterable, *args, **kwargs):
    """
    Helper function to unpickle method in parallel execution.

    Args:
        function (bytes): Pickled method.
        iterable: Input argument.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Any: Result of applying the method.
    """
    import dill

    function = dill.loads(function)
    return function(iterable, *args, **kwargs)


def _pmap_joblib(function: Callable, iterable: Sequence, n_jobs: int, progress_bar: bool):
    """
    Internal function for parallel mapping using Joblib.

    Args:
        function (Callable): Method to apply.
        iterable (Sequence): Sequence of arguments.
        n_jobs (int): Number of parallel jobs.
        progress_bar (bool): Show progress bar.

    Returns:
        Sequence: Sequence of results.
    """
    from joblib import Parallel, delayed

    if progress_bar:
        return Parallel(n_jobs)(delayed(function)(args) for args in tqdm(iterable))
    else:
        return Parallel(n_jobs)(delayed(function)(args) for args in iterable)
