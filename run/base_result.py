import logging
from contextlib import contextmanager
from functools import wraps

import numpy as np
from munch import Munch
from plumbum import local

from plaster.tools.schema import check
from plaster.tools.utils import utils

log = logging.getLogger(__name__)


_enable_disk_memoize = False


@contextmanager
def enable_disk_memoize():
    global _enable_disk_memoize
    try:
        _enable_disk_memoize = True
        yield
    finally:
        _enable_disk_memoize = False


def enable_disk_memoize_from_notebook():
    global _enable_disk_memoize
    _enable_disk_memoize = True


def disk_memoize():
    """
    Only used for memoizing methods of BaseResult classes.

    Ignores args that are instances of BaseResult so that we
    will get coherency between restarts of a jupyter kernel
    """

    def _wraps(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert isinstance(args[0], BaseResult)
            if _enable_disk_memoize:
                keep_args = tuple([a for a in args if not isinstance(a, BaseResult)])
                h = hash(keep_args + tuple(sorted(kwargs.items()))) ** 2
                path = args[0]._folder / f"_cache_{func.__name__}_{h:X}.pkl"
                if path.exists():
                    rv = utils.pickle_load(path)
                else:
                    rv = func(*args, **kwargs)
                    utils.pickle_save(path, rv)
                return rv
            else:
                return func(*args, **kwargs)

        return wrapper

    return _wraps


class ArrayResult:
    """
    A common result type is a large array that is larger than memory.
    Instances of this class allow those to be written to files and then
    streamed as needed.
    """

    def __init__(self, filename, dtype, shape, mode="r", offset=0, order="C"):
        self._fp = None
        self.filename = filename
        self.dtype = dtype
        self.shape = shape
        self.mode = mode
        self.offset = offset
        self.order = order
        self.arr()

    def arr(self):
        if self._fp is None:
            self._fp = np.memmap(
                self.filename,
                dtype=self.dtype,
                shape=self.shape,
                mode=self.mode,
                offset=self.offset,
                order=self.order,
            )
        return self._fp

    def flush(self):
        if self._fp is not None:
            self._fp.flush()

    def reshape(self, new_shape):
        """Will truncate the size of the memory mapped file if getting smaller"""
        n_elems = np.product(new_shape)
        if n_elems == 0:
            # self.arr().base.resize() does not allow zero size
            # so we make it 1 buyt note that self.shape below
            # will have the true zero size so this should just
            # end up with some wasted file space.
            n_elems = 1
        self.arr().base.resize(n_elems * self.arr().itemsize)
        self.flush()
        del self._fp
        self._fp = None
        self.shape = new_shape
        if self.mode == "w+":
            # In the case that this was originally opened to overwrite
            # now that it has been truncated and will re-open on the
            # next call to arr() we must ensure that it doesn't overwrite
            # by changing the mode to "r+"
            self.mode = "r+"

    def __getstate__(self):
        self.flush()
        return (
            self.filename,
            self.dtype,
            self.shape,
            self.mode,
            self.offset,
            self.order,
        )

    def __setstate__(self, state):
        self._fp = None
        (
            self.filename,
            self.dtype,
            self.shape,
            self.mode,
            self.offset,
            self.order,
        ) = state
        # When re-loading, kick it in to read-only
        self.mode = "r"

    def __getitem__(self, index):
        fp = self.arr()
        return fp[index]

    def __setitem__(self, index, val):
        fp = self.arr()
        fp[index] = val


def trim_array(ndarray_or_arrayresult, new_size):
    """
    Remove tail rows from either a naked ndarray or an ArrayResults wrapper
    """
    check.t(ndarray_or_arrayresult, (np.ndarray, ArrayResult))

    if isinstance(ndarray_or_arrayresult, ArrayResult):
        shape = ndarray_or_arrayresult.arr().shape
        ndarray_or_arrayresult.reshape((new_size, *shape[1:]))
    else:
        ndarray_or_arrayresult = ndarray_or_arrayresult[0:new_size]

    return ndarray_or_arrayresult
