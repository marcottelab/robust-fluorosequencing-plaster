import itertools
from typing import Callable

import numpy as np

from plaster.tools.schema import check


class FancyIndexer:
    """
    Used to index multi-dimensional objects

    TODO: Add negtative offsets
    """

    def __init__(self, lengths, lookup_fn: Callable, context=None):
        """
        lengths: a list of lengths for each dimension
        """
        self.n_dims = len(lengths)
        self.lookup_fn = lookup_fn
        check.list_or_tuple_t(lengths, int)
        self.lengths = np.array(lengths)
        self.context = context

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            index = (index,)

        assert isinstance(index, tuple)
        iters = [None] * self.n_dims
        singletons = [False] * self.n_dims
        iter_lens = [0] * self.n_dims

        for i, idx in enumerate(index):
            if isinstance(idx, (list, tuple)):
                iters[i] = list(idx)
            elif isinstance(idx, slice):
                iters[i] = list(range(*idx.indices(self.lengths[i])))
            elif isinstance(idx, np.ndarray):
                iters[i] = idx.tolist()
            elif isinstance(idx, int) or np.issubdtype(idx, np.integer):
                if idx < 0 or idx >= self.lengths[i]:
                    raise IndexError(
                        f"Tried to index {index} while lengths are {self.lengths}"
                    )
                singletons[i] = True
                iters[i] = [idx]
            else:
                raise TypeError("Unknown iterator type")

        # creating indexes
        for i in range(self.n_dims):
            if iters[i] is None:
                iters[i] = tuple(list(range(self.lengths[i])))
            iter_lens[i] = len(iters[i])

        ret_arrays = []
        dim = None

        # iterate over all relevant combinations
        for iz in itertools.product(*iters):
            if self.context is not None:
                ret_array = self.lookup_fn(*iz, context=self.context)
            else:
                ret_array = self.lookup_fn(*iz)

            assert isinstance(ret_array, np.ndarray)
            _dim = ret_array.shape
            assert dim is None or _dim == dim
            dim = _dim
            ret_arrays += [ret_array]

        arr = np.array(ret_arrays).reshape((*tuple(iter_lens), *dim))

        new_shape = [
            n for i, n in enumerate(arr.shape[: self.n_dims]) if not singletons[i]
        ]
        new_shape += [i for i in arr.shape[self.n_dims :]]

        return arr.reshape(tuple(new_shape))
