import numpy as np

from plaster.tools.c_common.c_common_tools import DytPepType

DYT_COL = 0
PEP_COL = 1
CNT_COL = 2


def validate(dytpeps: np.ndarray):
    assert dytpeps.shape[1] == 3
    assert dytpeps.dtype == DytPepType
    assert dytpeps.flags["C_CONTIGUOUS"]


def n_dyts(dytpeps: np.ndarray):
    validate(dytpeps)
    return int(np.max(dytpeps[:, DYT_COL]) + 1)


def n_peps(dytpeps: np.ndarray):
    validate(dytpeps)
    # return int(np.max(dytpeps[:, PEP_COL]) + 1)
    #
    # I don't think this is correct.  This will return the largest
    # value pep_i in dytpeps.  dytpeps only contains pep_i that
    # produce non-dark rows, so there are a number of "holes" in
    # the sequence of pep_i.  Instead, we really want the number of
    # unique pep_i in the dytpeps structure, and we don't have to
    # add a +1 because this will include the 0/null-row.
    #
    return len(np.unique(dytpeps[:, PEP_COL]))


def n_samples(dytpeps: np.ndarray):
    validate(dytpeps)
    return int(np.sum(dytpeps[:, CNT_COL]))


def argsort_dytpeps(dytpeps: np.ndarray):
    # SORT dytpeps by peptide (col [1]) first then by count (col [2])
    # Note that np.lexsort puts the primary sort key LAST in the argument
    validate(dytpeps)
    return np.lexsort((-dytpeps[:, CNT_COL], dytpeps[:, PEP_COL]))


def sort_dytpeps(dytpeps: np.ndarray):
    return dytpeps[argsort_dytpeps(dytpeps)]
