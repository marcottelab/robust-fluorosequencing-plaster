import ctypes as c
import pathlib
import sysconfig
from contextlib import contextmanager, redirect_stdout
from io import StringIO

import numpy as np
from plumbum import local

from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import (
    CException,
    F64Arr,
    Tab,
    U16Arr,
    U64Arr,
)
from plaster.tools.schema import check
from plaster.tools.utils import utils


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class PeakFindContext(c_common_tools.FixupStructure):
    # Operates on a single field stack so that we don't have
    # to realize all fields in to memory simultaneously

    # fmt: off
    _fixup_fields = [
        ("ch_ims", U16Arr),  # Stack of all channels for this field, cycle
        ("n_channels", "Size"),
        ("max_n_locs", "Size"),
        ("im_mea", "Size"),

        ("sub_locs_tab", Tab, "SubLoc"),

        # Outputs
        ("out_n_locs", U64Arr),  # Size == 1
        ("out_locs", F64Arr),  # (out_n_peaks, 2), where 2 is: (y, x)
        ("out_warning_n_locs_overflow", U64Arr),  # Size == 1
        ("out_debug", U64Arr),
    ]
    # fmt: on


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class SubLoc(c_common_tools.FixupStructure):
    # fmt: off
    _fixup_fields = [
        ("loc_ids", "ChannelLocId"),
        ("ambiguous", "Size")
    ]
    # fmt: on


PeakFindContext.struct_fixup()
SubLoc.struct_fixup()

_lib = None

MODULE_DIR = pathlib.Path(__file__).parent


class Radmat:
    # The columns of a radmat
    sig = 0
    noi = 1
    bg_med = 2
    bg_std = 3


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib = c.CDLL(
        (MODULE_DIR / "peak_find_ext").with_suffix(
            sysconfig.get_config_var("EXT_SUFFIX")
        )
    )

    lib.context_init.argtypes = [
        c.POINTER(PeakFindContext),
    ]
    lib.context_init.restype = c.c_char_p

    lib.context_free.argtypes = [
        c.POINTER(PeakFindContext),
    ]

    lib.peak_find.argtypes = [
        c.POINTER(PeakFindContext),
        c_common_tools.typedef_to_ctype("Index"),  # Index reg_i,
    ]
    lib.peak_find.restype = c.c_char_p

    _lib = lib
    return lib


@contextmanager
def context(ch_ims):
    lib = load_lib()

    check.array_t(ch_ims, ndim=3, dtype=np.uint16, c_contiguous=True)
    n_channels, height, width = ch_ims.shape
    assert width == height and utils.is_power_of_2(width)
    im_mea = width

    max_n_locs = 8196
    # out_locs = np.zeros((max_n_locs, 2), dtype=np.float64)
    # out_n_locs = np.zeros((1,), dtype=np.uint64)
    out_debug = np.zeros((im_mea, im_mea), dtype=np.uint64)
    out_warning_n_locs_overflow = np.zeros((1,), dtype=np.uint64)

    sub_locs_tab = Tab.allocate(SubLoc, max_n_locs)

    ctx = PeakFindContext(
        ch_ims=U16Arr.from_ndarray(ch_ims),
        n_channels=n_channels,
        im_mea=im_mea,
        max_n_locs=max_n_locs,
        out_debug=U64Arr.from_ndarray(out_debug),
        _out_debug=out_debug,
        out_warning_n_locs_overflow=U64Arr.from_ndarray(out_warning_n_locs_overflow),
        _out_warning_n_locs_overflow=out_warning_n_locs_overflow,
        sub_locs_tab=sub_locs_tab,
        _sub_locs_tab=sub_locs_tab,
    )

    error = lib.context_init(ctx)
    if error is not None:
        raise CException(error)

    try:
        yield ctx
    finally:
        lib.context_free(ctx)


def peak_find_on_peak_label_ims(ch_ims):
    """
    Given a set of channel images where the pixels are 0 if no peak
    found and otherwise the label ("peak id") for the respective channels.
    This function reconciles the multi-channel peak_labels into one
    single list of "canonical peaks" and returns a mapping "locs_ids"
    which allows us to map from the canonical peak id space back into the
    original per-channel peak ids.

    Returns:
        loc_ids: ndarray[n_canonical_peaks, n_channels]{int64}
        ambiguous: ndarray[n_canonical_peaks]{bool(int64)}
    """
    lib = load_lib()

    with context(ch_ims) as ctx:
        reg_i = 0
        error = lib.peak_find(ctx, reg_i)
        if error is not None:
            raise CException(error)

        n_canonical_rows = ctx.sub_locs_tab.n_rows
        n_channels = ch_ims.shape[0]

        # loc_ids (n_locs, n_channels)

        # Each channel has its own list of peaks and the loc_ids
        # gives the mapping from the canonical list of peaks
        # (shared by all channels) back to the local per-channel id.
        #
        # Example: Suppose there's 3 channels and peak 0 in the canonical list
        # came from peak 10 in channel 0 and also peak 22 in channel 1 but
        # channel 2 did not contribute. We'd have:
        #  [
        #    [10, 22, 0]
        #  ]
        #
        # Note that this function expects that the background is labelled zero
        # and it returns the canonical list WIHTOUT the zero reservation so
        # peak_i == 0 is a VALID peak in the returned canonical list.

        # Two different peak_ids in the SAME channel with overlapping pixels
        # is called "ambiguous" because there was not a single clear answer
        # to the question "How many peaks are at this location"

        # TODO consdier renaming loc_ids to confirm to array naming convention something
        # something like "canonical_peak_i_to_per_ch_i"

        # TODO: Both of the following python loops can likely be a single numpy operation

        loc_ids = np.array(
            [
                [ctx._sub_locs_tab._arr[i].loc_ids[ch_i] for ch_i in range(n_channels)]
                for i in range(n_canonical_rows)
            ]
        )

        # Ambiguous (n_locs)
        ambiguous = np.array(
            [ctx._sub_locs_tab._arr[i].ambiguous for i in range(n_canonical_rows)]
        )

        return loc_ids, ambiguous


def peak_find_get_ctx(ch_ims):
    lib = load_lib()

    with context(ch_ims) as ctx:
        reg_i = 0
        error = lib.peak_find(ctx, reg_i)
        if error is not None:
            raise CException(error)

        n_rows = ctx.sub_locs_tab.n_rows
        n_channels = ch_ims.shape[0]

        return (
            ctx,
            # Locations (n_locs, n_channels)
            np.array(
                [
                    [
                        ctx._sub_locs_tab._arr[i].loc_ids[ch_i]
                        for ch_i in range(n_channels)
                    ]
                    for i in range(n_rows)
                ]
            ),
            # Ambiguous (n_locs)
            np.array([ctx._sub_locs_tab._arr[i].ambiguous for i in range(n_rows)]),
        )
