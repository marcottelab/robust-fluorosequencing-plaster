import ctypes as c
import pathlib
import sysconfig
from contextlib import contextmanager, redirect_stdout
from io import StringIO

import numpy as np
from plumbum import local

from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import CException, F64Arr
from plaster.tools.schema import check
from plaster.tools.utils import utils
from plaster.tools.zap import zap


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class RadiometryContext(c_common_tools.FixupStructure):
    # Operates on a single field stack so that we don't have
    # to realize all fields in to memory simultaneously

    # fmt: off
    _fixup_fields = [
        ("cy_ims", F64Arr),  # Sub-pixel aligned  (n_cycles, height, width)
        ("locs", F64Arr),  # Sub-pixel centered  (n_peaks, 2), where 2 is: (y, x)
        ("reg_psf_samples", F64Arr),  # Reg_psf samples (n_divs, n_divs, 3) (y, x, (sig_x, sig_y, rho))

        # Parameters
        ("ch_i", "Index"),
        ("n_cycles", "Size"),
        ("n_peaks", "Size"),
        ("n_divs", "Float64"),
        ("peak_mea", "Size"),
        ("height", "Float64"),
        ("width", "Float64"),
        ("raw_height", "Float64"),
        ("raw_width", "Float64"),

        # Outputs
        ("out_radiometry", F64Arr),  # (n_peaks, n_cycles, 4), where 4 is: (signal, noise, bg_med, bg_std)
    ]
    # fmt: on


RadiometryContext.struct_fixup()

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
        (MODULE_DIR / "radiometry_ext").with_suffix(
            sysconfig.get_config_var("EXT_SUFFIX")
        )
    )

    lib.context_init.argtypes = [
        c.POINTER(RadiometryContext),
    ]
    lib.context_init.restype = c.c_char_p

    lib.context_free.argtypes = [
        c.POINTER(RadiometryContext),
    ]

    lib.radiometry_field_stack_peak_batch.argtypes = [
        c.POINTER(RadiometryContext),  # RadiometryContext context
        c_common_tools.typedef_to_ctype("Index"),  # Index peak_start_i,
        c_common_tools.typedef_to_ctype("Index"),  # Index peak_stop_i,
    ]
    lib.radiometry_field_stack_peak_batch.restype = c.c_char_p

    _lib = lib
    return lib


@contextmanager
def context(cy_ims, locs, reg_psf_samples, peak_mea, ch_i):
    """
    with radiometry.context(...) as ctx:
        zap.work_orders(do_radiometry, ...)

    """
    lib = load_lib()

    check.array_t(cy_ims, ndim=3, dtype=np.float64)
    n_cycles, height, width = cy_ims.shape

    check.array_t(locs, ndim=2, dtype=np.float64)
    check.affirm(locs.shape[1] == 2)
    n_peaks = locs.shape[0]

    check.array_t(reg_psf_samples, ndim=3)
    n_divs, n_divs_w, n_params = reg_psf_samples.shape
    assert n_divs == n_divs_w
    assert n_params == 3

    out_radiometry = np.zeros((n_peaks, n_cycles, 4), dtype=np.float64)

    ctx = RadiometryContext(
        cy_ims=F64Arr.from_ndarray(cy_ims),
        locs=F64Arr.from_ndarray(locs),
        _locs=locs,
        ch_i=ch_i,
        n_cycles=n_cycles,
        n_peaks=n_peaks,
        n_divs=n_divs,
        peak_mea=peak_mea,
        height=height,
        width=width,
        reg_psf_samples=F64Arr.from_ndarray(reg_psf_samples),
        out_radiometry=F64Arr.from_ndarray(out_radiometry),
        _out_radiometry=out_radiometry,
    )

    error = lib.context_init(ctx)
    if error is not None:
        raise CException(error)

    try:
        yield ctx
    finally:
        lib.context_free(ctx)


def _do_radiometry_field_stack_peak_batch(
    ctx: RadiometryContext, peak_start_i: int, peak_stop_i: int
):
    """
    Worker for radiometry_field_stack() zap
    """
    lib = load_lib()

    error = lib.radiometry_field_stack_peak_batch(ctx, peak_start_i, peak_stop_i)
    if error is not None:
        raise CException(error)


def radiometry_cy_ims(cy_ims, locs, reg_psf_samples, peak_mea, ch_i):
    """
    Compute radiometry on the stack of cycle images for one field on one channel

    Returns:
        output_radmat: ndarray(n_peaks, n_cycles, (sig, noi, bg_med, bg_std))
    """
    with context(
        cy_ims=cy_ims,
        locs=locs,
        reg_psf_samples=reg_psf_samples,
        peak_mea=peak_mea,
        ch_i=ch_i,
    ) as ctx:
        check.array_t(locs, ndim=2, dtype=np.float64)
        n_peaks = locs.shape[0]
        if n_peaks > 0:
            batches = zap.make_batch_slices(n_rows=locs.shape[0], _batch_size=100)
            with zap.Context(trap_exceptions=False, mode="thread"):
                zap.work_orders(
                    [
                        dict(
                            fn=_do_radiometry_field_stack_peak_batch,
                            ctx=ctx,
                            peak_start_i=batch[0],
                            peak_stop_i=batch[1],
                        )
                        for batch in batches
                    ]
                )

        return ctx._out_radiometry
