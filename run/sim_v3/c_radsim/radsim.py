import ctypes as c
import pathlib
import sysconfig
from contextlib import contextmanager
from typing import Optional

import numpy as np

from plaster.run.priors import Priors
from plaster.run.sim_v3 import dyt_helpers
from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import (
    RNG,
    CException,
    DytPepType,
    DytType,
    F32Arr,
    PriorParameterType,
    RadType,
    RowKType,
    SizeType,
    Tab,
    U8Arr,
    U32Arr,
)
from plaster.tools.schema import check
from plaster.tools.zap import zap

_lib = None

MODULE_DIR = pathlib.Path(__file__).parent


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib = c.CDLL(
        (MODULE_DIR / "radsim_ext").with_suffix(sysconfig.get_config_var("EXT_SUFFIX"))
    )

    # C_COMMON
    # ------------------------------------------------------
    lib.sanity_check.argtypes = []
    lib.sanity_check.restype = c.c_int

    # radsim
    # ------------------------------------------------------
    lib.radsim_batch.argtypes = [
        c.POINTER(RadSimContext),
        c.POINTER(RNG),
        c_common_tools.typedef_to_ctype("Index"),  # Index start_dytpep_row_i,
        c_common_tools.typedef_to_ctype("Index"),  # Index stop_dytpep_row_i,
    ]
    lib.radsim_batch.restype = c.c_char_p

    lib.radsim_context_init.argtypes = [
        c.POINTER(RadSimContext),
    ]
    lib.radsim_context_init.restype = c.c_char_p

    lib.radsim_context_free.argtypes = [
        c.POINTER(RadSimContext),
    ]
    lib.radsim_context_free.restype = c.c_char_p

    # Sample Peps
    # ------------------------------------------------------
    lib.sample_pep.argtypes = [
        c.POINTER(SamplePepsContext),
        c.POINTER(RNG),
        c_common_tools.typedef_to_ctype("Index"),  # Index start_pep_i,
        c_common_tools.typedef_to_ctype("Index"),  # Index stop_pep_i,
    ]
    lib.sample_pep.restype = c.c_char_p

    lib.sample_peps_context_init.argtypes = [
        c.POINTER(SamplePepsContext),
    ]
    lib.sample_peps_context_init.restype = c.c_char_p

    lib.sample_peps_context_free.argtypes = [
        c.POINTER(SamplePepsContext),
    ]
    lib.sample_peps_context_free.restype = c.c_char_p

    _lib = lib

    assert lib.sanity_check() == 0

    return lib


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class SamplePepsContext(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("dytpeps", U32Arr),
        ("n_samples_per_pep", "Index"),
        ("out_dytpeps", U32Arr),
        ("pep_i_to_dytpep_i", U32Arr),
        ("pep_i_to_n_dytpeps", U32Arr),
        ("n_peps", "Size"),
    ]


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class RadSimContext(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("n_cycles", "Size"),
        ("n_channels", "Size"),
        ("dytmat", U8Arr),
        ("dytpeps", U32Arr),
        ("dytpep_i_to_out_i", U32Arr),
        ("use_lognormal_model", "Bool"),
        ("row_k_sigma", "PriorParameterType"),
        ("ch_illum_priors", Tab, "ChIllumPriors"),
        ("out_radmat", F32Arr),
        ("out_row_ks", F32Arr),
        ("out_dyt_iz", U32Arr),
        ("out_pep_iz", U32Arr),
    ]


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class ChIllumPriors(c_common_tools.FixupStructure):
    # Illumination priors for a single channel
    _fixup_fields = [
        ("gain_mu", "Float64"),
        ("gain_sigma", "Float64"),
        ("bg_mu", "Float64"),
        ("bg_sigma", "Float64"),
        ("row_k_sigma", "Float64"),
    ]


RadSimContext.struct_fixup()
ChIllumPriors.struct_fixup()
SamplePepsContext.struct_fixup()


# pep_sample
# ---------------------------------------------------------------------------------


@contextmanager
def pep_sample_context(
    dytpeps: np.ndarray,
    n_samples_per_pep: int,
    n_peps: int,
):
    """
    dytpeps is a table of 3 columns
        The columns are: dyt_i, pep_i, cnt
        So eassential each row is saying "this dyttrack was created by this peptide n times."
        See dyt_helpers.py

    This is a context wrapper to the C module that will sample these
    rows so that we have a number of samples per peptide which
    is actually performed by the _do_pep_sample_batch in parallel.
    """

    lib = load_lib()

    check.array_t(dytpeps, shape=(None, 3), dtype=DytPepType, c_contiguous=True)
    check.t(n_samples_per_pep, int)
    assert DytPepType == np.uint32  # Because of the use of U32Arr below

    # COPY the dytpeps to the output and zero out the counts; these will be resampled
    out_dytpeps = dytpeps.copy()
    out_dytpeps[:, 2] = 0  # 2 is the column of the counts

    # pep_i_to_dytpep_i is allocated with +1 so that a simple length calculation for
    # each peptide block can be done be taking (pep_i_to_dytpep_i[i+1] - pep_i_to_dytpep_i[i])
    pep_i_to_dytpep_i = np.zeros((n_peps + 1,), dtype=DytPepType)
    check.array_t(pep_i_to_dytpep_i, c_contiguous=True)

    pep_i_to_n_dytpeps = np.zeros((n_peps + 1,), dtype=DytPepType)
    check.array_t(pep_i_to_n_dytpeps, c_contiguous=True)

    check.array_t(out_dytpeps, shape=(None, 3), dtype=DytPepType, c_contiguous=True)

    ctx = SamplePepsContext(
        dytpeps=U32Arr.from_ndarray(dytpeps),
        n_samples_per_pep=n_samples_per_pep,
        out_dytpeps=U32Arr.from_ndarray(out_dytpeps),
        _out_dytpeps=out_dytpeps,
        # Set by context_init
        pep_i_to_dytpep_i=U32Arr.from_ndarray(pep_i_to_dytpep_i),
        pep_i_to_n_dytpeps=U32Arr.from_ndarray(pep_i_to_n_dytpeps),
        _pep_i_to_dytpep_i=pep_i_to_dytpep_i,
        _pep_i_to_n_dytpeps=pep_i_to_n_dytpeps,
        n_peps=n_peps,
    )

    error = lib.sample_peps_context_init(ctx)

    if error is not None:
        raise CException(error)

    try:
        yield ctx
    finally:
        lib.sample_peps_context_free(ctx)


def _do_pep_sample_batch(
    ctx: SamplePepsContext,
    start_pep_i: int,
    stop_pep_i: int,
    *,
    seed: Optional[int] = None,
):
    """
    The worker of the zap to run a batch of peptides
    """
    check.t(ctx, SamplePepsContext)
    check.t(start_pep_i, int)
    check.t(stop_pep_i, int)
    assert 0 <= start_pep_i < stop_pep_i

    # TODO: Remove when Flyte fixes Optional[int] behavior
    # Flyte internally converts Optional[int] types to float
    # so the RNG seed has to be returned to int
    if seed is not None:
        seed = int(seed)
    lib = load_lib()

    rng = RNG(seed)

    error = lib.sample_pep(ctx, rng, start_pep_i, stop_pep_i)
    if error is not None:
        raise CException(error)


def c_pep_sample(
    dytpeps: np.ndarray,
    n_samples_per_pep: int,
    *,
    seed: Optional[int] = None,
    progress=None,
):
    """
    Generate radmat from (dyts)
    """
    check.array_t(dytpeps, shape=(None, 3), dtype=DytPepType, c_contiguous=True)
    check.t(n_samples_per_pep, int)
    assert DytPepType == np.uint32  # Because of the use of U32Arr below

    # TODO: Remove when Flyte fixes Optional[int] behavior
    # Flyte internally converts Optional[int] types to float
    # so the RNG seed has to be returned to int
    if seed is not None:
        seed = int(seed)
    n_peps = dyt_helpers.n_peps(dytpeps)

    with pep_sample_context(dytpeps, n_samples_per_pep, n_peps) as ctx:
        batches = zap.make_batch_slices(n_rows=n_peps, _batch_updates=1)
        with zap.Context(trap_exceptions=False, mode="thread", progress=progress):
            zap.work_orders(
                [
                    dict(
                        fn=_do_pep_sample_batch,
                        ctx=ctx,
                        start_pep_i=batch[0],
                        stop_pep_i=batch[1],
                        seed=seed,
                    )
                    for batch in batches
                ]
            )

        return ctx._out_dytpeps


# radsim
# ---------------------------------------------------------------------------------


@contextmanager
def radsim_context(
    dytmat: np.ndarray,  # (n_dyts, n_channels, n_cycles)
    dytpeps: np.ndarray,
    priors: Priors,
    use_lognormal_model: bool,
    out_radmat: np.ndarray,
    out_row_ks: np.ndarray,
    out_dyt_iz: np.ndarray,
    out_pep_iz: np.ndarray,
):
    """
    The priors are passed all the way down to this level so that we can
    eventually pass the prior parameters into the C code for full
    distribution re-sampling as opposed to point-estimates.
    """
    lib = load_lib()

    check.array_t(dytmat, ndim=3, dtype=DytType, c_contiguous=True)
    dyt_helpers.validate(dytpeps)
    assert SizeType == np.uint64  # Because of the use of U64Arr below
    assert RadType == np.float32  # Because of the use of F32Arr below
    assert RowKType == np.float32  # Because of the use of F32Arr below
    assert DytPepType == np.uint32  # Because of the use of U32Arr below

    n_dyts, n_channels, n_cycles = dytmat.shape

    n_samples_total = dyt_helpers.n_samples(dytpeps)
    check.array_t(out_radmat, shape=(None, n_channels, n_cycles), dtype=RadType)
    assert out_radmat.shape[0] >= n_samples_total

    check.array_t(out_row_ks, dtype=RowKType)
    assert out_row_ks.shape[0] >= n_samples_total

    check.array_t(out_dyt_iz, dtype=DytPepType)
    assert out_dyt_iz.shape[0] >= n_samples_total

    check.array_t(out_pep_iz, dtype=DytPepType)
    assert out_pep_iz.shape[0] >= n_samples_total

    # See note above about these being point-estaimtes and will
    # eventually be full prior distributions that will be sampled from C.
    ch_illum_priors_arr = np.array(
        [
            (
                priors.get_mle(f"gain_mu.ch_{ch_i}"),
                priors.get_mle(f"gain_sigma.ch_{ch_i}"),
                priors.get_mle(f"bg_mu.ch_{ch_i}"),
                priors.get_mle(f"bg_sigma.ch_{ch_i}"),
            )
            for ch_i in range(n_channels)
        ],
        dtype=PriorParameterType,
    )

    dytpep_i_to_out_i = np.cumsum(dytpeps[:, 2], dtype=DytPepType)

    ctx = RadSimContext(
        n_cycles=n_cycles,
        n_channels=n_channels,
        dytmat=U8Arr.from_ndarray(dytmat),
        dytpeps=U32Arr.from_ndarray(dytpeps),
        dytpep_i_to_out_i=U32Arr.from_ndarray(dytpep_i_to_out_i),
        use_lognormal_model=use_lognormal_model,
        row_k_sigma=priors.get_mle("row_k_sigma"),
        ch_illum_priors=Tab.from_ndarray(
            ch_illum_priors_arr, expected_dtype=PriorParameterType
        ),
        out_radmat=F32Arr.from_ndarray(out_radmat),
        out_row_ks=F32Arr.from_ndarray(out_row_ks),
        out_dyt_iz=U32Arr.from_ndarray(out_dyt_iz),
        out_pep_iz=U32Arr.from_ndarray(out_pep_iz),
        _out_radmat=out_radmat,
        _out_row_ks=out_row_ks,
        _out_dyt_iz=out_dyt_iz,
        _out_pep_iz=out_pep_iz,
    )

    error = lib.radsim_context_init(ctx)
    if error is not None:
        raise CException(error)

    try:
        yield ctx
    finally:
        lib.radsim_context_free(ctx)


def _do_radsim_batch(
    ctx: RadSimContext,
    start_dytpep_row_i: int,
    stop_dytpep_row_i: int,
    *,
    seed: Optional[int] = None,
):
    """
    The worker of the zap to run a batch of peptides
    """
    check.t(ctx, RadSimContext)
    check.t(start_dytpep_row_i, int)
    check.t(stop_dytpep_row_i, int)
    assert 0 <= start_dytpep_row_i < stop_dytpep_row_i

    # TODO: Remove when Flyte fixes Optional[int] behavior
    # Flyte internally converts Optional[int] types to float
    # so the RNG seed has to be returned to int
    if seed is not None:
        seed = int(seed)
    lib = load_lib()

    rng = RNG(seed)
    error = lib.radsim_batch(ctx, rng, start_dytpep_row_i, stop_dytpep_row_i)
    if error is not None:
        raise CException(error)


def c_radsim(
    dytmat: np.ndarray,
    dytpeps: np.ndarray,
    priors: Priors,
    use_lognormal_model: bool,
    out_radmat: np.ndarray,
    out_row_ks: np.ndarray,
    out_dyt_iz: np.ndarray,
    out_pep_iz: np.ndarray,
    *,
    seed: Optional[int] = None,
    progress=None,
):
    """
    Generate radmat from (dyts).

    Multi-threads so the output buffers have to be available to all threads.

    This means that the dytpeps can not have any that reference the nul-dyt
    and that means that we have to have tracked that nul-dyt in the dyt sampler.

    Skips dyt's of zero so the number of returned rows may be less than
    in the dytpeps counts.
    """
    check.array_t(dytmat, ndim=3, dtype=DytType, c_contiguous=True)
    check.array_t(dytpeps, shape=(None, 3), dtype=DytPepType, c_contiguous=True)
    check.t(priors, Priors)

    # TODO: Remove when Flyte fixes Optional[int] behavior
    # Flyte internally converts Optional[int] types to float
    # so the RNG seed has to be returned to int
    if seed is not None:
        seed = int(seed)

    with radsim_context(
        dytmat,
        dytpeps,
        priors,
        use_lognormal_model,
        out_radmat=out_radmat,
        out_row_ks=out_row_ks,
        out_dyt_iz=out_dyt_iz,
        out_pep_iz=out_pep_iz,
    ) as ctx:
        n_dytpeps = dytpeps.shape[0]

        # _limit_slice to skip the nul-record
        batches = zap.make_batch_slices(
            n_rows=n_dytpeps, _limit_slice=slice(1, n_dytpeps, 1), _batch_updates=1
        )

        # See above similar code about why this is temporaily in slow "thread" mode
        with zap.Context(trap_exceptions=False, mode="thread", progress=progress):
            zap.work_orders(
                [
                    dict(
                        fn=_do_radsim_batch,
                        ctx=ctx,
                        start_dytpep_row_i=batch[0],
                        stop_dytpep_row_i=batch[1],
                        seed=seed,
                    )
                    for batch in batches
                ]
            )
