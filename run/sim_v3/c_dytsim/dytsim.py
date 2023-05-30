import ctypes as c
import pathlib
import sysconfig
from contextlib import contextmanager
from typing import Optional

import numpy as np
import structlog

from plaster.run.sim_v3 import dyt_helpers
from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import (
    RNG,
    CBBType,
    CException,
    CycleKindType,
    DytPepType,
    DytType,
    Hash,
    IndexType,
    PCBType,
    RecallType,
    SizeType,
    Tab,
    xRNG,
)
from plaster.tools.zap import zap

_lib = None

MODULE_DIR = pathlib.Path(__file__).parent

logger = structlog.get_logger()


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib = c.CDLL(
        (MODULE_DIR / "dytsim_ext").with_suffix(sysconfig.get_config_var("EXT_SUFFIX"))
    )

    # C_COMMON
    lib.sanity_check.argtypes = []
    lib.sanity_check.restype = c.c_int

    # DYTSIM
    lib.dyt_n_bytes.argtypes = [c.c_uint64, c.c_uint64]
    lib.dyt_n_bytes.restype = c.c_uint64

    lib.prob_to_p_i.argtypes = [c.c_double]
    lib.prob_to_p_i.restype = c.c_uint64

    lib.dytsim_batch.argtypes = [
        c.POINTER(DytSimContext),
        c.POINTER(RNG),
        c_common_tools.typedef_to_ctype("Index"),  # Index start_pep_i,
        c_common_tools.typedef_to_ctype("Index"),  # Index stop_pep_i,
    ]
    lib.dytsim_batch.restype = c.c_char_p

    lib.context_init.argtypes = [
        c.POINTER(DytSimContext),
    ]
    lib.context_init.restype = c.c_char_p

    lib.context_free.argtypes = [
        c.POINTER(DytSimContext),
    ]
    lib.context_free.restype = c.c_char_p

    lib.copy_results.argtypes = [
        c.POINTER(DytSimContext),
        c.POINTER(c_common_tools.typedef_to_ctype("DytType")),
        c.POINTER(c_common_tools.typedef_to_ctype("DytPepType")),
    ]
    lib.copy_results.restype = c.c_char_p

    lib.xrng_uint64.argtypes = [c.POINTER(xRNG)]
    lib.xrng_uint64.restype = c.c_uint64

    _lib = lib

    assert lib.sanity_check() == 0

    return lib


# fmt: off
# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class DytSimContext(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("n_peps", "Size"),
        ("n_cycles", "Size"),
        ("n_samples", "Size"),
        ("n_channels", "Size"),
        ("pi_bleach", "PIType"),
        ("pi_cyclic_block", "PIType"),
        ("pi_initial_block", "PIType"),
        ("pi_detach", "PIType"),
        ("pi_edman_success", "PIType"),
        ("pi_label_fail", "PIType"),
        ("allow_edman_cterm", "Uint64"),
        ("cycles", Tab, "CycleKindType"),
        ("pcbs", Tab, "Uint8"),
        ("cbbs", Tab, "Uint8"),
        ("pep_recalls", Tab, "RecallType"),
        ("count_only", "Size"),
        ("n_max_dyts", "Size"),
        ("n_max_dytpeps", "Size"),
        ("n_dyt_row_bytes", "Size"),
        ("n_max_dyt_hash_recs", "Size"),
        ("n_max_dytpep_hash_recs", "Size"),
        ("pep_i_to_pcb_i", Tab, "Index"),

        # Used for count_only...
        ("out_counts", Tab, "Size"),  # (n_dyts, n_dytpeps)

        # Allocated by C...
        ("_work_order_lock", "pthread_mutex_t *"),
        ("_tab_lock", "pthread_mutex_t *"),
        ("_dytrecs", Tab, "DytRec"),
        ("_dytpeps", Tab, "?"),
        ("_dyt_hash", Hash, "Uint8"),
        ("_dytpep_hash", Hash, "Uint8"),
    ]


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class PCB(c_common_tools.FixupStructure):
    # Are a kind of "flu" encoding. See def pcbs() in sim_v3_params
    _fixup_fields = [
        ("pep_i", "PCBType"),
        ("ch_i", "PCBType"),
        ("p_bright", "PCBType"),
        ("p_bleach", "PCBType"),
    ]


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class CBB(c_common_tools.FixupStructure):
    # Holds probabilities for each channel
    _fixup_fields = [
        ("ch_i", "PCBType"),
        ("p_bright", "PCBType"),
        ("p_bleach", "PCBType"),
    ]


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class Counts(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("n_new_dyts", "Size"),
        ("n_new_dytpeps", "Size"),
    ]


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class DytPepRec(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("dyt_i", "Index"),
        ("pep_i", "Index"),
        ("n_reads", "Size"),
    ]
# fmt: on


DytSimContext.struct_fixup()
PCB.struct_fixup()
Counts.struct_fixup()
DytPepRec.struct_fixup()


def max_counts(n_peps, n_labels, n_channels):
    """
    See https://docs.google.com/spreadsheets/d/1GIuox8Rm5H6V3HbazYC713w0grnPSHgEsDD7iH0PwS0/edit#gid=0

    Based on experiments using the count_only option
    I found that n_dyts and n_max_dytpeps grow linearly w/ n_peps

    After some fiddling I think the following

    So, for 5 channels, 15 cycles, 750_000 peptides:
      Dyts = (8 + 8 + 5 * 15) = 91 * 250 * 750_000 = 17_062_500_000 = 17GB
      DytPepRec = (8 + 8 + 8) = 24 * 450 * 750_000 = 8_100_000_000 = 8GB
      Total = 25 GB

    So, that's a lot, but that's an extreme case...
    I could bring it down in several ways:
    I could store all as 32-bit which would make it:
      Dyts = (4 + 4 + 5 * 15) = 91 * 250 * 750_000 = 15_562_500_000 = 15GB
      DytPepRec = (4 + 4 + 4) = 12 * 450 * 750_000 = 4_050_000_000 = 4GB
      Total = 19GB

    Or, I could stochastically remove low-count dyecounts
    which would be a sort of garbage collection operation
    which would probably better than half memory but at more compute time.

    For now, at channel counts I'm likely to run I don't think it will be a problem.

    10/28/2020 DHW changed these equations to include n_labels and n_channels. See third sheet in linked doc
    This is potentially slightly better, but there's still another factor besides n_peps, n_labels, and n_channels that needs to be used,
    though I'm not sure what it is yet.
    ZBS: I bet that another factor is the MEAN of the length of the peptides that are generated by
    that protease. You can get the length of a peptide in the "prep_result". There are a variety of dataframe requests and with a little
    panda-kung-fu you should be able to get the mean length
    3/29/22 DSW changed the exponent to 2 and multiplied by 10 to get BSA survey ntbc and aspn working
    """
    # It's possible that the max doesn't grow linearly across n_labels, though I'm not sure.
    # For some reason when n_labels == 1 things act really differently, not sure what that's about
    n_max_dyts = n_peps * n_channels * max(n_labels, 2) ** 2.0 * 50 * 10 + 100_000
    n_max_dytpeps = n_peps * n_channels * max(n_labels, 2) ** 2.0 * 90 * 10 + 100_000
    return n_max_dyts, n_max_dytpeps


@contextmanager
def context(
    pcbs: np.ndarray,  # pcb = (p)ep_i, (c)h_i, (b)right_prob == like a "flu" with brightness probability)
    cbbs: np.ndarray,
    n_samples: int,
    n_channels: int,
    n_labels: int,
    cycles: np.ndarray,
    p_cyclic_block: float,
    p_initial_block: float,
    p_detach: float,
    p_edman_fail: float,
    p_label_fail: float,
    allow_edman_cterm: bool,
):
    count_only = 0  # Set to 1 to use the counting mechanisms

    lib = load_lib()

    assert pcbs.dtype == PCBType
    assert pcbs.flags["C_CONTIGUOUS"]
    assert cycles.dtype == CycleKindType
    assert cycles.flags["C_CONTIGUOUS"]

    n_peps = int(np.max(pcbs[:, 0]) + 1)
    n_cycles = cycles.shape[0]
    n_dyt_row_bytes = lib.dyt_n_bytes(n_channels, n_cycles)

    if count_only == 1:
        n_max_dyts = 1
        n_max_dyt_hash_recs = 100_000_000
        n_max_dytpeps = 1
        n_max_dytpep_hash_recs = 100_000_000

    else:
        n_max_dyts, n_max_dytpeps = max_counts(n_peps, n_labels, n_channels)

        hash_factor = 1.5
        n_max_dyt_hash_recs = int(hash_factor * n_max_dyts)
        n_max_dytpep_hash_recs = int(hash_factor * n_max_dytpeps)

        dyt_gb = n_max_dyts * n_dyt_row_bytes / 1024**3
        dytpep_gb = n_max_dytpeps * c.sizeof(DytPepRec) / 1024**3
        if dyt_gb + dytpep_gb > 10:
            logger.warning(
                "sim buffers > 10GB",
                current=round(dyt_gb + dytpep_gb, 3),
                dyt_gb=dyt_gb,
                dytpep_gb=dytpep_gb,
                n_max_dyts=n_max_dyts,
                n_max_dytpeps=n_max_dytpeps,
            )

    pep_i_to_pcb_i = np.zeros(
        (n_peps + 1,), dtype=IndexType
    )  # + 1 so that we can do end of table lookup
    pep_recalls = np.zeros(n_peps, dtype=RecallType)
    out_counts = np.zeros((2,), dtype=SizeType)  # (n_dyts, n_dytpeps)

    ctx = DytSimContext(
        n_peps=n_peps,
        n_cycles=n_cycles,
        n_samples=n_samples,
        n_channels=n_channels,
        pi_cyclic_block=lib.prob_to_p_i(p_cyclic_block),
        pi_initial_block=lib.prob_to_p_i(p_initial_block),
        pi_detach=lib.prob_to_p_i(p_detach),
        pi_edman_success=lib.prob_to_p_i(1.0 - p_edman_fail),
        pi_label_fail=lib.prob_to_p_i(p_label_fail),
        allow_edman_cterm=allow_edman_cterm,
        pcbs=Tab.from_ndarray(pcbs, expected_dtype=PCBType),
        cbbs=Tab.from_ndarray(cbbs, expected_dtype=CBBType),
        cycles=Tab.from_ndarray(cycles, expected_dtype=CycleKindType),
        pep_recalls=Tab.from_ndarray(pep_recalls, expected_dtype=RecallType),
        count_only=count_only,
        n_max_dyts=int(n_max_dyts),
        n_max_dytpeps=int(n_max_dytpeps),
        n_dyt_row_bytes=n_dyt_row_bytes,
        n_max_dyt_hash_recs=int(n_max_dyt_hash_recs),
        n_max_dytpep_hash_recs=int(n_max_dytpep_hash_recs),
        pep_i_to_pcb_i=Tab.from_ndarray(pep_i_to_pcb_i, expected_dtype=IndexType),
        out_counts=Tab.from_ndarray(
            out_counts, expected_dtype=SizeType
        ),  # ((n_dyts, n_dytpeps))
        _pep_recalls=pep_recalls,
    )

    error = lib.context_init(ctx)
    if error is not None:
        raise CException(error)

    try:
        yield ctx
    finally:
        lib.context_free(ctx)


def _do_dytsim_pep_batch(
    ctx: DytSimContext, start_pep_i: int, stop_pep_i: int, seed: Optional[int] = None
):
    """
    The worker of the zap to run a batch of peptides
    """
    assert type(ctx) == DytSimContext
    assert type(start_pep_i) == int
    assert type(stop_pep_i) == int

    assert 0 <= start_pep_i < stop_pep_i

    # TODO: Remove when Flyte fixes Optional[int] behavior
    # Flyte internally converts Optional[int] types to float
    # so the RNG seed has to be returned to int
    if seed is not None:
        seed = int(seed)
    lib = load_lib()

    rng = RNG(seed)
    error = lib.dytsim_batch(ctx, rng, start_pep_i, stop_pep_i)
    if error is not None:
        raise CException(error)


def c_dytsim(
    pcbs: np.ndarray,  # pcb = (p)ep_i, (c)h_i, (b)right_prob == like a "flu" with brightness probability)
    cbbs: np.ndarray,
    n_samples: int,
    n_channels: int,
    n_labels: int,
    cycles: np.ndarray,
    p_cyclic_block: float,
    p_initial_block: float,
    p_detach: float,
    p_edman_fail: float,
    p_label_fail: float,
    allow_edman_cterm: bool,
    *,
    seed: Optional[int] = None,
    progress=None,
    use_zap: bool = True,
    start_idx: Optional[int] = None,
    stop_idx: Optional[int] = None,
):
    """
    Generate dye tracks (dyts) by Monte Carlo sampling

    Arguments:
        pcbs: This is an encoding of flus. See SimV3Params.j()
            Each peptide has a row per amino-acid and either a
            channel number or a np.nan to indicate a label at that
            position, plus a p_bright for that aa.
            n_samples: number of samples to try ...
                BUT NOT NEC. THE NUMBER RETURNED! -- because
                all-dark samples are not returned.
                See "Dealing with dark-rows" above

    Returns:
        dyemat: ndarray(n_uniq_dyetracks, n_channels, n_cycle)
        dytpep: ndarray(dye_i, pep_i, count)
        pep_recalls: ndarray(n_peps)
    """
    # TODO: Remove when Flyte fixes Optional[int] behavior
    # Flyte internally converts Optional[int] types to float
    # so the RNG seed has to be returned to int
    if seed is not None:
        seed = int(seed)
    lib = load_lib()

    with context(
        pcbs,
        cbbs,
        n_samples,
        n_channels,
        n_labels,
        cycles,
        p_cyclic_block,
        p_initial_block,
        p_detach,
        p_edman_fail,
        p_label_fail,
        allow_edman_cterm,
    ) as ctx:
        n_peps = int(np.max(pcbs[:, 0])) + 1

        if use_zap:
            batches = zap.make_batch_slices(n_rows=n_peps, _batch_updates=1)
            # See similar comments in c_radsim about process vs thread
            with zap.Context(trap_exceptions=False, mode="thread", progress=progress):
                zap.work_orders(
                    [
                        dict(
                            fn=_do_dytsim_pep_batch,
                            ctx=ctx,
                            start_pep_i=batch[0],
                            stop_pep_i=batch[1],
                            seed=seed,
                        )
                        for batch in batches
                    ]
                )
        else:
            if start_idx is None:
                start_idx = 0
            if stop_idx is None:
                stop_idx = int(pcbs[-1, 0]) + 1
            _do_dytsim_pep_batch(ctx, start_idx, stop_idx, seed=seed)

        # CONVERT from (expandable) tables to a fixed size
        # The results are in the (expandable) Tables ctx.dyts and ctx.dytpeps.
        # The dytmats have an additional nul row added so a +1 is added
        n_chcy = ctx.n_channels * ctx.n_cycles
        dytmat = np.zeros((ctx._dytrecs.n_rows + 1, n_chcy), dtype=DytType)
        assert dytmat.dtype == DytType
        assert dytmat.flags["C_CONTIGUOUS"]

        # We reserve a nul-record at [0] so we need to add one here
        # and then skip it during copy
        dytpeps = np.zeros((ctx._dytpeps.n_rows + 1, 3), dtype=DytPepType)
        assert dytpeps.dtype == DytPepType
        assert dytpeps.flags["C_CONTIGUOUS"]

        lib.copy_results(
            ctx,
            dytmat.ctypes.data_as(
                c.POINTER(c_common_tools.typedef_to_ctype("DytType"))
            ),
            dytpeps.ctypes.data_as(
                c.POINTER(c_common_tools.typedef_to_ctype("DytPepType"))
            ),
        )

        # SORT dytmat lexically so that dimmest row is first and remap dytpeps accordingly
        n_rows, n_cols = dytmat.shape
        lex_cols = tuple(dytmat[:, n_cols - i - 1] for i in range(n_cols))
        sort_args = np.lexsort(lex_cols)
        lut = np.zeros((n_rows,), dtype=int)
        lut[sort_args] = np.arange(n_rows, dtype=int)
        dytpeps[:, 0] = lut[dytpeps[:, 0]]

        # SORT dytpeps so that they are by pep first and count second
        dytpeps = dyt_helpers.sort_dytpeps(dytpeps)

        return dytmat[sort_args], dytpeps, ctx._pep_recalls


def xrng_uint64(xrng: int):
    lib = load_lib()
    return lib.xrng_uint64(xrng)
