import ctypes as c
import pathlib
import sysconfig
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
from plumbum import local

from plaster.tools.c_common import c_common_tools
from plaster.tools.c_common.c_common_tools import Tab
from plaster.tools.schema import check
from plaster.tools.utils import utils

_lib = None

MODULE_DIR = pathlib.Path(__file__).parent


def load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib = c.CDLL(
        (MODULE_DIR / "survey_v2_ext").with_suffix(
            sysconfig.get_config_var("EXT_SUFFIX")
        )
    )

    # C_COMMON
    lib.sanity_check.argtypes = []
    lib.sanity_check.restype = c.c_int

    # SURVEY_V2
    lib.context_start.argtypes = [
        c.POINTER(SurveyV2Context),
    ]
    lib.context_start.restype = c.c_int

    _lib = lib

    assert lib.sanity_check() == 0

    return lib


# WARNING: Updating this class, derived from
#     c_common_tools.FixupStructure,
# requires updating the corresponding *.h to maintain
# consistency between C and Python.
class SurveyV2Context(c_common_tools.FixupStructure):
    _fixup_fields = [
        ("dytmat", Tab, "Uint8"),
        ("dytpeps", Tab, "Uint64"),
        ("pep_i_to_dytpep_row_i", Tab, "Uint64"),
        ("dyt_i_to_n_reads", Tab, "Uint64"),
        ("dyt_i_to_mlpep_i", Tab, "Uint64"),
        ("output_pep_i_to_isolation_metric", Tab, "Float32"),
        ("output_pep_i_to_mic_pep_i", Tab, "Uint64"),
        ("next_pep_i", "Index"),
        ("n_threads", "Size"),
        ("n_flann_cores", "Size"),
        ("n_peps", "Size"),
        ("n_neighbors", "Size"),
        ("n_dyts", "Size"),
        ("n_dyt_cols", "Size"),
        ("distance_to_assign_an_isolated_pep", "Float32"),
        ("work_order_lock", "pthread_mutex_t *"),
        ("flann_params", "struct FLANNParameters *"),
        ("flann_index_id", "void *"),  # typedef void* flann_index_t;
    ]


SurveyV2Context.struct_fixup()

global_progress_callback = None


@c.CFUNCTYPE(c.c_voidp, c.c_int, c.c_int, c.c_int)
def progress_fn(complete, total, retry):
    if global_progress_callback is not None:
        global_progress_callback(complete, total, retry)


IsolationNPType = np.float32


# Wrapper for survey that prepares buffers for csurvey
def survey(
    n_peps,
    dytmat,
    dytpeps,
    n_threads=1,
    progress=None,
):
    lib = load_lib()

    # Vars
    n_dyts = dytmat.shape[0]

    pep_column_in_dytpeps = 1

    global global_progress_callback
    global_progress_callback = progress

    # SETUP the dytmat table
    check.array_t(dytmat, dtype=np.uint8, c_contiguous=True)

    # BUILD a LUT from dyt_i to most-likely peptide i (mlpep_i)
    # The dytpep_df can have missing pep_i (there are peptides that have no dyt_i)
    # But all dyt have peps.
    dytpep_df = pd.DataFrame(dytpeps, columns=["dyt_i", "pep_i", "n_reads"])

    # EXTRACT the row in each dyt_i group that has the most reads; this is the Most-Likely-Pep
    dyt_i_to_mlpep_i = dytpep_df.loc[
        dytpep_df.groupby(["dyt_i"])["n_reads"].idxmax()
    ].reset_index()
    assert np.unique(dyt_i_to_mlpep_i.dyt_i).tolist() == list(range(n_dyts))

    dyt_i_to_mlpep_i = dyt_i_to_mlpep_i.pep_i.values
    assert (len(dyt_i_to_mlpep_i)) == n_dyts
    dyt_i_to_mlpep_i = np.ascontiguousarray(dyt_i_to_mlpep_i, dtype=np.uint64)
    check.array_t(dyt_i_to_mlpep_i, dtype=np.uint64, c_contiguous=True)

    # FILL-in missing pep_i from the dataframe
    # This is tricky because there can be duplicate "pep_i" rows and the simple reindex
    # answer from SO doesn't work in that case so we need to make a list of the missing rows
    new_index = pd.Index(np.arange(n_peps), name="pep_i")

    # Drop duplicates from dytpep_df so that the reindex can work...
    missing = dytpep_df.drop_duplicates("pep_i").set_index("pep_i").reindex(new_index)

    # Now missing has all rows, and the "new" rows (ie those that were missing in dytpep_df)
    # have NaNs in their dyt_i fields, so select those out.
    missing = missing[np.isnan(missing.dyt_i)].reset_index()

    # Now we can merge those missing rows into the dytpep_df
    dytpep_df = pd.merge(
        dytpep_df, missing, on="pep_i", how="outer", suffixes=["", "_dropme"]
    ).drop(columns=["dyt_i_dropme", "n_reads_dropme"])
    dytpep_df = dytpep_df.sort_values(["pep_i", "dyt_i"]).reset_index(drop=True)
    dytpep_df = dytpep_df.fillna(0).astype(np.uint64)

    # SETUP the dyt_i_to_n_reads
    assert np.unique(dytpep_df.dyt_i).tolist() == list(range(n_dyts))
    dyt_i_to_n_reads = np.ascontiguousarray(
        dytpep_df.groupby("dyt_i").sum().reset_index().n_reads.values, dtype=np.uint64
    )

    # SETUP the dytpeps tab, sorting by pep_i.  All pep_i must occur in this.
    assert np.unique(dytpep_df.pep_i).tolist() == list(range(n_peps))
    dytpeps = np.ascontiguousarray(dytpep_df.values, dtype=np.uint64)
    check.array_t(dytpeps, dtype=np.uint64, c_contiguous=True)

    _pep_i_to_dytpep_row_i = np.unique(dytpep_df.pep_i, return_index=1)[1].astype(
        np.uint64
    )
    pep_i_to_dytpep_row_i = np.zeros((n_peps + 1), dtype=np.uint64)
    pep_i_to_dytpep_row_i[0:n_peps] = _pep_i_to_dytpep_row_i
    pep_i_to_dytpep_row_i[n_peps] = dytpeps.shape[0]
    check.array_t(pep_i_to_dytpep_row_i, dtype=np.uint64, c_contiguous=True)

    # SANITY CHECK
    # print(", ".join([f"{i}" for i in pep_i_to_dytpep_row_i.tolist()]))
    assert np.all(np.diff(pep_i_to_dytpep_row_i) >= 0), "bad pep_i_to_dytpep_row_i"

    pep_i_to_isolation_metric = np.zeros((n_peps,), dtype=IsolationNPType)
    check.array_t(pep_i_to_isolation_metric, dtype=IsolationNPType, c_contiguous=True)
    pep_i_to_mic_pep_i = np.zeros((n_peps,), dtype=np.uint64)
    check.array_t(pep_i_to_mic_pep_i, dtype=np.uint64, c_contiguous=True)

    ctx = SurveyV2Context(
        dytmat=Tab.from_ndarray(dytmat, expected_dtype=np.uint8),
        dytpeps=Tab.from_ndarray(dytpeps, expected_dtype=np.uint64),
        pep_i_to_dytpep_row_i=Tab.from_ndarray(
            pep_i_to_dytpep_row_i, expected_dtype=np.uint64
        ),
        dyt_i_to_n_reads=Tab.from_ndarray(dyt_i_to_n_reads, expected_dtype=np.uint64),
        dyt_i_to_mlpep_i=Tab.from_ndarray(dyt_i_to_mlpep_i, expected_dtype=np.uint64),
        output_pep_i_to_isolation_metric=Tab.from_ndarray(
            pep_i_to_isolation_metric, expected_dtype=IsolationNPType
        ),
        output_pep_i_to_mic_pep_i=Tab.from_ndarray(
            pep_i_to_mic_pep_i, expected_dtype=np.uint64
        ),
        n_threads=1,
        n_flann_cores=n_threads,
        n_peps=n_peps,
        n_neighbors=10,
        n_dyts=n_dyts,
        n_dyt_cols=dytmat.shape[1],
        distance_to_assign_an_isolated_pep=10,  # TODO: Find this by sampling.
        progress_fn=progress_fn,
    )

    lib.context_start(ctx)

    return pep_i_to_mic_pep_i, pep_i_to_isolation_metric
