import enum
from dataclasses import dataclass

import numba as nb
import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin

from plaster.run.base_result import ArrayResult
from plaster.run.base_result_dc import (
    BaseResultDC,
    LazyField,
    generate_flyte_result_class,
    lazy_field,
)
from plaster.run.sim_v3.sim_v3_params import Marker, SeqParams, SimV3Params
from plaster.tools.utils import utils


class CountsField(enum.IntEnum):
    N_HEAD_ALL_CH = 0
    N_TAIL_ALL_CH = 1
    N_DYES_ALL_CH = 2
    N_DYES_MAX_ANY_CH = 3


@nb.jit(
    nb.void(
        nb.int64[::1],
        nb.int64[::1],
        nb.int64,
        nb.int64,
        nb.uint8[:, ::1],
        nb.int64[:, ::1],
    ),
    nopython=True,
    cache=True,
)
def _gen_flus(pep_iz, aa_ch_iz, n_channels, n_cycles, out_flus, out_counts):
    zero = ord("0")
    dot = ord(".")
    space = ord(" ")
    semicolon = ord(";")
    comma = ord(",")
    n_rows = pep_iz.shape[0]
    n_cols = out_flus.shape[1]
    pep_i = 0
    pep_i_start_row_i = 0
    sum_head_per_ch = np.zeros((n_channels,), np.int64)
    sum_tail_per_ch = np.zeros((n_channels,), np.int64)
    for row_i in range(n_rows):
        row_pep_i = pep_iz[row_i]
        if row_pep_i != pep_i:
            # end of a pep run
            n_aa_in_pep = row_i - pep_i_start_row_i
            n_head = min(n_aa_in_pep, n_cycles)
            col_i = 0

            for ch_i in range(n_channels):
                sum_head_per_ch[ch_i] = 0

            while col_i < n_head:
                dst_i = col_i
                src_i = pep_i_start_row_i + col_i
                aa_ch_i = aa_ch_iz[src_i]
                if aa_ch_i == -1:
                    out_flus[pep_i, dst_i] = dot
                else:
                    sum_head_per_ch[aa_ch_i] += 1
                    out_flus[pep_i, dst_i] = aa_ch_i + zero
                col_i += 1

            # Sum up remainders (aka "tail") in each channel
            for ch_i in range(n_channels):
                sum_tail_per_ch[ch_i] = 0

            while col_i < n_aa_in_pep:
                src_i = pep_i_start_row_i + col_i
                aa_ch_i = aa_ch_iz[src_i]
                if aa_ch_i >= 0:
                    sum_tail_per_ch[aa_ch_i] += 1
                col_i += 1

            # pad if pep < n_cycles
            while col_i < n_cycles:
                out_flus[pep_i, col_i] = space
                col_i += 1

            dst_i = n_cycles
            if dst_i < n_cols:
                out_flus[pep_i, dst_i] = space
                dst_i += 1
            if dst_i < n_cols:
                out_flus[pep_i, dst_i] = semicolon
                dst_i += 1

            for ch_i in range(n_channels):
                sum_tail = min(sum_tail_per_ch[ch_i], 99)  # Saturate at 99
                tens = sum_tail // 10
                ones = sum_tail % 10
                if dst_i < n_cols and tens > 0:
                    out_flus[pep_i, dst_i] = tens + zero
                    dst_i += 1
                if dst_i < n_cols:
                    out_flus[pep_i, dst_i] = ones + zero
                    dst_i += 1
                if dst_i < n_cols and ch_i < n_channels - 1:
                    out_flus[pep_i, dst_i] = comma
                    dst_i += 1

            # Fill
            while dst_i < n_cols:
                out_flus[pep_i, dst_i] = space
                dst_i += 1

            # Counts
            sum_head = np.sum(sum_head_per_ch)
            sum_tail = np.sum(sum_tail_per_ch)
            max_all_ch = 0
            for ch_i in range(n_channels):
                max_all_ch = max(
                    max_all_ch, sum_head_per_ch[ch_i] + sum_tail_per_ch[ch_i]
                )

            out_counts[pep_i, CountsField.N_HEAD_ALL_CH.value] = sum_head
            out_counts[pep_i, CountsField.N_TAIL_ALL_CH.value] = sum_tail
            out_counts[pep_i, CountsField.N_DYES_ALL_CH.value] = sum_head + sum_tail
            out_counts[pep_i, CountsField.N_DYES_MAX_ANY_CH.value] = max_all_ch

            pep_i_start_row_i = row_i
            pep_i = row_pep_i


def gen_flus(pep_seqs_df, ch_by_aa, n_cycles, has_header=False):
    """
    Generates fluorosequence string like: "..0.1..1. ;1,2
    and adds in various counting statistics.  Note that the "head" portion
    of the flu is exactly n_edmans long, since edmans are the only kind of
    cycles that reveal a dye location.

    The core of this is implemented in numba in _gen_flus which
    operates on a large numeric numpy array.

    Args:
        pep_seqs_df: dataframe as described in PrepV2Result
        ch_by_aa: dict mapping aa to channel (-1 for non-labeled aas)
        n_cycles: number of edman cycles
        has_header: True if first row of pep_seqs_df is header

    Returns:
        tuple: (flus,out_counts) where flus is a list of strings, and out_counts is
                a tuple (a,b,c,d), a=sum_head, b=sum_tail, c=total, d=max_all_ch

    """

    df = pep_seqs_df.sort_values(["pep_i", "pep_offset_in_pro"]).reset_index(drop=True)

    if not has_header:
        assert df.iloc[0].aa != "", "Existing header found"
        df.loc[-1] = [
            0,
            "",
            0,
        ]  # header,  Note -1 is BEFORE the 0th index, not -1 in the python list sense
        df = df.sort_index().reset_index(drop=True)

    assert df.iloc[-1].aa != "", "Existing footer found"

    df = df.sort_index().reset_index(drop=True)
    n_rows = len(df)
    last_pep_i = df.loc[n_rows - 1].pep_i
    df.loc[n_rows] = [last_pep_i + 1, "", 0]  # footer
    df = df.sort_index().reset_index(drop=True)

    aa_i_by_aa = {
        aa: ch_by_aa.get(aa, -1) for aa_i, aa in enumerate(np.unique(df["aa"].values))
    }
    n_channels = np.max(list(ch_by_aa.values())) + 1
    df["aa_i"] = df.aa.replace(aa_i_by_aa)

    # 3 for each channel like "10," and +2 for the space and semicolon
    flu_len = n_cycles + n_channels * 3 + 2
    n_peps = np.max(df.pep_i.unique())

    # Confirm the header/footer were added
    assert df.iloc[0].pep_i == 0 and df.iloc[1].pep_i != 0
    assert df.iloc[-1].pep_i == n_peps and df.iloc[-2].pep_i != n_peps

    out_flus = np.zeros((n_peps, flu_len), np.uint8)
    out_counts = np.zeros((n_peps, len(CountsField)), np.int64)
    _gen_flus(
        df.pep_i.values, df.aa_i.values, n_channels, n_cycles, out_flus, out_counts
    )
    str_buffer = out_flus.tobytes().decode("ascii")
    from textwrap import wrap

    flus = wrap(str_buffer, flu_len)

    return flus, out_counts


@dataclass
class SimV3Result(BaseResultDC, DataClassJsonMixin):
    name = "sim_v3"
    filename = "sim_v3.pkl"

    params: SimV3Params
    _train_dytmat: LazyField = lazy_field(
        np.ndarray
    )  # unique (n_rows, n_channels * n_cycles)
    _train_pep_recalls: LazyField = lazy_field(np.ndarray)
    _train_dytpeps: LazyField = lazy_field(
        np.ndarray
    )  # (n, 3) where 3 are: (dyt_i, pep_i, count)
    _train_radmat: LazyField = lazy_field(ArrayResult)
    _train_true_pep_iz: LazyField = lazy_field(ArrayResult)
    _train_true_dye_iz: LazyField = lazy_field(ArrayResult)
    _train_true_row_ks: LazyField = lazy_field(ArrayResult)
    _test_dytmat: LazyField = lazy_field(np.ndarray)
    _test_radmat: LazyField = lazy_field(ArrayResult)
    _test_true_dye_iz: LazyField = lazy_field(ArrayResult)
    _test_true_pep_iz: LazyField = lazy_field(ArrayResult)
    _test_true_row_ks: LazyField = lazy_field(ArrayResult)
    _flus: LazyField = lazy_field(pd.DataFrame)  # Generated by this module

    @property
    def train_dytmat(self):
        return self._train_dytmat.get()

    @train_dytmat.setter
    def train_dytmat(self, val):
        self._train_dytmat.set(val)

    @property
    def train_pep_recalls(self):
        return self._train_pep_recalls.get()

    @train_pep_recalls.setter
    def train_pep_recalls(self, val):
        self._train_pep_recalls.set(val)

    @property
    def train_dytpeps(self):
        return self._train_dytpeps.get()

    @train_dytpeps.setter
    def train_dytpeps(self, val):
        self._train_dytpeps.set(val)

    @property
    def train_radmat(self):
        return self._train_radmat.get()

    @train_radmat.setter
    def train_radmat(self, val):
        self._train_radmat.set(val)

    @property
    def train_true_pep_iz(self):
        return self._train_true_pep_iz.get().arr()

    @train_true_pep_iz.setter
    def train_true_pep_iz(self, val):
        self._train_true_pep_iz.set(val)

    @property
    def train_true_dye_iz(self):
        return self._train_true_dye_iz.get()

    @train_true_dye_iz.setter
    def train_true_dye_iz(self, val):
        self._train_true_dye_iz.set(val)

    @property
    def train_true_row_ks(self):
        return self._train_true_row_ks.get()

    @train_true_row_ks.setter
    def train_true_row_ks(self, val):
        self._train_true_row_ks.set(val)

    @property
    def test_dytmat(self):
        return self._test_dytmat.get()

    @test_dytmat.setter
    def test_dytmat(self, val):
        self._test_dytmat.set(val)

    @property
    def test_radmat(self):
        return self._test_radmat.get()

    @test_radmat.setter
    def test_radmat(self, val):
        self._test_radmat.set(val)

    @property
    def test_true_dye_iz(self):
        return self._test_true_dye_iz.get()

    @test_true_dye_iz.setter
    def test_true_dye_iz(self, val):
        self._test_true_dye_iz.set(val)

    @property
    def test_true_pep_iz(self):
        return self._test_true_pep_iz.get()

    @test_true_pep_iz.setter
    def test_true_pep_iz(self, val):
        self._test_true_pep_iz.set(val)

    @property
    def test_true_row_ks(self):
        return self._test_true_row_ks.get()

    @test_true_row_ks.setter
    def test_true_row_ks(self, val):
        self._test_true_row_ks.set(val)

    def __repr__(self):
        try:
            return f"SimV3Result with {self.dyemat.shape[0]} rows ; with {self.dyemat.shape[1]} features"
        except Exception:
            return "SimV3Result"

    def _generate_flu_info(self, prep_results):
        """
        Generates fluoro-sequence string like: "..0.1..1. ;1,2
        and adds in various counting statistics.  Note that the "head" portion
        of the flu is exactly n_edmans long, since edmans are the only kind of
        cycles that reveal a dye location.

        This needs to be moved into C but this is non-trivial because the
        aa's are stored as strings. Mostly these are one character strings
        implying we could make an easy lookup table but longer-than-one-character
        strings are possible in the form: "S[p]" or similar.

        Thus we'd need to make a more complex lookup table and this will
        take some effort.

        Meanwhile to speed this up I'm going to just parallelize it.
        """

        if len(prep_results.pepseqs()) > 0:
            flustrs, counts = gen_flus(
                prep_results.pepseqs(),
                self.params.ch_by_aa,
                self.params.seq_params.n_edmans,
                has_header=True,
            )
            n_peps = len(flustrs)

            df = pd.DataFrame(
                dict(
                    pep_i=np.arange(n_peps),
                    flustr=flustrs,
                    n_head_all_ch=counts[:, CountsField.N_HEAD_ALL_CH],
                    n_tail_all_ch=counts[:, CountsField.N_TAIL_ALL_CH],
                    n_dyes_all_ch=counts[:, CountsField.N_DYES_ALL_CH],
                    n_dyes_max_any_ch=counts[:, CountsField.N_DYES_MAX_ANY_CH],
                )
            )

            # note that flu_count tells you how unique a flu is, but if allow_edman_cterm is False
            # for a sim, identical flus can produce different dyetracks!  So in that case this is
            # no longer a measure of uniqueness.
            df_flu_count = df.groupby("flustr").size().reset_index(name="flu_count")
            self._flus.set(
                (
                    df.set_index("flustr")
                    .join(df_flu_count.set_index("flustr"))
                    .reset_index()
                )
            )
        else:
            self._flus.set(pd.DataFrame(columns=["pep_i", "flustr"]))

    def flat_train_radmat(self):
        assert self.train_radmat.arr().ndim == 3
        return utils.mat_flatter(self.train_radmat.arr())

    def flat_test_radmat(self):
        assert self.test_radmat.arr().ndim == 3
        return utils.mat_flatter(self.test_radmat.arr())

    def flus(self):
        return self._flus.get()

    @property
    def n_channels(self):
        return self.params.n_channels

    def peps__flus(self, prep_result):
        return (
            prep_result.peps()
            .set_index("pep_i")
            .join(self._flus.get().set_index("pep_i"))
            .sort_index()
            .reset_index()
        )

    def peps__flus__unique_flus(self, prep_result):
        df = self.peps__flus(prep_result)
        return df[df.flu_count == 1]

    def pros__peps__pepstrs__flus(self, prep_result):
        return (
            prep_result.pros__peps__pepstrs()
            .set_index("pep_i")
            .join(self._flus.get().set_index("pep_i"))
            .sort_index()
            .reset_index()
        )

    @classmethod
    def from_prep_v2_fixture(cls, prep_v2_result, folder, markers: str, n_edmans=5):
        """
        Run a (likely small) simulation to make a SimResult fixture for testing

        labels: a CSV list of aas. Eg: "DE,C"
            Common labels: "DE", "C", "Y", "K", "H"
        """
        from plaster.run.sim_v3.sim_v3_worker import sim_v3

        assert type(markers) == str
        markers = markers.split(",")

        sim_v3_params = SimV3Params(
            markers=[Marker(aa=marker) for marker in markers],
            seq_params=SeqParams(n_edmans=n_edmans),
            n_samples_train=100,
            n_samples_test=20,
        )

        return sim_v3(sim_v3_params, prep_v2_result, folder)


SimV3FlyteResult = generate_flyte_result_class(SimV3Result)
