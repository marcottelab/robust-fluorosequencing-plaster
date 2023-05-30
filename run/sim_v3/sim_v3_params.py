import logging

_log = logging.getLogger(__name__)

import string
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin

from plaster.tools.aaseq.aaseq import aa_str_to_list
from plaster.tools.c_common import c_common_tools


def _clip(obj: Any, k: str, min_: Number, max_: Number) -> None:
    """
    Clip value of obj's attribute k to range [min_, max_].
    """
    v = getattr(obj, k)
    if not min_ <= v <= max_:
        _log.warning(f"{k} clipped from {v} to [{min_}., {max_}]")
        setattr(obj, k, np.clip(v, min_, max_))


# TODO: Write a class method to parse a list of these into Labels
@dataclass
class Marker(DataClassJsonMixin):
    """
    Configuration information for dye markers attached to amino acids.

    Attributes
    ----------
    aa : str
        Amino acid[s] to which the markers are attached. One or more uppercase
        letters, eg "C" or "DE". If multiple letters are specified, then
        each letter will receive the same marker.
    gain_mu : float
        Signal mean for a single instance of this marker. (Greater than 0.)
    gain_sigma : float
        Signal standard deviation for a single instance of this marker. (Greater than 0.)
    bg_mu : float
        Background mean for this marker channel.
    bg_sigma : float
        Background standard deviation for this marker channel. (Greater than 0.)
    p_bleach : float
        Per-cycle bleaching probability for this marker. (Between 0. and 1.)
    p_dud : float
        Dud dye probability for this marker. (Between 0. and 1.)
    """

    aa: str
    gain_mu: float = 8000.0
    gain_sigma: float = 1200.0
    bg_mu: float = 0.0
    bg_sigma: float = 150.0
    p_bleach: float = 0.06
    p_dud: float = 0.18

    def __post_init__(self):
        self.aa = "".join(sorted(set(self.aa)))
        if set(self.aa) - set(string.ascii_uppercase):
            raise ValueError("aa must be alphabetic and uppercase")
        if any([x < 0.0 for x in [self.gain_mu, self.gain_sigma, self.bg_sigma]]):
            raise ValueError(
                "gain_mu, gain_sigma, and bg_sigma all must be greater than 0."
            )

        for k in ["p_bleach", "p_dud"]:
            _clip(self, k, 0.0, 1.0)


@dataclass
class SeqParams(DataClassJsonMixin):
    """
    Configuration information for fluorosequencing runs.

    Attributes
    ----------
    n_pre : int
        Number of preliminary illumination cycles used for image calibration.
    n_mocks : int
        Number of chemical wash cycles done without Edman degradation.
    n_edmans : int
        Number of Edman degradation cycles.
    allow_edman_cterm : bool
        Allow Edman degradation of the C-terminal amino acid. If False, the C-terminal
        amino acid will remain regardless of the number of Edman degradation steps.
    p_cyclic_block : float
        Probability of the peptide suffering irreversible damage per Edman cycle.
        This will cause subsequent Edman cycles to fail. (Between 0. and 1.)
    p_initial_block : float
        Probability of the peptide suffering irreversible damage before the
        first Edman cycle.
        This will cause subsequent Edman cycles to fail. (Between 0. and 1.)
    p_detach : float
        Probability of the peptide detaching per cycle. (Between 0 and 1.)
    p_edman_failure : float
        Probability that an Edman cycle will fail to detach an amino acid. (Between 0 and 1.)
    row_k_sigma : float
        Row-to-row standard deviation for all signal gains. Expressed as a fraction of the
        signal gain. (Greater than 0.)
    gain_sigma_scaling_linear : bool
        If True, scale multicount sigma linearly, otherwise (default) by sqrt(count)
    """

    n_pres: int = 1
    n_mocks: int = 0
    n_edmans: int = 15
    allow_edman_cterm: bool = False
    p_cyclic_block: float = 0.005
    p_initial_block: float = 0.2
    p_detach: float = 0.005
    p_edman_failure: float = 0.05
    p_label_failure: float = 0.0
    row_k_sigma: float = 0.0
    gain_sigma_scaling_linear: bool = False

    def __post_init__(self):
        if any([x < 0 for x in [self.n_pres, self.n_mocks, self.n_edmans]]):
            raise ValueError("n_pres, n_mocks, and n_edmans all must be 0 or greater")
        if self.n_pres + self.n_mocks < 1:
            raise ValueError("At least one of n_pres or n_mocks must be > 0")
        if self.row_k_sigma < 0.0:
            raise ValueError("row_k_sigma must be positive.")

        for k in ["p_cyclic_block", "p_initial_block", "p_detach", "p_edman_failure"]:
            _clip(self, k, 0.0, 1.0)


@dataclass
class Fret(DataClassJsonMixin):
    """Class representing a FRET interaction.

    Attributes
    ----------
    donor: int
        The donor dye channel. This channel loses intensity as a result of
        interactions with the acceptor channel.
    acceptor: int
        The acceptor dye channel. This channel is currently unaffected by
        FRET, since we do not currently observe acceptors during donor
        illumination.
    fret_mu: float
        Mean FRET efficiency of the donor-acceptor pair. For a single
        donor-acceptor pair, intensity will be modified by
        (1.0 - rng.normal(fret_mu, fret_sigma)). Values will be limited to
        the range 0.0-1.0.
    fret_sigma: float
        Standard deviation of the FRET efficiency of the donor-acceptor pair.
    flat: bool (default=True)
        If True, the FRET efficiency will be applied to donors once if any
        acceptors or quenching donors are present. If False, the FRET
        efficiency will be applied once per acceptor or quenching donor, and
        the total FRET effect will be = (1.0 - fret) ** n_acceptors.
    """

    donor: int
    acceptor: int
    fret_mu: float
    fret_sigma: float = 0.0
    flat: bool = True

    def __post_init__(self):
        if any([x < 0 for x in [self.donor, self.acceptor]]):
            raise ValueError("donor and acceptor cannot be set to negative values")
        if not 0.0 <= self.fret_mu <= 1.0:
            raise ValueError("fret_mu must be in range [0., 1.]")
        if self.fret_sigma < 0.0:
            raise ValueError("fret_sigma must be >= 0.")


@dataclass
class Label(Marker, DataClassJsonMixin):
    dye_name: Optional[str] = None
    label_name: Optional[str] = None
    channel_name: Optional[str] = None
    ch_i: Optional[int] = None
    ptm_only: bool = False


@dataclass
class SimV3Params(DataClassJsonMixin):

    markers: List[Marker]
    frets: List[Fret] = field(default_factory=list)  # move to priors?
    seq_params: SeqParams = field(default_factory=SeqParams)
    n_samples_train: int = 5000
    n_samples_test: int = 1000
    is_survey: bool = False
    span_dyetracks: bool = False

    n_span_labels: int = 1  # deprecated, use n_span_labels_list instead
    n_span_labels_list: list[int] = field(default_factory=list)

    test_decoys: bool = False
    channels: dict = field(default_factory=dict)
    dyes: List[dict[str, str]] = field(default_factory=list)
    labels: List[Label] = field(default_factory=list)
    seed: Optional[int] = None

    def __post_init__(self):

        self.dyes = [
            dict(dye_name=f"dye_{ch}", channel_name=f"ch_{ch}")
            for ch, _ in enumerate(self.markers)
        ]

        # Note the extra for loop because "DE" needs to be split into "D" & "E"
        # which is done by aa_str_to_list() - which also handles PTMs like S[p]
        self.labels = [
            Label(
                aa=this_aa,
                gain_mu=marker.gain_mu,
                gain_sigma=marker.gain_sigma,
                bg_mu=marker.bg_mu,
                bg_sigma=marker.bg_sigma,
                p_bleach=marker.p_bleach,
                p_dud=marker.p_dud,
                dye_name=f"dye_{ch}",
                label_name=f"label_{ch}",
                channel_name=f"channel_{ch}",
                ch_i=ch,
                ptm_only=False,
            )
            for ch, marker in enumerate(self.markers)
            for this_aa in aa_str_to_list(marker.aa)
        ]

        mentioned_channels = {label.channel_name: False for label in self.labels}
        self.channels = {
            ch_name: i for i, ch_name in enumerate(sorted(mentioned_channels.keys()))
        }

        for fret in self.frets:
            missing_fret_channels = {fret.donor, fret.acceptor} - set(
                self.channels.values()
            )
            if missing_fret_channels:
                error_msg = f"FRET interaction {fret} contains channels not present in params: {missing_fret_channels}"
                raise ValueError(error_msg)

        self.n_channels = len(self.channels)

        # the ints here and below are needed to force Flyte to keep these as ints
        # Flyte somehow decides to force them to floats if they're not explicitly cast
        self.n_cycles = int(
            self.seq_params.n_pres + self.seq_params.n_mocks + self.seq_params.n_edmans
        )

        self.cycles_array = np.zeros(
            (self.n_cycles,), dtype=c_common_tools.CycleKindType
        )
        i = 0
        for _ in range(int(self.seq_params.n_pres)):
            self.cycles_array[i] = c_common_tools.CYCLE_TYPE_PRE
            i += 1
        for _ in range(int(self.seq_params.n_mocks)):
            self.cycles_array[i] = c_common_tools.CYCLE_TYPE_MOCK
            i += 1
        for _ in range(int(self.seq_params.n_edmans)):
            self.cycles_array[i] = c_common_tools.CYCLE_TYPE_EDMAN
            i += 1

        self.label_priors_df = pd.DataFrame(self.labels)
        if "aa" not in self.label_priors_df.columns:
            raise ValueError(
                "aa column not found in label_priors_df. Did you remember to specify markers?"
            )
        self.channel_priors_df = (
            pd.DataFrame(self.labels).drop(columns=["aa"]).drop_duplicates()
        )

        self.ch_by_aa = {
            self.label_priors_df.loc[idx, "aa"]: int(
                self.label_priors_df.loc[idx, "ch_i"]
            )
            for idx in self.label_priors_df.index
        }

        # __post_init__ data field migration
        #
        # 2023.05.16
        # Support the transition from n_span_labels:int to n_span_labels_list:list[int]
        # Elsewhere we will go out of our way to ensure n_span_labels:int is not written
        # to disk, such that eventually, we can remove this code, and the n_span_labels:int
        # field.  Update: this isn't being done yet.
        # See https://app.shortcut.com/erisyon/story/7641/create-schema-migration-functionality-for-dataclass-based-classes
        if not self.n_span_labels_list:
            # If we are being contructed via a serialize from disk of the old version that
            # uses the int, the list will be empty, and we need to convert.
            self.n_span_labels_list = [self.n_span_labels] * self.n_channels
            self.n_span_labels = -1  # cause any downstream use of this to fail

    def pcbs(self, pep_seq_df):
        """
        pcb stands for (p)ep_i, (c)hannel_i, (b)right_probability

        This is a structure that is liek a "flu" but with an extra bright probability.

        Each peptide has a row for each amino acid
            That row has a columns (pep_i, ch_i, p_bright)
            And it will have np.nan for ch_i and p_bright **IF THERE IS NO LABEL**

        bright_probability is the inverse of all the ways a dye can fail to be visible
        ie the probability that a dye is active.

        pep_seq_df: Any DataFrame with an "aa" column

        Returns:
            contiguous ndarray(:, 4) where the 4 columns are:
                pep_i, ch_i, p_bright, p_bleach
        """

        # This join is what effectively "labels" the peptides, as the pcbs returned
        # from this fn is what is used by simulation to produce dye tracks.
        # This is now only true if you are not modeling sequential labeling in which
        # mislabeling may occur, and this needs to be modeled for each sample.
        labeled_pep_df = pep_seq_df.join(
            self.label_priors_df.set_index("aa"), on="aa", how="left"
        )

        # p_dud historically captured both the phenomenon of dyes being duds, but also
        # of a failure of chemistry binding the click-linker to the amino-acid of iterest,
        # and the failure of the click-clack chemistry, and presumably also the failure of
        # the chemistry that binds the dye to the clack -- basically anything that could
        # have caused the dye to not be attached to the amino acid.
        #
        # Some of this may now be captured in p_label_fail, which is used in the
        # sequential labeling model.
        labeled_pep_df.loc[:, "p_bright"] = 1.0 - labeled_pep_df.p_dud

        labeled_pep_df.sort_values(by=["pep_i", "pep_offset_in_pro"], inplace=True)
        return np.ascontiguousarray(
            labeled_pep_df.loc[:, ["pep_i", "ch_i", "p_bright", "p_bleach"]].values
        )

    def cbbs(self) -> np.ndarray:
        """
        cbb stands for (c)hannel (b)right probability (b)leach probability and is similar to
        the fn just above.  If we want to model sequential labeling within dytsim, it means
        we need to know these probabilites per channel, since mislabel events may occur and
        we'll need to modifiy the p_bright and p_bleach probs for any mislabeled aas.

        Returns:
            contiguous array(:, 3) where the 3 columns are:
                ch_i, p_bright, p_bleach
        """
        cbbs_df = self.label_priors_df.drop_duplicates(subset=["ch_i"]).copy()
        cbbs_df["p_bright"] = 1.0 - cbbs_df.p_dud
        cbbs_df.sort_values(by=["ch_i"], inplace=True)
        return np.ascontiguousarray(
            cbbs_df.loc[:, ["ch_i", "p_bright", "p_bleach"]].values
        )
