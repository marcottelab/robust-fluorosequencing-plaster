"""
This is the "Virtual Fluoro-Sequencer". It uses Monte-Carlo simulation
to sample to distribution of dyetracks created by error modes of each
labelled peptide.

Nomenclature
    Flu
        A vector with a 0 or 1 in each position (1 means there's a dye in that channel at that position)
    PCB:
        (p)ep, (c)hannel, (b)rightness probability
        This is lise a "flu" in that each amino-acid position
        in a peptide has a channel number or a np.nan.
        It also has another column for the probability that
        the particular label is bright (ie not a dud)
    Evolution
        The sim makes an array of n_sample copies of the flus and then modifies those along each cycle.
        evolution has shape: (n_samples, n_channels, len_of_peptide, n_cycles).
    Dye Tracks, aka "dyts"
        The dye counts are the sum along the axis=3 of evolution
    Cycles:
        There's three kinds of chemical cycles: Pres, Mocks, Edmans.
        At moment we treat pres and mocks the same in the sim
        but in reality they are different because bleaching effects are
        actually caused by both light and chemistry. We currently
        conflate these two effects into one parameters which makes
        pres and mocks essentially the same.
        The imaging happens _after_ the cycle. So, PMEEE means there are 5 images,
        ie. the first image is _after_ the first pre.
    Radiometry:
        The brightness space in which real data from a scope lives.
        Each channel (dye) has different properties of brightess and variance.
        When the simulator runs, it produced "dyetracks" which are
        similar to radiometry except they have no noise and unit-brightness for all dyes.
    dyemat:
        A matrix form of of the dyetracks. Maybe either be 3 dimensional (n_samples, n_channels, n_cycles)
        or can be unwound into a 2-D mat like: (n_samples, n_channels * n_cycles)
    dytpeps:
        A structure that maps the index of the dyemat row (dyt_i) to
        a given peptide (pep_i) with a count indiciating how many
        times the pep_i generated the specific dyt_i.
    pep_recalls:
        The fraction of the dyt simulations for a given peptide that
        generated non-dark (ie observable) dyetracks.
    radmat:
        Similar to dyemat, but in radiometry space.
        This is a floating point value of dyemat times the gain and with noise.
    p_*:
        The probability of an event


Dealing with dark-rows
    When a peptide is simulated to generate dyetracks (dyts)
    in _dytmat_sim a peptide may result in some all-zero dyetracks
    and these are NOT in the resulting dytpeps structure.
    But, they have been accounted for by a decreease in the pep_recalls
    that is returned.

    When the dye tracks are later converted into radiometry
    samples this is done so that each peptide is trained on
    the same number of samples which means that we must
    RESAMPLE the dyetracks.

    For example:

        Suppose peptide 1 had 50% recall meaning that it generated
        500 dytpep counts total on a requested 1000.

        Suppose peptide 2 had 100% recall.

        When we sample radmat in _radmat_sim we want the
        same number of radmat rows to the radrows from peptide 1
        so we resample from the set to get 1000 rows.

        In the case that recall is perfect then there is not need to
        resample.

Dealing with unlabelled peptides
    Sometimes, a peptide will have no labels and thus has 0% recall.
    These all-dark samples will be unsampled in the _dytmat_sim().


    Note that the output buffers for train_radmat and train_dyt_iz,
    etc. are PRE-ALLOCATED (n_samples * n_peps) and therefore
    the unsampled rows have to be truncated.

    Remember that later calculation of the peptide the recall
    must include the pep_recalls term not just rely on the
    recall per call.  See Callbag

"""
import itertools
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import structlog

from plaster.run.base_result import ArrayResult
from plaster.run.prep_v2.prep_v2_result import PrepV2Result
from plaster.run.prep_v2.prep_v2_worker import photobleaching_fixture
from plaster.run.sim_v3 import dyt_helpers
from plaster.run.sim_v3.c_dytsim import dytsim
from plaster.run.sim_v3.c_radsim import radsim
from plaster.run.sim_v3.sim_v3_params import Fret, Marker, SeqParams, SimV3Params
from plaster.run.sim_v3.sim_v3_result import SimV3Result
from plaster.tools.c_common.c_common_tools import DytPepType, RadType, RowKType


def _dytmat_sim(
    sim_params: SimV3Params,
    pcbs: np.ndarray,
    n_samples: int,
    use_zap: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert type(sim_params) == SimV3Params
    assert pcbs.shape[1] == 4

    # get bright and bleach probs per channel for sequential-label model
    cbbs = sim_params.cbbs()

    dyemat, dytpeps, pep_recalls = dytsim.c_dytsim(
        pcbs,
        cbbs,
        n_samples,
        sim_params.n_channels,
        len(sim_params.labels),
        sim_params.cycles_array,
        sim_params.seq_params.p_cyclic_block,
        sim_params.seq_params.p_initial_block,
        sim_params.seq_params.p_detach,
        sim_params.seq_params.p_edman_failure,
        sim_params.seq_params.p_label_failure,
        sim_params.seq_params.allow_edman_cterm,
        seed=sim_params.seed,
        progress=None,
        use_zap=use_zap,
    )

    return dyemat, dytpeps, pep_recalls


def _sample_pep_dytmat(dytpep_group, n_samples):
    """
    Sample a peptide's dyetracks with probability weighted
    by the dytpep's counts.

    See "Dealing with dark-rows" above.

    Arguments:
        dytpep_group: ndarray[n_dts_for_one_peptide, 3]
        n_samples: number of samples requested
    Returns:
        Array of indicies into the dyetracks (ie sampling 0-th column
        of dyetracks)
    """

    dyt_iz = dytpep_group[:, 0]
    counts = dytpep_group[:, 2]

    sum_counts = counts.sum()
    # sum_counts is the number of NON-DARK samples that
    # was sampled during dyetrack simulation
    # it might be less than n_samples_per_pep but
    # in the case that it is identical then there is no need to resample

    if dytpep_group.shape[0] > 0 and sum_counts > 0:

        if sum_counts == n_samples:
            # No need to resample, just return the dyt_iz repeated out
            return np.repeat(dyt_iz.astype(int), counts.astype(int))

        else:
            # Resample by the probability of occurrence
            prob = counts / sum_counts
            return np.random.choice(dytpep_group[:, 0], n_samples, p=prob)

    else:
        # No rows, return stub
        return np.zeros((0,), dtype=int)


def fret_mod(
    dyetrack: np.ndarray,
    frets: Sequence[Fret],
    rng: np.random._generator.Generator,
) -> np.ndarray:
    """Model FRET effects on a dyetrack.

    Given a dyetrack, return a new dyetrack with donor dye intensities
    modified by a FRET factor during cycles where acceptor dyes are present.

    Args
    ----
    dyetrack: np.ndarray
        The input dyetrack.
    frets: Sequence[Fret]
        A Sequence that uses Fret objects to describe FRET
        effects on the system. Donor channel intensities will be reduced by
        (1.0 - rng.normal(Fret.fret_mu, Fret.fret_sigma)) while dyes are present
        in the acceptor channel. FRET effects will be limited to the range
        0.0-1.0. Frets will be applied in order.
        Note that self-quenching interactions can be modeled by specifying
        the same channel for donor and acceptor. In this case, donor intensity
        will be reduced as long as >1 dye is present in the channel.

    Returns
    -------
    A FRET-modified output dyetrack of the same shape as the input dyetrack.

    """
    new_dyetrack = dyetrack.astype(float)
    for item in frets:
        fret_factor = np.clip(1.0 - rng.normal(item.fret_mu, item.fret_sigma), 0.0, 1.0)
        if item.donor == item.acceptor:
            if item.flat:
                new_dyetrack[item.donor, :] = new_dyetrack[item.donor, :] * (
                    fret_factor ** (dyetrack[item.acceptor, :] > 1)
                )
            else:
                new_dyetrack[item.donor, :] = new_dyetrack[item.donor, :] * (
                    fret_factor ** np.maximum((dyetrack[item.acceptor, :] - 1.0), 0.0)
                )
        else:
            if item.flat:
                new_dyetrack[item.donor, :] = new_dyetrack[item.donor, :] * (
                    fret_factor ** (dyetrack[item.acceptor, :] > 0)
                )
            else:
                new_dyetrack[item.donor, :] = new_dyetrack[item.donor, :] * (
                    fret_factor ** dyetrack[item.acceptor, :]
                )
    return new_dyetrack


def _radmat_sim(
    dytmat_cube: np.ndarray,
    dytpeps: np.ndarray,
    sim_params: SimV3Params,
    n_samples: int,
    out_radmat: np.ndarray,
    out_row_ks: np.ndarray,
    out_dyt_iz: np.ndarray,
    out_pep_iz: np.ndarray,
    *,
    seed: Optional[int] = None,
    skip_resample: bool = True,
    gain_sigma_scaling_linear=False,
):
    """Convert a dyetrack matrix (peptide dye counts per channel) and
    dyetrack-peptide list generated in the simulator into a radiometry matrix
    (observed intensity per channel). Models microscopy processes and dye-dye
    interactions such as FRET.

    Args
    ----
    dytmat_cube: np.ndarray
        Input dyetrack matrix. 3D representation of observed dyetracks whose
        dimensions are:
            0: n_dyetracks
            1: n_channels (observation wavelengths)
            2: n_cycles (number of chemical cycles that perturb the peptides).
        These dyetracks will be transformed into radiometry matrices according
        to the statistical information in dytpeps.
    dytpeps: np.ndarray
        Dyetrack-peptide matrix. Correlates dyetracks in dytmat_cube with peptides
        generated by the input to the simulator and counts the number of times each
        peptide generated each dyetrack. Columns are:
            0: Dyetrack (dim 0 in dytmat_cube)
            1: Peptide index (referencing peptides supplied to dytmat generation)
            2: Counts of dyetracks generated by each peptide
    sim_params: sim_v3_params.SimV3Params
        Contains error parameters and FRET interaction info for radiometry sims.
    n_samples: int
        Number of times each peptide will be sampled. Specified in sim_params
        for n_samples_train and n_samples_test. Must be equal to the number of
        counts generated for each peptide in dytpeps.
        TODO: Replace with counts observed from dytpeps.
    out_radmat: np.ndarray
        Predefined array for radmat results. Must be 3D and have dimensions
            0: n_peptides * n_samples
            1: n_channels
            2: n_cycles
    out_row_ks: np.ndarray
        Predefined array of length n_peptides * n_samples containing the row_k
        row intensity modification for each radmat row. (This is a model of
        experimental error from earlier methods and may be deprecated at some
        future point.)
    out_dyt_iz: np.ndarray
        Predefined array of length n_peptides * n_samples containing the index of
        the dyetrack that generated each radiometry row.
    out_pep_iz: np.ndarray
        Predefined array of length n_peptides * n_samples containing the index of
        the peptide that generated each radiometry row.
    seed: Optiona[int] = None
        Random number generator seed.
    skip_resample: bool = True
        It was once true that normally the dytpeps were resampled based on their
        proportion in each peptide prior to generation of the radiometry matrix.
        But the default behavior of the dytsim is to generate the requested n_samples
        unless the peptide is completely dark, so resampling is not really necessary.
        If False, then do resample anyway.
    gain_sigma_scaling_linear: bool = False
        Should multicount signal sigma be scaling linearly with dyecount
        or as sqrt()
    """

    assert len(dytmat_cube.shape) == 3
    assert len(dytpeps.shape) == 2

    # TODO: Remove when Flyte fixes Optional[int] behavior
    # Flyte internally converts Optional[int] types to float
    # so the RNG seed has to be returned to int
    if seed is not None:
        seed = int(seed)

    rng = np.random.default_rng(seed)

    sorted_dytpeps = dyt_helpers.sort_dytpeps(dytpeps)

    if skip_resample:
        resampled_dytpeps = sorted_dytpeps
    else:
        resampled_dytpeps = radsim.c_pep_sample(
            sorted_dytpeps, n_samples, seed=seed, progress=None
        )

    # set up rng shape arrays
    channel_priors_df = sim_params.channel_priors_df

    gain_mu = np.stack(
        [
            mu * np.ones((n_samples, dytmat_cube.shape[2]))
            for mu in channel_priors_df.gain_mu
        ],
        axis=1,
    )
    gain_sigma = np.stack(
        [
            sigma * np.ones((n_samples, dytmat_cube.shape[2]))
            for sigma in channel_priors_df.gain_sigma
        ],
        axis=1,
    )
    gain_var = np.stack(
        [
            sigma * sigma * np.ones((n_samples, dytmat_cube.shape[2]))
            for sigma in channel_priors_df.gain_sigma
        ],
        axis=1,
    )
    bg_mu = np.stack(
        [
            mu * np.ones((n_samples, dytmat_cube.shape[2]))
            for mu in channel_priors_df.bg_mu
        ],
        axis=1,
    )
    bg_sigma = np.stack(
        [
            sigma * np.ones((n_samples, dytmat_cube.shape[2]))
            for sigma in channel_priors_df.bg_sigma
        ],
        axis=1,
    )
    bg_var = np.stack(
        [
            sigma * sigma * np.ones((n_samples, dytmat_cube.shape[2]))
            for sigma in channel_priors_df.bg_sigma
        ],
        axis=1,
    )

    row_k_mu = np.ones(gain_mu.shape[0])
    row_k_sigma = np.ones(gain_mu.shape[0]) * sim_params.seq_params.row_k_sigma

    # generate dyetracks
    collector = {}
    accum_idx = 0
    # skip resampled_dytpeps row 0
    # iterating over the dytpep matrix
    for dytpep_idx in range(1, resampled_dytpeps.shape[0]):

        dytmat_idx, pep_idx, pep_count = resampled_dytpeps[dytpep_idx, :]

        if pep_idx not in collector:
            collector[pep_idx] = []

        for _ in range(pep_count):
            # apply FRET modifications
            this_dyetrack = fret_mod(
                dytmat_cube[dytmat_idx, :, :], sim_params.frets, rng
            )
            collector[pep_idx].append(this_dyetrack)
            out_dyt_iz[accum_idx] = dytmat_idx
            out_pep_iz[accum_idx] = pep_idx
            accum_idx = accum_idx + 1

    # take the FRET modified dyetracks and apply noise perturbations on them
    for index, k in enumerate(collector):
        row_ks = rng.normal(row_k_mu, row_k_sigma)

        if gain_sigma_scaling_linear:
            # Method prior to 12 dec 2022 by DSW during FRET work
            # This is not the mathemetically "correct" way to sum normal distributions
            # but emperically it fits our data better. See
            # https://docs.google.com/presentation/d/1yhBM-OwmbKh8D9ajYb9gPaz_r-ezoRR1IyK31eGi5Do/
            illum = np.stack(collector[k]) * rng.normal(gain_mu, gain_sigma)
            +rng.normal(bg_mu, bg_sigma)
        else:
            # Method as implemented in C versions of radiometry, and also by whatprot,
            # which does a "mathematically correct" convolution of normal distributions.
            mu = collector[k] * gain_mu + bg_mu
            sigma = np.sqrt(collector[k] * gain_var + bg_var)
            illum = rng.normal(mu, sigma)

        # np.newaxis helps broadcast
        out_radmat[(index * n_samples) : ((index + 1) * n_samples), :, :] = (
            row_ks[:, np.newaxis, np.newaxis] * illum
        )

        out_row_ks[(index * n_samples) : ((index + 1) * n_samples)] = row_ks


def sim_v3(
    sim_params: SimV3Params,
    prep_result: PrepV2Result,
    folder: Path,
    *,
    skip_generate_flus: bool = False,
    skip_resample: bool = True,
) -> SimV3Result:

    test_dytmat = None
    test_radmat = None
    test_pep_iz = None
    test_dyt_iz = None
    test_row_ks = None

    # TODO: DRY this code!

    # Training data
    #   * always includes decoys
    #   * may include radiometry

    n_channels = sim_params.n_channels
    n_cycles = sim_params.n_cycles
    gain_sigma_scaling_linear = sim_params.seq_params.gain_sigma_scaling_linear

    if sim_params.span_dyetracks:
        n_span_labels_list = sim_params.n_span_labels_list

        train_dytmat, train_dytpeps, train_pep_recalls = span_dytsim(
            n_channels, n_cycles, n_span_labels_list, sim_params.n_samples_train
        )

    else:
        (
            train_dytmat,
            train_dytpeps,
            train_pep_recalls,
        ) = prep_result.get_photobleaching()

        if train_dytmat is None:
            # This is a regular, non-photo-bleaching run
            pepseqs = prep_result.pepseqs__with_decoys()
            assert type(pepseqs) == pd.DataFrame  # (pep_i, aa, pep_off_in_pro)
            pcbs = sim_params.pcbs(
                pepseqs
            )  # (p)ep_i, (c)hannel_i, (b)right_probability

            train_dytmat, train_dytpeps, train_pep_recalls = _dytmat_sim(
                sim_params,
                pcbs,
                sim_params.n_samples_train,
            )

    n_dyts = train_dytmat.shape[0]

    assert train_dytmat.shape[1] == n_channels * n_cycles

    # dytpeps are a map between dyetracks and peptides with a count
    # Example:
    #   (2, 5, 110) => dyt_i=2 was generated by pep_i==5 110 times
    #   (2, 7, 50)  => dyt_i=2 was generated by pep_i==7 50 times
    assert train_dytpeps.shape[1] == 3

    if train_dytpeps.shape[0] > 0:
        # train_dytpeps can be empty if there's no chance of brightness
        assert dyt_helpers.n_dyts(train_dytpeps) == n_dyts

    # This n_peps is the number of unique peptides that produced non-dark
    # dye-tracks in _dytmay_sim() above.  It also includes the 0/null-peptide
    # row thus the -1 in n_samples_total below.
    n_peps = dyt_helpers.n_peps(train_dytpeps)
    n_samples_total = (
        n_peps - 1
    ) * sim_params.n_samples_train  # -1 to exclude the nul record

    # memory maps can not deal with empty files so in the weird
    # boundary case of no samples, give it one.
    n_samples_total = max(1, n_samples_total)

    train_radmat = ArrayResult(
        str(folder / "_train_radmat.arr"),
        shape=(n_samples_total, n_channels, n_cycles),
        dtype=RadType,
        mode="w+",
    )
    train_row_ks = ArrayResult(
        str(folder / "_train_row_ks.arr"),
        shape=(n_samples_total,),
        dtype=RowKType,
        mode="w+",
    )
    train_dyt_iz = ArrayResult(
        str(folder / "_train_dyt_iz.arr"),
        shape=(n_samples_total,),
        dtype=DytPepType,
        mode="w+",
    )
    train_pep_iz = ArrayResult(
        str(folder / "_train_pep_iz.arr"),
        shape=(n_samples_total,),
        dtype=DytPepType,
        mode="w+",
    )

    _radmat_sim(
        train_dytmat.reshape(
            (
                train_dytmat.shape[0],
                n_channels,
                n_cycles,
            )
        ),
        train_dytpeps,
        sim_params,
        sim_params.n_samples_train,
        out_radmat=train_radmat.arr(),
        out_row_ks=train_row_ks.arr(),
        out_dyt_iz=train_dyt_iz.arr(),
        out_pep_iz=train_pep_iz.arr(),
        seed=sim_params.seed,
        skip_resample=skip_resample,
        gain_sigma_scaling_linear=gain_sigma_scaling_linear,
    )

    # TODO: Split train and test functions
    # Test data
    #   * can optionally include decoys
    #   * always includes radiometry
    #   * may include dyetracks
    #   * skipped if is_survey
    # -----------------------------------------------------------------------
    if not sim_params.is_survey:

        if sim_params.span_dyetracks:
            test_dytmat = train_dytmat
            test_dytpeps = train_dytpeps
            test_dytpeps[:, -1] = (
                np.ones(test_dytpeps.shape[0]) * sim_params.n_samples_test
            )
            test_dytpeps[0, -1] = 0

        else:
            (
                test_dytmat,
                test_dytpeps,
                _,
            ) = prep_result.get_photobleaching()

            if test_dytmat is None:
                # This is a regular, non-photo-bleaching run
                if sim_params.test_decoys:
                    test_dytmat, test_dytpeps, _ = _dytmat_sim(
                        sim_params,
                        sim_params.pcbs(prep_result.pepseqs__with_decoys()),
                        sim_params.n_samples_test,
                    )
                else:
                    test_dytmat, test_dytpeps, _ = _dytmat_sim(
                        sim_params,
                        sim_params.pcbs(prep_result.pepseqs__no_decoys()),
                        sim_params.n_samples_test,
                    )

        n_peps = dyt_helpers.n_peps(test_dytpeps)
        n_samples_total = (
            n_peps - 1
        ) * sim_params.n_samples_test  # -1 to exclude nul record

        # See above about this max
        n_samples_total = max(1, n_samples_total)

        test_radmat = ArrayResult(
            str(folder / "_test_radmat.arr"),
            shape=(n_samples_total, n_channels, n_cycles),
            dtype=RadType,
            mode="w+",
        )
        test_row_ks = ArrayResult(
            str(folder / "_test_row_ks.arr"),
            shape=(n_samples_total,),
            dtype=RowKType,
            mode="w+",
        )
        test_dyt_iz = ArrayResult(
            str(folder / "_test_dyt_iz.arr"),
            shape=(n_samples_total,),
            dtype=DytPepType,
            mode="w+",
        )
        test_pep_iz = ArrayResult(
            str(folder / "_test_pep_iz.arr"),
            shape=(n_samples_total,),
            dtype=DytPepType,
            mode="w+",
        )

        _radmat_sim(
            test_dytmat.reshape(
                (
                    test_dytmat.shape[0],
                    n_channels,
                    n_cycles,
                )
            ),
            test_dytpeps,
            sim_params,
            sim_params.n_samples_test,
            out_radmat=test_radmat.arr(),
            out_row_ks=test_row_ks.arr(),
            out_dyt_iz=test_dyt_iz.arr(),
            out_pep_iz=test_pep_iz.arr(),
            seed=sim_params.seed,
            skip_resample=skip_resample,
            gain_sigma_scaling_linear=gain_sigma_scaling_linear,
        )

    sim_result = SimV3Result(
        params=sim_params,
        _flus=None,
    )
    sim_result.train_dytmat = train_dytmat
    sim_result.train_radmat = train_radmat
    sim_result.train_pep_recalls = train_pep_recalls
    sim_result.train_true_pep_iz = train_pep_iz
    sim_result.train_true_dye_iz = train_dyt_iz
    sim_result.train_dytpeps = train_dytpeps
    sim_result.train_true_row_ks = train_row_ks
    sim_result.test_dytmat = test_dytmat
    sim_result.test_radmat = test_radmat
    sim_result.test_true_pep_iz = test_pep_iz
    sim_result.test_true_dye_iz = test_dyt_iz
    sim_result.test_true_row_ks = test_row_ks

    sim_result.set_folder(folder)

    # HACK: Put this back in!
    if not skip_generate_flus:
        sim_result._generate_flu_info(prep_result)

    return sim_result


def sim_v3_photobleaching(
    n_cycles,
    n_count,
    n_samples,
    gain_mu,
    gain_sigma,
    bg_sigma,
    row_k_sigma,
    folder,
    seed: Optional[int] = None,
) -> SimV3Result:
    """
    Works on one channel only
    """

    prep_result = photobleaching_fixture(n_cycles, n_count, folder)

    sim_params = SimV3Params(
        markers=[
            Marker(
                aa="X",
                gain_mu=gain_mu,
                gain_sigma=gain_sigma,
                bg_sigma=bg_sigma,
                p_bleach=0.0,
                p_dud=0.0,
            )
        ],
        seq_params=SeqParams(
            n_edmans=(n_cycles - 1),
            p_edman_failure=0.0,
            p_detach=0.0,
            row_k_sigma=row_k_sigma,
        ),
        n_samples_train=n_samples,
        n_samples_test=1,
    )

    sim_result = sim_v3(
        sim_params=sim_params,
        prep_result=prep_result,
        folder=folder,
        skip_generate_flus=True,
    )
    return sim_result


def gen_dyetrack_span_one_channel(n_cycles: int, n_span_labels: int) -> np.ndarray:
    """Generate single-channel dytmat spanning the space defined by n_cycles and n_span_labels.

    Dyetracks are monotonically decreasing integer sequences of length n_cycles with maximum value
    n_span_labels. For n_cycles = 4 and  n_span_labels = 2, the dyetracks spanning the space are:
    0000,
    1000, 1100, 1110, 1111,
    2000, 2100, 2110, 2111,
    2200, 2210, 2211,
    2220, 2221,
    2222

    Similar to prep_v2_worker.triangle_dytmat but a little faster, and always allows multidrops
    (decreases of >1 dye from one cycle to the next).

    Args
    ----
    n_cycles : int
        Total number of cycles in the dyetrack.
    n_span_labels: int
        Maximum number of labels observable in the dyetrack.

    Returns
    -------
    A single channel dytmat with dyetracks spanning the (n_cycles, n_span_labels ) space.

    """
    assert type(n_span_labels) == int

    blank = np.zeros(n_cycles, dtype=np.uint32)
    work_items = [blank]
    tracks = []
    for _ in range(n_span_labels):
        for item in work_items:
            for count in range(n_cycles + 1):
                dyt = item.copy()
                dyt[:count] += 1
                tracks.append(dyt)
        # make a copy
        work_items = [x for x in np.unique(tracks, axis=0)]
    return np.vstack(np.unique(tracks, axis=0))


def gen_dyetrack_span(n_cycles: int, n_span_labels_list: list[int]) -> list[np.ndarray]:
    """
    Build a list of single-channel spanning-dyetracks based on number of labels in each channel
    """
    assert type(n_span_labels_list) == list
    tracks_per_channel = []
    for n in n_span_labels_list:
        tracks_per_channel.append(
            gen_dyetrack_span_one_channel(n_cycles=n_cycles, n_span_labels=n)
        )
    return tracks_per_channel


def span_dytsim(
    n_channels: int, n_cycles: int, n_span_labels_list: list[int], n_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate spanning dyetracks across multiple channels, mimicking the output of _dytmat_sim.

    Generates a Cartesian product across the result of gen_dyetrack_span. Setting n_channels > 2 and
    n_span_labels > 2 can lead to very large dytmats that take a long time to compute, so be careful.

    Args
    ----
    n_channels : int
        Number of channels in the experiment.
    n_cycles : int
        Total number of cycles in each dyetrack.
    n_span_labels_list: list[int]
        Maximum number of labels observable in each dyetrack, per channel
    n_samples : int
        Number of samples to add to the synthetic dytmat matrix. Affects number of downstream samples
        for classification but does not add any computation in this function.

    Returns
    -------
    Three element tuple mimicking the output of _dytmat_sim:
        - a dytmat matrix containing dyetracks spanning the (n_channels, n_cycles, n_span_labels ) space
        - a dytpep matrix containing a synthetic dyetrack-peptide matrix for consumption by sim_v3_span
        - a train_pep_recalls matrix consisting of ones for each dyetrack except for the null dyetrack
    """
    assert len(n_span_labels_list) == n_channels
    tracks = gen_dyetrack_span(n_cycles, n_span_labels_list)
    span_dytmat = np.vstack(list(map(np.concatenate, itertools.product(*tracks))))
    span_dytpeps = np.column_stack(
        [
            range(span_dytmat.shape[0]),
            range(span_dytmat.shape[0]),
            n_samples * np.ones(span_dytmat.shape[0]),
        ]
    ).astype(np.uint32)
    # set span_dytpeps 0 row count to 0
    span_train_pep_recalls = np.ones(span_dytpeps.shape[0])
    if span_dytpeps[0, 0] == 0 and np.all(span_dytmat[0, :] == 0):
        span_dytpeps[0, 2] = 0
        span_train_pep_recalls[0] = 0.0
    return span_dytmat, span_dytpeps, span_train_pep_recalls
