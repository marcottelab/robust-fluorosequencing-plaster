# utilities for dealing with whatprot

# TODO: I don't know that whatprot-specific helper stuff belongs in tools.  It probably
# belongs in plaster.run.whatprot_v1


import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()

import plaster.genv2.generators.sim as simgen

# TODO: these params originated in reporting and some are still set by humans
# while looking at data in sigproc-related reports, but it feels strange to be
# pulling params from "reporting" into the run pipeline.  See comments in
# report_params_filtering.py.
from plaster.reports.helpers.report_params import FilterParams
from plaster.reports.helpers.report_params_filtering import (
    get_classify_default_filtering_df,
)
from plaster.run.base_result import ArrayResult
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult, PrepV2Result
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2ResultDC
from plaster.run.sim_v3.sim_v3_params import SimV3Params
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult, SimV3Result, gen_flus
from plaster.run.whatprot_v1.whatprot_v1_params import WhatprotV1Params

# ====================================================================================================


def with_name_suffix(p: Path, name_suffix: str, suffix: str = None) -> Path:
    """
    Returns filename with name_suffix added
        e.g. with_name_suffix( Path('/some/path/file.csv'), '_newstuff', 'xyz' )
        e.g. /some/path/file.csv => /some/path/file_newstuff.xyz
    """
    if not suffix:
        suffix = p.suffix
    return p.with_name(p.stem + name_suffix).with_suffix(suffix)


# ====================================================================================================
# init


def whatprot_init() -> Path:
    """
    Verifies that whatprot is built and installed at /whatprot, and adds
    the whatprot python folder to os.path so that programs can import whatprot
    python modules and functions.

    Returns:
        Path to whatprot executable for running whatprot
    """

    whatprot_root = Path("/whatprot")
    whatprot_python = whatprot_root / "python"
    whatprot_exe = whatprot_root / "cc_code/bin/release/whatprot"

    if not all(
        [whatprot_root.exists(), whatprot_python.exists(), whatprot_exe.exists()]
    ):
        logger.error(
            "You must install and build whatprot at /whatprot (or create a symlink from there)"
        )
        raise FileNotFoundError

    if whatprot_python not in sys.path:
        sys.path.append(str(whatprot_python))
    try:
        from convert_radiometries import convert_radiometries
    except:
        logger.error(
            "Failed to add whatprot python folder to sys.path",
            whaprot_python=whatprot_python,
        )
        raise

    return whatprot_exe


# I like the idea of this getting initialized exactly once for various reasons.
# But it presents a problem to integration testing, presumably because it is running
# on an instance without whatprot installed, because the PR that includes this work
# is what added whatprot to the plaster image.  Or maybe it's not even running in
# a plaster-image container - in which case whatprot would probably never be
# installed and any module that causes this to be included will fail when it tries
# to import convert_radiometries in the fn above.  So for now I'll revert to requiring
# callers to call whatprot_init() to get the whatprot_exe path.
#
# whatprot_exe = whatprot_init()

# ====================================================================================================
# write peps


def write_peps_csv(
    path: os.PathLike, prep_result: PrepV2Result, real_peptides: Optional[int] = None
):
    """
    Writes the peptides file read by whatprot for generating dye-seqs.  This file
    is also used to know which proteins the peptides come from, and this information
    can be used by whatprot PR calculation for protein calls.

    We can take advantage of this for experiments with already-digested peptides and
    decoys (e.g. Marvel) to assign actual peptides to protein 0 and all others to
    protein 1, allowing us to make some precision statements about sigproc data when we
    know only a certain set of proteins to be present.

    If real_peptides is not None, it can specifiy the count of real peptides, and those
    peptides will be assigned to protein 0, and the rest to protein 1.
    """
    df = prep_result.pros__peps__pepstrs()
    with open(path, "w", newline="") as f:  # note newline required for csv_writer
        csv_writer = csv.writer(f)
        # skip null-peptide, and make pro_i 0-based like whatprot
        for pro_i, pepstr in zip(df["pro_i"][1:] - 1, df["seqstr"][1:]):
            if real_peptides is not None:
                pro_i = 0 if pro_i < real_peptides else 1
            # f.write(f"{pepstr},{pro_i}\n")
            csv_writer.writerow([pepstr, pro_i])

    # Write a "real" file with only the real peptides if real_peptides was spec'd
    if real_peptides is not None:
        p = Path(path)
        new_path = p.with_name(p.stem + "_real").with_suffix(p.suffix)
        with open(
            new_path, "w", newline=""
        ) as f:  # note newline required for csv_writer
            csv_writer = csv.writer(f)
            # skip null-peptide, and make pro_i 0-based like whatprot
            for pepstr in df["seqstr"][1 : real_peptides + 1]:
                # f.write(f"{pepstr},0\n")
                csv_writer.writerow([pepstr, 0])


# ====================================================================================================
# write dyeseqs - two different implementations, one using whatprot code, one using Erisyon


def write_dyeseqs_via_whatprot(
    seqs_path: os.PathLike,
    sim_result: SimV3Result,
    peps_path: os.PathLike,
):
    """
    Writes the whatprot-format dyeseq file by calling the whatprot-provided
    dye_seqs_from_peptides function.
    """

    def labels_from_erisyon_markers(markers: list[simgen.Marker]) -> list:
        labels = []
        for m in markers:
            labels.append(m.aa)
        return labels

    labels = labels_from_erisyon_markers(sim_result.params.markers)

    # Write the dyeseqs.tsv file using whatprot code
    whatprot_init()
    from dye_seqs_from_peptides import dye_seqs_from_peptides

    # Write both versions of this file -- one that gives only the
    # first peptide for a dyeseq, and one that gives all of them.
    # The latter is only used for whatprot's pr_curve routines.
    seqs_all_path = seqs_path.with_name(seqs_path.stem + "_all").with_suffix(
        seqs_path.suffix
    )
    dye_seqs_from_peptides(peps_path, labels, seqs_path, "first")
    dye_seqs_from_peptides(peps_path, labels, seqs_all_path, "all")


def whatprot_dyeseqs(
    prep_result: PrepV2Result, sim_result: SimV3Result
) -> pd.DataFrame:
    """Generates whatprot-formatted dyeseqs from Erisyon prep and sim data.

    Returns:
        DataFrame: containing (pep_i,dyeseq) Note: pep_i is 1-based, as Erisyon.
                   Empty dyeseqs are dropped, but dupes remain.
    """
    # Look at whatprot dyeseqs file
    #
    # Notes:
    #
    #  - unlabeled peps are not included
    #  - duplicate dyeseqs are not included
    #  - only the first peptide that creates a dyeseq is referenced
    #  - dyeseqs are possibly as long as peptide (n_edmans doesn't matter)
    #  - dyeseqs always stop at the last labeled position

    # We can generate the same kind of dyeseqs from the erisyon data by setting
    # the n_cycles param to some large value, at least as large as the longest
    # peptide and stripping some trailing info.
    #
    # We drop empty dyeseqs, but retain dupes and 1-based pep_i indices.
    # To drop dupes and get a lookup for 0-based canonical pep_i, use the
    # ...canonical() fn below.

    ps_df = prep_result.pepseqs()
    max_pep_len = ps_df.pep_i.value_counts().max()

    flus, counts = gen_flus(
        ps_df,
        sim_result.params.ch_by_aa,
        max_pep_len,
        has_header=True,
    )

    # format erisyon flu as whatprot dyeseq (get rid of remainder info and trailing .)
    # '....0..0.....0....     ;0,0' becomes => '....0..0.....0'
    flus = [f.rstrip(";0,").rstrip(". ") for f in flus]

    df = pd.DataFrame(data=flus, columns=["dyeseq"])
    df["pep_i"] = list(range(len(flus)))
    df = df[df["dyeseq"].astype(bool)]
    return df[["pep_i", "dyeseq"]]


def whatprot_dyeseqs_canonical(
    prep_result: PrepV2Result, sim_result: SimV3Result
) -> tuple[pd.DataFrame, dict]:
    """Generates whatprot-formatted dyeseqs from Erisyon prep and sim data,
    and removes duplicate dyeseqs, using 0-based pep_i indices like whatprot.
    A lookup table is created that maps all 'dupe' pep_i to their canonical
    'proxy' pep_i to facilitate working with whatprot in which only the
    first pep_i that generates a given dyeseq is kept.

    Returns:
         DataFrame: (dyeseq,pep_i,pep_count) (just like whatprot dyeseq file)
         dict: pep_i => canonical pep_i
               e.g. if peps 1,5,8,9 all generate the same dyeseq, then only
                    pep 1 will appear in the DataFrame, the pep_count for that
                    row will be 4, and peps 5,8,9 will each map to 1 in the
                    lookup table.
    """

    df = whatprot_dyeseqs(prep_result, sim_result)
    df["pep_i"] = df["pep_i"] - 1  # to 0-based indices

    # Create the lookup table (dict) allowing pep_i that will soon
    # be dropped to map to a 'canonical' pep_i used by whatprot
    #
    def build_lookup(peps: pd.Series) -> dict:
        lookup = {}
        for i, pep_list in peps.items():
            for p in pep_list:
                lookup[p] = pep_list[0]
        return lookup

    by_dyeseq = df.groupby("dyeseq")
    df_peplist = by_dyeseq["pep_i"].agg(list).reset_index()
    lookup = build_lookup(df_peplist["pep_i"])

    # Count number of peps for each dyeseq, and drop all but first pep,
    # returning df with cols ordered like whatprot dyeseq file.
    df["pep_count"] = by_dyeseq.transform("size")
    df = df.drop_duplicates(subset="dyeseq", keep="first")[
        ["dyeseq", "pep_count", "pep_i"]
    ]

    return df, lookup


# ====================================================================================================
# seq_params


def write_seq_params_json(path: os.PathLike, sim_config: SimV3Params):
    """Writes a json file containing sequenceing params in the
    whatprot required format"""

    from munch import Munch

    from plaster.tools.utils.utils import json_save

    ch_models = []
    for m in sim_config.markers:
        ch_models.append(
            Munch(
                p_bleach=m.p_bleach,
                p_dud=m.p_dud,
                bg_sig=m.bg_sigma,
                mu=m.gain_mu,
                sig=m.gain_sigma,
            )
        )

    params = Munch(
        p_edman_failure=sim_config.seq_params.p_edman_failure,
        p_detach=sim_config.seq_params.p_detach,
        p_initial_block=sim_config.seq_params.p_initial_block,
        p_cyclic_block=sim_config.seq_params.p_cyclic_block,
        channel_models=ch_models,
    )

    json_save(path, params)


# ====================================================================================================
# radmat


def write_whatprot_format_radmat(
    path: os.PathLike,
    signal: Union[np.ndarray, ArrayResult],
    params: SimV3Params,
    filter_df: Optional[pd.DataFrame] = None,
    true_peps_df: Optional[pd.DataFrame] = None,
):
    """
    Writes a radiometry matrix as passed in 'signal' to either a .tsv file, or a .npy file, depending on the suffix of the
    filename passed in argument 'path'.

    If filter_df is not None, this is used to filter the rows of the radiometry matrix.

    If true_peps_df is not None, then a "_true_peps.csv" file is written adjacent to the radiometry matrix indicating the
    peptide indices from which each row of the radiometry came from.

    Note: the logic for the tsv format was adapted from code written by AMB.  There is a lot of logic there relating to
    titles of columns that ends up not mattering.  And, you can also just save the .npy format directly.

    Either way, this radiometry matrix still will need to be further converted by the whatprot "convert_radiometries" code,
    see further below.
    """

    tsv_format = path.suffix == ".tsv"

    if tsv_format:
        # Adapated from Angela's code
        channel_dfs = []
        wp_radmat_df = pd.DataFrame()
        cycle_types = ["_P", "_M", "_E"]
        cycle_counts = [
            params.seq_params.n_pres,
            params.seq_params.n_mocks,
            params.seq_params.n_edmans,
        ]
        for ch in range(params.n_channels):
            col_titles = []
            for cycle_type, cycle_count in zip(cycle_types, cycle_counts):
                for cy in range(cycle_count):
                    col_titles.append(f"Ch{ch+1}{cycle_type}{cy+1}")
            channel_dfs.append(pd.DataFrame(data=signal[:, ch, :], columns=col_titles))

        wp_radmat_df = pd.concat(channel_dfs, axis=1)

        # Now filter it and write it.
        if filter_df is not None:
            wp_radmat_df = wp_radmat_df.loc[filter_df.pass_all, :]
        wp_radmat_df.to_csv(path, sep="\t", header=False, index=False)

    else:
        assert path.suffix == ".npy"
        assert (
            filter_df is None
        )  # at present, I'm only saving the .npy file from simulations, so no filtering
        np.save(path.with_suffix(".npy"), signal.arr(), allow_pickle=False)

    # If we were passed a true_peps_df, filter it and write it alongside the radmat.
    if true_peps_df is not None:
        if filter_df is not None:
            # as of this writing, this will never happen bc we don't filter simulated radiometry
            true_peps_df = true_peps_df[filter_df.pass_all, :]
        with open(str(path.with_suffix("")) + "_true_peps.csv", "w") as file:
            file.write(f"{len(true_peps_df.index)}\n")
            true_peps_df.to_csv(file, sep="\t", header=None, index=False)


# ====================================================================================================
# high-level prepare input used by flyte tasks


def shlex_cmdline_escaping(cmd_list: list[str]) -> list[str]:
    """
    Use shlex to provide quoting/escaping against injection attacks.
    """
    import shlex

    cmd = shlex.join(cmd_list)  # get the quoting and escaping provided by shlex
    cmd_list = shlex.split(cmd)
    return cmd_list


def prepare_whatprot_train_input(
    params: WhatprotV1Params,
    prep_result: PrepV2Result,
    sim_result: SimV3Result,
    whatprot_path: Path,
):
    """
    Given a folder to an erisyon sim or vfs, create all of the input files in order to train the whatprot
    classifier on this sim.  Whatprot doesn't really have a separate "training" phase, but we can prepare
    these files so that they are later passed to whatprot in conjunction with radiometry from a different sim
    or sigproc.

    sim_job_path: path to Erisyon (sim or vfs) job where prep and sim subfolders can be found
    whatprot_path: path where the whatprot input files should be written.

    Returns:
        tuple(str) : a list of filenames that were written
    """

    whatprot_path.mkdir(exist_ok=True)

    # Write the peps from our simulation
    wp_peps = whatprot_path / "erisyon_peps.csv"
    write_peps_csv(wp_peps, prep_result)

    # Write dye-seqs file: do this two ways - one with whatprot code and one with Erisyon code
    #
    wp_dye_seqs = whatprot_path / "whatprot_dyeseq.tsv"
    write_dyeseqs_via_whatprot(wp_dye_seqs, sim_result, wp_peps)

    # Write the erisyon_dyeseqs.tsv file based on erisyon-input and logic, for comparison.
    # Note peps_canonical, a lookup table we'll use later.
    erisyon_dye_seqs = whatprot_path / "erisyon_dyeseq.tsv"
    df, peps_canonical = whatprot_dyeseqs_canonical(prep_result, sim_result)
    with open(erisyon_dye_seqs, "w") as file:
        file.write(f"{sim_result.params.n_channels}\n")
        file.write(f"{len(df.index)}\n")
        df.to_csv(file, sep="\t", header=False, index=False)

    # Create seq_params.json, and create dye-tracks file, which requires a whatprot simulation.

    wp_dye_tracks = whatprot_path / "whatprot_dyetrk.tsv"
    wp_seq_params = whatprot_path / "erisyon_seq_params.json"
    write_seq_params_json(wp_seq_params, sim_result.params)

    whatprot_exe = whatprot_init()

    # note: sim_result.params.n_cycles is n_pres + n_mocks + n_edmans, which could differ from the -t docs above.
    # fmt: off
    cmd_list = [
        whatprot_exe,
        'simulate', 'dt',
        '-t', sim_result.params.n_cycles,
        '-g', params.wp_numgenerate,
        '-P', wp_seq_params,
        '-S', wp_dye_seqs,
        '-T', wp_dye_tracks
    ]
    # fmt: on
    cmd_list = [
        str(c) for c in cmd_list
    ]  # convert Path objects, numeric literals, etc.

    # Use shlex to join and then re-split this command to benefit from its quoting/escaping
    # though this may be overkill here -- the filenames have been constructed in our code
    # and the params are all numerical and have been validated via dataclass schema.

    cmd_list = shlex_cmdline_escaping(cmd_list)
    subprocess.run(cmd_list)

    names = [
        wp_peps,  # the peptides from the simulation
        wp_dye_seqs,  # the dye-seqs as computed by whatprot code
        erisyon_dye_seqs,  # the dye-seqs as computed by erisyon code
        wp_seq_params,  # the seq-params from the erisyon sim used by whatprot sim
        wp_dye_tracks,  # the dye-tracks simulated by whatprot
    ]
    return tuple(map(str, names)), peps_canonical


# ====================================================================================================


def prepare_whatprot_classify_sim_input(
    sim_result: SimV3Result, whatprot_path: os.PathLike, peps_canonical: dict
):
    # Preprocess simulated radmat for consumption by whatprot.

    signal = sim_result.test_radmat
    filter_df = None
    wp_filtered_sim_radmat = whatprot_path / "erisyon_sim_radmat.npy"

    # for simulated data only, we can also write the "true" peptide indices
    # corresponding to this radiometry.  A complication is that whatprot drops all
    # peps with duplicate dyeseq, and retains only the initial pep_i along with
    # a count that is used to adjust the PR for that pep.  To use whatprot PR
    # routines for comparison, it's useful to produce a true_peps file that collapses
    # duplicates in this way, which we can do with a lookup table we've
    # previously computed when creating the dyeseqs file in an earlier cell.
    pep_i_0based = sim_result.test_true_pep_iz.arr() - 1
    pep_i_whatprot = [
        peps_canonical[p] for p in pep_i_0based
    ]  # this line simply maps peps with same dyeseq to a single pep_i like whatprot does
    true_peps_df = pd.DataFrame(pep_i_whatprot, copy=False, columns=["pep_i"])
    wp_true_peps = str(wp_filtered_sim_radmat.with_suffix("")) + "_true_peps.csv"

    write_whatprot_format_radmat(
        wp_filtered_sim_radmat, signal, sim_result.params, filter_df, true_peps_df
    )

    # - num_channels is the number of colors of fluorophore.
    # - num_mocks is the number of mocks in the input file. This many cycles will be removed
    #   from the beginning of every read.
    # - num_cycles is the number of Edmans plus one (i.e., including the 'pre-edman' cycle
    #   which you should always sequence with).

    converted_name = wp_filtered_sim_radmat.with_suffix("").name + "_converted.tsv"
    wp_converted_sim_radmat = wp_filtered_sim_radmat.with_name(converted_name)

    # This module is in whatprot and we dynamically add that path in whatprot_init()
    whatprot_init()
    from convert_radiometries import convert_radiometries

    convert_radiometries(
        sim_result.params.n_channels,
        sim_result.params.seq_params.n_mocks,
        sim_result.params.seq_params.n_pres
        + sim_result.params.seq_params.n_edmans,  # does whatprot need n_pres to be 1?
        str(wp_filtered_sim_radmat),
        wp_converted_sim_radmat,
    )

    names = [
        wp_filtered_sim_radmat,  # the sim radmat out of erisyon sim
        wp_converted_sim_radmat,  # the sim radmat as converted by whatprot
        wp_true_peps,  # the true pep_i per row of the sim radmat
    ]
    return tuple(map(str, names))


# ====================================================================================================


def prepare_whatprot_classify_sigproc_input(
    ims_import_result: ImsImportResult,
    sigproc_result: SigprocV2ResultDC,
    filter_params: FilterParams,
    filter_reject_thresh_all_cycles: bool,
    whatprot_path: os.PathLike,
    sim_params: SimV3Params,
):

    # TODO: convert this either make use of the sigproc_result passed, or get a
    # jobs_folder name from this so the code can be used as is.

    # pre-process and filter sigproc data for consumption by whatprot.

    wp_converted_sigproc_radmat = wp_filtered_sigproc_radmat = ""
    true_peps_df = None

    signal = sigproc_result.sig()
    logger.info("signal", shape=signal.shape)

    filter_df = get_classify_default_filtering_df(
        sigproc=sigproc_result,
        ims_import=ims_import_result,
        params=filter_params,
        all_cycles=filter_reject_thresh_all_cycles,
    )

    logger.info(
        "fitering df",
        shape=filter_df.shape,
        pass_all=filter_df.pass_all.sum(),
    )

    wp_filtered_sigproc_radmat = whatprot_path / "erisyon_sigproc_radmat.tsv"
    write_whatprot_format_radmat(
        wp_filtered_sigproc_radmat,
        signal,
        sim_params,
        filter_df,
        true_peps_df,
    )

    converted_name = wp_filtered_sigproc_radmat.with_suffix("").name + "_converted.tsv"
    wp_converted_sigproc_radmat = wp_filtered_sigproc_radmat.with_name(converted_name)

    # This module is in whatprot and we dynamically add that path in whatprot_init()
    whatprot_init()
    from convert_radiometries import convert_radiometries

    convert_radiometries(
        sim_params.n_channels,
        sim_params.seq_params.n_mocks,
        sim_params.seq_params.n_pres
        + sim_params.seq_params.n_edmans,  # does whatprot need n_pres to be 1?
        str(wp_filtered_sigproc_radmat),
        wp_converted_sigproc_radmat,
    )

    names = [
        wp_filtered_sigproc_radmat,  # the filtered sigproc radmat out of erisyon, if passed
        wp_converted_sigproc_radmat,  # the sigproc radmat, as converted by whatprot
    ]
    return tuple(map(str, names))


# ====================================================================================================
# Checks that all whatprot input files exist, adds a couple outfile paths, and returns filenames


def find_whatprot_files(
    whatprot_path: os.PathLike,
):
    """
    Look for canonical whatprot input and output filesnames in whatprot_path and return a dict
    which indicates if the file is present or not.
    """

    wp_peps = whatprot_path / "erisyon_peps.csv"
    wp_dye_seqs = whatprot_path / "whatprot_dyeseq.tsv"
    erisyon_dye_seqs = whatprot_path / "erisyon_dyeseq.tsv"
    wp_dye_tracks = whatprot_path / "whatprot_dyetrk.tsv"
    wp_seq_params = whatprot_path / "erisyon_seq_params.json"
    wp_filtered_sim_radmat = whatprot_path / "erisyon_sim_radmat.npy"
    wp_converted_sim_radmat = with_name_suffix(
        wp_filtered_sim_radmat, "_converted", ".tsv"
    )
    wp_true_peps = with_name_suffix(wp_filtered_sim_radmat, "_true_peps", ".csv")
    wp_filtered_sigproc_radmat = whatprot_path / "erisyon_sigproc_radmat.tsv"
    wp_converted_sigproc_radmat = with_name_suffix(
        wp_filtered_sigproc_radmat, "_converted"
    )

    # outputs - may not be present
    wp_sim_predictions = with_name_suffix(
        wp_converted_sim_radmat, "_predictions", ".csv"
    )
    wp_sigproc_predictions = with_name_suffix(
        wp_converted_sigproc_radmat, "_predictions", ".csv"
    )

    names = [
        wp_peps,  # the peptides from the simulation
        wp_dye_seqs,  # the dye-seqs as computed by whatprot code
        erisyon_dye_seqs,  # the dye-seqs as computed by erisyon code
        wp_seq_params,  # the seq-params from the erisyon sim used by whatprot sim
        wp_dye_tracks,  # the dye-tracks simulated by whatprot
        wp_filtered_sim_radmat,  # the sim radmat out of erisyon sim
        wp_converted_sim_radmat,  # the sim radmat as converted by whatprot
        wp_true_peps,  # the true pep_i per row of the sim radmat
        wp_filtered_sigproc_radmat,  # the filtered sigproc radmat out of erisyon, if passed
        wp_converted_sigproc_radmat,  # the sigproc radmat, as converted by whatprot
        wp_sim_predictions,  # the whatprot predictions of the sim radmat
        wp_sigproc_predictions,  # the whatprot predictions of the sigproc radmat
    ]

    files_dict = {}
    for p in names:
        files_dict[str(p)] = p.exists()

    return files_dict


# ====================================================================================================


def gather_whatprot_filenames_train_test(
    train_folder: str, test_folder: str, whatprot_subfolder="whatprot"
):
    # Find all of the whatprot input files we think should be in place for both the test and training data.
    # This is used by notebooks for flexibility in mixing and matching training/testing sims and sigproc
    # results, but the simpler fn below is more specific and this one will be deprecated soon.
    key_names = [
        "wp_peps",
        "wp_dye_seqs",
        "erisyon_dye_seqs",
        "wp_seq_params",
        "wp_dye_tracks",
        "wp_filtered_sim_radmat",
        "wp_converted_sim_radmat",
        "wp_true_peps",
        "wp_filtered_sigproc_radmat",
        "wp_converted_sigproc_radmat",
        "wp_sim_predictions",
        "wp_sigproc_predictions",
    ]

    folders = [train_folder, test_folder]
    key_suffix = ["_train", "_test"]
    wp_files_dict = {}
    for f, ks in zip(folders, key_suffix):
        files_dict = find_whatprot_files(f / whatprot_subfolder)
        wp_files_dict.update(
            {
                f"{k}{ks}": {"name": fn, "exists": exist}
                for k, fn, exist in zip(
                    key_names, files_dict.keys(), files_dict.values()
                )
            }
        )

    return wp_files_dict


# ====================================================================================================


def gather_whatprot_filenames_test(folder: os.PathLike):
    # Find all of the whatprot input files we think should be in place for a whatprot
    # run that is trained on a sim and classifies either a sim or a sigproc.  The key_names
    # here have suffixes that indicate how the file is used: _train files come from the
    # training simulation, and _test files are input and output to/from classification.
    # All of these will be stored in a whatprot_subfolder (defaults to 'whatprot') of the
    # test data folder, whether that is a sim or a sigproc.
    key_names = [
        "wp_peps_train",
        "wp_dye_seqs_train",
        "erisyon_dye_seqs_train",
        "wp_seq_params_train",
        "wp_dye_tracks_train",
        "wp_filtered_sim_radmat_test",
        "wp_converted_sim_radmat_test",
        "wp_true_peps_test",
        "wp_filtered_sigproc_radmat_test",
        "wp_converted_sigproc_radmat_test",
        "wp_sim_predictions_test",
        "wp_sigproc_predictions_test",
    ]

    files_dict = find_whatprot_files(folder)
    assert len(key_names) == len(files_dict)
    wp_files_dict = {
        k: {"name": fn, "exists": exist}
        for k, fn, exist in zip(key_names, files_dict.keys(), files_dict.values())
    }

    return wp_files_dict


# ====================================================================================================


def get_whatprot_fit_params(stdout_filename: Path, bounds: str = "") -> dict:
    """
    Parse the stdout file from whatprot to extract parameter values from a fit.
    If bounds is specified as 'upper' or 'lower' then find the parameter
    values corresponding to the upper or lower bounds of a confidence interval
    computed via bootstrapping.
    """

    if bounds not in ["", "lower", "upper"]:
        msg = f'Bad bounds params "{bounds}"'
        logger.error(msg)
        raise ValueError(msg)

    if not stdout_filename.exists():
        msg = f'File doesn\'t exist: "{stdout_filename}"'
        logger.error(msg)
        raise FileExistsError(msg)

    s = stdout_filename.read_text()
    number_pattern = "[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?"
    value = f"(.nan|{number_pattern})"
    params = [
        "p_edman_failure",
        "p_initial_block",
        "p_cyclic_block",
        "p_detach",
        "log(L)",
        "step",
        "time",
    ]
    ch_params = [
        "p_bleach_ch*",
        "p_dud_ch*",
    ]

    regex = [
        "Edman",
        "Initial",
        "Cyclic",
        "Detach",
        "log\(L",
        "step-",
        "run tim",
    ]

    ch_regex = [
        "Channel *.+Bleach",
        "Channel *.+Dud",
    ]

    params_dict = {}

    def re_add_param_value(p, r):
        e = f"{bounds}.*{r}\D+: {value}(,|\)|\s)"
        m = re.search(e, s)
        if m:
            params_dict[p] = float(m[1])

    for p, r in zip(params, regex):
        re_add_param_value(p, r)
    for p, r in zip(ch_params, ch_regex):
        for ch in range(4):
            p_c = p.replace("*", str(ch))
            r_c = r.replace("*", str(ch))
            re_add_param_value(p_c, r_c)

    return params_dict


# ====================================================================================================


def get_first_dyeseq(dyeseq_file: Path) -> str:
    """
    Look in the file passed, which is assumed to be a whatprot-formatted dyeseq
    tsv file, and get the first dyeseq present.

    e.g.:

    1
    1
    .0..0	1	0

    """

    lines = dyeseq_file.read_text().splitlines()
    return lines[2].split("\t")[0]
