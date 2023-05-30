"""
    runs whatprot
"""
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from flytekit import Resources, current_context, task

from plaster import env
from plaster.tools.utils.utils import json_load

env.configure_logging()
import structlog

logger = structlog.get_logger()

import plaster.tools.classifier_helpers.whatprot as whatprot_erisyon
from plaster.reports.helpers.report_params import FilterParams
from plaster.run.ims_import.ims_import_result import ImsImportFlyteResult
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2FlyteResult
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult
from plaster.run.whatprot_v1.whatprot_v1_params import WhatprotV1Params
from plaster.run.whatprot_v1.whatprot_v1_result import (
    WhatprotClassifyV1FlyteResult,
    WhatprotClassifyV1ResultDC,
    WhatprotFitBootstrapV1FlyteResult,
    WhatprotFitBootstrapV1ResultDC,
    WhatprotFitV1FlyteResult,
    WhatprotFitV1ResultDC,
    WhatprotPostClassifyV1FlyteResult,
    WhatprotPostClassifyV1ResultDC,
    WhatprotPreV1FlyteResult,
    WhatprotPreV1ResultDC,
)

# How to break up the tasks for whatprot?
#
# One line of thinking is with respect to what functionality you need in the containers
# per task.  If you were integrating some random tool that was already containerized,
# you might prefer to mod that as little as possible, and just write a python @task (or shell task)
# function for flyte that requires that container, and runs that executable on
# your pre-processed input.  It would spit out some files as output, and you'd write
# a flyte result class that packages that information.  You might then have
# a subsequent task (back in plaster domain) that does further post-processing or
# reporting on this output.
#
#
# 1. task to prepare input for tool - runs in plaster container
# 2. task to run tool based on input from (1)  - runs in tool container (no plaster functionality)
# 3. task to post-process tool results - runs in plaster container
#
# Another option is to always plan on integrating the tool with the plaster-container,
# and having the "tool-plaster" container.  This is what I've done as a start with
# whatprot - I build whatprot into the plaster container, and then can do everything
# with that container.
#
# The advantage of this approach is that it allows you to deploy this container anywhere
# and have access to the plaster+tool functionality.  This way will inevitably lead to
# plaster+tool1+tool2+tool3 etc.  So the container will grow larger.  But it means that
# these plaster+tools workflows can be executed outside of the flyte context -- e.g., from
# a notebook, from some command line, any way you choose via the single container.
#
# The multi-container approach works cleanly for flyte, but you will always need some
# infra that supports multi-container workflows wherever you'd like to do plaster+tool
# processing.
#
# whatprot is a very special case however; it is built by the lab our company came out of,
# so we are very close to it, and have some hopes it will be a significant tool in our
# pipeline.  So it's not just a random 3rd party tool we wish to use.


# Another idea for below is to remove much of the biz logic from these task fns
# and move the logic into the "worker" file, which is more agnostic to the workflow
# orchestration that is wrapping the work.  This is the pattern for most other job
# types, but that's largely an artifact of those worker functions existing beforehand,
# and the @task functions being created as wrappers to  those worker fns for
# integration into Flyte workflows.
#
# The @task is of course Flyte, but so are all the FlyteResult-based classes being
# passed around.  These are dataclass results wrapped in a simple class that binds
# them to a dir:FlyteDirectory and provides load/save functionality.


# This was previously a flyte task, but I needed to create a couple of "sizes" of this
# task and want to keep things as DRY as possible...
#
# This task currently uses both plaster and whatprot functionality to prepare input files for
# whatprot. It could be refactored to only do the plaster-based pre-processing here, and
# push other whatprot-based conversion to the next task, so that we'd only need a container
# with plaster here.
def whatprot_pre_flyte_task_worker(
    whatprot_params: WhatprotV1Params,
    prep_train_flyte_result: PrepV2FlyteResult,
    sim_train_flyte_result: SimV3FlyteResult,
    sim_test_flyte_result: Optional[SimV3FlyteResult] = None,
    ims_import_flyte_result: Optional[ImsImportFlyteResult] = None,
    sigproc_flyte_result: Optional[SigprocV2FlyteResult] = None,
) -> WhatprotPreV1FlyteResult:

    logger.info("whatprot load erisyon prep and sim")
    prep_result = prep_train_flyte_result.load_result()
    sim_result = sim_train_flyte_result.load_result()

    cwd = current_context().working_directory
    whatprot_folder = Path(cwd) / "whatprot"

    logger.info("whatprot prepare_whatprot_train_input (sim)")
    (
        wp_peps,  # the peptides from the simulation
        wp_dye_seqs,  # the dye-seqs as computed by whatprot code
        erisyon_dye_seqs,  # the dye-seqs as computed by erisyon code (indentical)
        wp_seq_params,  # the seq-params from the erisyon sim used by whatprot sim
        wp_dye_tracks,  # the dye-tracks simulated by whatprot
    ), peps_canonical = whatprot_erisyon.prepare_whatprot_train_input(
        whatprot_params, prep_result, sim_result, whatprot_folder
    )

    classify_sigproc = sigproc_flyte_result is not None

    sim_train_params = sim_result.params
    sim_test_params = None
    wp_seq_params_dict = json_load(wp_seq_params)
    logger.info("seq_params", wp_seq_params_dict=wp_seq_params_dict)

    logger.info("whatprot prepare_classfiy ")
    if classify_sigproc:
        sigproc_result = sigproc_flyte_result.load_result()
        ims_import_result = ims_import_flyte_result.load_result()
        (
            wp_filtered_sigproc_radmat,  # the filtered sigproc radmat out of erisyon
            wp_converted_sigproc_radmat,  # the sigproc radmat, as converted by whatprot
        ) = whatprot_erisyon.prepare_whatprot_classify_sigproc_input(
            ims_import_result,
            sigproc_result,
            whatprot_params.filter_params,
            whatprot_params.filter_reject_thresh_all_cycles,
            whatprot_folder,
            sim_train_params,
        )

    else:
        sim_test_result = sim_test_flyte_result.load_result()
        sim_test_params = sim_test_result.params

        (
            wp_filtered_sim_radmat,  # the sim radmat out of erisyon sim
            wp_converted_sim_radmat,  # the sim radmat as converted by whatprot
            wp_true_peps,  # the true pep_i per row of the sim radmat
        ) = whatprot_erisyon.prepare_whatprot_classify_sim_input(
            sim_test_result, whatprot_folder, peps_canonical
        )

    # Construct the result object and set_folder pointing to the place we
    # just wrote the files.  The FlyteDirectory dir inside of the FlyteResult
    # will get set to this location, and allow us to automatically download
    # that folder on request.
    wp_result = WhatprotPreV1ResultDC(
        params=whatprot_params,
        sim_train_params=sim_train_params,
        # wp_seq_params=wp_seq_params_dict,
        test_data_type="sigproc" if classify_sigproc else "sim",
        sim_test_params=sim_test_params,
    )
    wp_result.set_folder(whatprot_folder)
    return WhatprotPreV1FlyteResult.from_inst(wp_result)


@task(requests=Resources(cpu="4", mem="32Gi"), limits=Resources(cpu="16", mem="128Gi"))
def whatprot_pre_flyte_task(
    whatprot_params: WhatprotV1Params,
    prep_train_flyte_result: PrepV2FlyteResult,
    sim_train_flyte_result: SimV3FlyteResult,
    *,
    sim_test_flyte_result: Optional[SimV3FlyteResult] = None,
    ims_import_flyte_result: Optional[ImsImportFlyteResult] = None,
    sigproc_flyte_result: Optional[SigprocV2FlyteResult] = None,
) -> WhatprotPreV1FlyteResult:
    return whatprot_pre_flyte_task_worker(
        whatprot_params=whatprot_params,
        prep_train_flyte_result=prep_train_flyte_result,
        sim_train_flyte_result=sim_train_flyte_result,
        sim_test_flyte_result=sim_test_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
    )


# I'm wondering if we can use our own Erisyon simulation that we've
# already done to supply dyeseqs and dyetracks to whatprot classify.  The issue is
# that there is potentially a mismatch in the model for the simulation, so it may
# produce a slightly different set of seqs/trks.  But how is this likely to affect
# the whatprot classifier?  We're already passing data to classify that comes from
# a different model, inasmuch as our model is not perfect.  How would it affect
# the training of the classifier, though, to pass it "training" data (dyeseqs, dyetracks)
# that comes from a slightly different model?
#
# I mentioned the above in the Informatics Team meeting to Matt on 28Aril2023 and
# he said he didn't think it would be a problem to use the Erisyon sim to gen
# the dyetracks.
#
@task(requests=Resources(cpu="16", mem="300Gi"), limits=Resources(cpu="32", mem="1Ti"))
def whatprot_pre_flyte_task_xl(
    whatprot_params: WhatprotV1Params,
    prep_train_flyte_result: PrepV2FlyteResult,
    sim_train_flyte_result: SimV3FlyteResult,
    *,
    sim_test_flyte_result: Optional[SimV3FlyteResult] = None,
    ims_import_flyte_result: Optional[ImsImportFlyteResult] = None,
    sigproc_flyte_result: Optional[SigprocV2FlyteResult] = None,
) -> WhatprotPreV1FlyteResult:
    return whatprot_pre_flyte_task_worker(
        whatprot_params=whatprot_params,
        prep_train_flyte_result=prep_train_flyte_result,
        sim_train_flyte_result=sim_train_flyte_result,
        sim_test_flyte_result=sim_test_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
    )


def subprocess_cmd_list(cmd_list: list, outpath: Path) -> list:
    cmd_list = [
        str(c) for c in cmd_list
    ]  # convert Path objects, numeric literals, etc.

    # Use shlex to join and then re-split this command to benefit from its quoting/escaping
    # though this may be overkill here -- the filenames have been constructed in our code
    # and the params are all numerical and have been validated via dataclass schema.

    cmd_list = whatprot_erisyon.shlex_cmdline_escaping(cmd_list)

    logger.info("subprocess run", cmd_list=cmd_list)
    result = subprocess.run(cmd_list, capture_output=True, text=True)
    with open(outpath / "stdout.txt", "w") as f:
        f.write(result.stdout)
    with open(outpath / "stderr.txt", "w") as f:
        f.write(result.stderr)

    return cmd_list


# This task also currently uses both whatprot and plaster functionality.  The idea is that it could
# be changed to only use whatprot functionality such that it runs a container that only has whatprot,
# but not plaster.  This feels a little cumbersome, and I'm not clear on how "extra" code gets pulled
# in for execution in a pod -- e.g. clearly the code inside the @task does, but even the params these
# task fns use are defined in plaster -- does all that just get pulled in as necessary?
#
# But we could try this.  The following task really only uses a plaster function to gather the input
# filenames into a dict for convenience.  It would be simple enough to drop this and just build
# the expected filenames based on the passed folder names.
#
# And we'd need to pull a bit of whatprot-based conversion from the previous task to this one if this
# is going to be the only "whatprot-container" task.
#
# wp classify is not super memory intensive, but it can use a lot of cores if avail.
#
def whatprot_classify_flyte_task_worker(
    whatprot_params: WhatprotV1Params,
    wp_pre_flyte_result: WhatprotPreV1FlyteResult,
) -> WhatprotClassifyV1FlyteResult:

    # The input files have been written by the pre task, and
    # must be fetched to the compute node for access by this task.
    wp_pre_result = (
        wp_pre_flyte_result.load_result()  # downloads results from blob store to local filesystem
    )
    test_data_type = wp_pre_result.test_data_type
    pre_path = wp_pre_result.folder
    wpf = whatprot_erisyon.gather_whatprot_filenames_test(pre_path)

    # Look for the radiometry we'll classify
    rad = wpf[f"wp_converted_{test_data_type}_radmat_test"]
    logger.info("whatprot_run_flyte_task", radmat_file=rad)
    assert rad["exists"]

    # But this task will also write a file, and for this we need to
    # write to the working_directory for this task (blob store)
    cwd = current_context().working_directory
    out_path = Path(cwd) / "whatprot"
    out_path.mkdir(exist_ok=True)
    logger.info("whatprot_run_flyte_task", out_path=out_path)

    wpf_out = whatprot_erisyon.gather_whatprot_filenames_test(out_path)

    pred = wpf_out[f"wp_{test_data_type}_predictions_test"]
    logger.info("whatprot_run_flyte_task", pred=pred)

    whatprot_exe = whatprot_erisyon.whatprot_init()

    # fmt: off
    cmd_list = [
        whatprot_exe,
        'classify', 'hybrid',
        '-k', str(whatprot_params.wp_neighbors),
        '-s', str(whatprot_params.wp_sigma),
        '-H', str(whatprot_params.wp_passthrough),
        '-p', str(whatprot_params.wp_hmmprune),
        '-P', wpf['wp_seq_params_train']['name'],
        '-S', wpf['wp_dye_seqs_train']['name'],
        '-T', wpf['wp_dye_tracks_train']['name'],
        '-R', rad['name'],
        '-Y', pred['name']
    ]
    # fmt: on

    cmd_list = subprocess_cmd_list(cmd_list, out_path)

    # Same pattern as the PRE task above, so we can set_folder().
    # Is this necessary or is there a more automated/transparent way?
    wp_result = WhatprotClassifyV1ResultDC(
        params=whatprot_params,
        cmd_list=cmd_list,
    )
    wp_result.set_folder(out_path)
    return WhatprotClassifyV1FlyteResult.from_inst(wp_result)


@task(requests=Resources(cpu="64", mem="24Gi"), limits=Resources(cpu="192", mem="64Gi"))
def whatprot_classify_flyte_task(
    whatprot_params: WhatprotV1Params,
    wp_pre_flyte_result: WhatprotPreV1FlyteResult,
) -> WhatprotClassifyV1FlyteResult:
    return whatprot_classify_flyte_task_worker(
        whatprot_params=whatprot_params,
        wp_pre_flyte_result=wp_pre_flyte_result,
    )


@task(
    requests=Resources(cpu="96", mem="128Gi"), limits=Resources(cpu="192", mem="256Gi")
)
def whatprot_classify_flyte_task_xl(
    whatprot_params: WhatprotV1Params,
    wp_pre_flyte_result: WhatprotPreV1FlyteResult,
) -> WhatprotClassifyV1FlyteResult:
    return whatprot_classify_flyte_task_worker(
        whatprot_params=whatprot_params,
        wp_pre_flyte_result=wp_pre_flyte_result,
    )


# This task only uses plaster functionality - it loads the results from whatprot into
# a format ready for analysis and reporting in the plaster world.
@task(requests=Resources(cpu="8", mem="8Gi"), limits=Resources(cpu="32", mem="32Gi"))
def whatprot_post_classify_flyte_task(
    whatprot_params: WhatprotV1Params,
    wp_pre_flyte_result: WhatprotPreV1FlyteResult,
    wp_classify_flyte_result: WhatprotClassifyV1FlyteResult,
    sim_train_flyte_result: Optional[SimV3FlyteResult] = None,
) -> WhatprotPostClassifyV1FlyteResult:

    # Load the output files produced by whatprot and normalize them in dataframes
    # such that they are ready for consumption by plaster-based analysis and viz tools.

    # The predictions file was written to the run_result folder.
    wp_classify_result = wp_classify_flyte_result.load_result()
    run_path = wp_classify_result.folder
    wpf = whatprot_erisyon.gather_whatprot_filenames_test(run_path)
    logger.info("whatprot_post_flyte_task", run_path=run_path, wpf=wpf)

    # Get the results of classification. This may have been a sim or sigproc classification.
    sigproc = wpf["wp_sigproc_predictions_test"]["exists"]
    if sigproc:
        wp_classify_test_df = pd.read_csv(wpf["wp_sigproc_predictions_test"]["name"])
        true_peps_df = None
    else:
        wp_classify_test_df = pd.read_csv(wpf["wp_sim_predictions_test"]["name"])
        # TODO: these true peps could be had from the sim_train_flyte_result instead.
        # But this is reading from the file that whatprot read from, which means we
        # need to look at the output of the pre task.
        wp_pre_result = wp_pre_flyte_result.load_result()
        wpf_pre = whatprot_erisyon.gather_whatprot_filenames_test(wp_pre_result.folder)
        logger.info("whatprot_post_flyte_task", wpf_pre=wpf_pre)
        true_peps_df = pd.read_csv(
            wpf_pre["wp_true_peps_test"]["name"],
            names=["true_pep_iz"],
            header=0,
        )

    def normalize_whatprot_preds_df(df, erisyon_df_with_trues):
        # note conversion to 1-based pep_iz for Erisyon analysis.
        assert erisyon_df_with_trues is None or len(df) == len(erisyon_df_with_trues)
        df["pred_pep_iz"] = df["best_pep_iz"] + 1
        df["scores"] = df["best_pep_score"]
        if erisyon_df_with_trues is not None:
            df["true_pep_iz"] = erisyon_df_with_trues["true_pep_iz"] + 1
        return df

    wp_classify_test_df = normalize_whatprot_preds_df(wp_classify_test_df, true_peps_df)

    # TODO: to get train_pep_recalls, load the Sim result for either train (if sigproc) or
    # test (sim_test_data) and get the train_pep_recalls...
    # Maybe also set the trues here so we don't have to pass in the pre result and read those files.
    # sim_flyte_result.load_result().train_pep_recalls

    wp_result = WhatprotPostClassifyV1ResultDC(
        params=whatprot_params,
        pred_pep_iz=wp_classify_test_df.pred_pep_iz,
        scores=wp_classify_test_df.scores,
    )

    if not sigproc:
        # set true values and train recalls.  we could set train recalls even for sigproc, right?
        wp_result.true_pep_iz.set(wp_classify_test_df.true_pep_iz)
        wp_result.train_pep_recalls.set(
            sim_train_flyte_result.load_result().train_pep_recalls
        )

    return WhatprotPostClassifyV1FlyteResult.from_inst(wp_result)


@task(requests=Resources(cpu="8", mem="8Gi"), limits=Resources(cpu="32", mem="32Gi"))
def whatprot_fit_flyte_task(
    whatprot_params: WhatprotV1Params,
    wp_pre_flyte_result: WhatprotPreV1FlyteResult,
) -> WhatprotFitV1FlyteResult:

    # The input files have been written by the pre task, and
    # must be fetched to the compute node for access by this task.
    wp_pre_result = (
        wp_pre_flyte_result.load_result()  # downloads results from blob store to local filesystem
    )
    test_data_type = wp_pre_result.test_data_type
    pre_path = wp_pre_result.folder
    wpf = whatprot_erisyon.gather_whatprot_filenames_test(pre_path)

    # Look for the radiometry we'll classify
    rad = wpf[f"wp_converted_{test_data_type}_radmat_test"]
    logger.info("whatprot_fit_flyte_task", radmat_file=rad)
    assert rad["exists"]

    # But this task will also write a file, and for this we need to
    # write to the working_directory for this task (blob store)
    cwd = current_context().working_directory
    out_path = Path(cwd) / "whatprot"
    out_path.mkdir(exist_ok=True)
    logger.info("whatprot_fit_flyte_task", out_path=out_path)

    wpf_out = whatprot_erisyon.gather_whatprot_filenames_test(out_path)

    pred = wpf_out[f"wp_{test_data_type}_predictions_test"]
    logger.info("whatprot_fit_flyte_task", pred=pred)

    whatprot_exe = whatprot_erisyon.whatprot_init()
    dyeseq = whatprot_erisyon.get_first_dyeseq(Path(wpf["wp_dye_seqs_train"]["name"]))

    # fmt: off
    cmd_list = [
        whatprot_exe,
        "fit",
        "-P", wpf["wp_seq_params_train"]["name"],
        "-L", whatprot_params.wp_stoppingthreshold,
        "-M", whatprot_params.wp_maxruntime_minutes,
        "-x", dyeseq,
        "-R", rad["name"],
    ]
    # fmt: on

    cmd_list = subprocess_cmd_list(cmd_list, out_path)

    # Same pattern as the PRE task above, so we can set_folder().
    # Is this necessary or is there a more automated/transparent way?
    wp_result = WhatprotFitV1ResultDC(
        params=whatprot_params,
        cmd_list=cmd_list,
        params_fit=whatprot_erisyon.get_whatprot_fit_params(out_path / "stdout.txt"),
    )
    wp_result.set_folder(out_path)
    logger.info(
        "whatprot fit result",
        params=wp_result.params,
        cmd_list=wp_result.cmd_list,
        params_fit=wp_result.params_fit,
    )
    return WhatprotFitV1FlyteResult.from_inst(wp_result)


# 50 is perhaps an odd number of cpus to request based on typical node configs, but I
# do this bc the default whatprot bootstrap does 100 fits -- so we can get these done
# in two passes with 50 cores, otherwise it takes three (and a pass may take 30-60mins)
@task(requests=Resources(cpu="50", mem="16Gi"), limits=Resources(cpu="128", mem="32Gi"))
def whatprot_fit_bootstrap_flyte_task(
    whatprot_params: WhatprotV1Params,
    wp_pre_flyte_result: WhatprotPreV1FlyteResult,
) -> WhatprotFitBootstrapV1FlyteResult:

    # The input files have been written by the pre task, and
    # must be fetched to the compute node for access by this task.
    wp_pre_result = (
        wp_pre_flyte_result.load_result()  # downloads results from blob store to local filesystem
    )
    test_data_type = wp_pre_result.test_data_type
    pre_path = wp_pre_result.folder
    wpf = whatprot_erisyon.gather_whatprot_filenames_test(pre_path)

    # Look for the radiometry we'll classify
    rad = wpf[f"wp_converted_{test_data_type}_radmat_test"]
    logger.info("whatprot_fit_bootstrap_flyte_task", radmat_file=rad)
    assert rad["exists"]

    # But this task will also write a file, and for this we need to
    # write to the working_directory for this task (blob store)
    cwd = current_context().working_directory
    out_path = Path(cwd) / "whatprot"
    out_path.mkdir(exist_ok=True)
    logger.info("whatprot_fit_bootstrap_flyte_task", out_path=out_path)

    wpf_out = whatprot_erisyon.gather_whatprot_filenames_test(out_path)

    pred = wpf_out[f"wp_{test_data_type}_predictions_test"]
    logger.info("whatprot_fit_bootstrap_flyte_task", pred=pred)

    whatprot_exe = whatprot_erisyon.whatprot_init()
    dyeseq = whatprot_erisyon.get_first_dyeseq(Path(wpf["wp_dye_seqs_train"]["name"]))

    # fmt: off
    cmd_list = [
        whatprot_exe,
        "fit",
        "-P", wpf["wp_seq_params_train"]["name"],
        "-L", whatprot_params.wp_stoppingthreshold,
        "-M", whatprot_params.wp_maxruntime_minutes,
        "-x", dyeseq,
        "-R", rad["name"],
        "-b", whatprot_params.wp_numbootstrap,
        "-c", whatprot_params.wp_confidenceinterval,
        "-Y", pred["name"],
    ]
    # fmt: on

    # A not-so-elegant way to do nothing if numbootstrap is 0. This is primarily for
    # development and testing since the idea is that we'll always do bootstrapping
    # in the production workflow to establish confidence intervals on the best fit.
    if whatprot_params.wp_numbootstrap > 0:
        cmd_list = subprocess_cmd_list(cmd_list, out_path)
        wp_result = WhatprotFitBootstrapV1ResultDC(
            params=whatprot_params,
            cmd_list=cmd_list,
            bootstrap_bounds_lower=whatprot_erisyon.get_whatprot_fit_params(
                out_path / "stdout.txt", "lower"
            ),
            bootstrap_bounds_upper=whatprot_erisyon.get_whatprot_fit_params(
                out_path / "stdout.txt", "upper"
            ),
            bootstrap_csv=pred["name"],
        )
    else:
        wp_result = WhatprotFitBootstrapV1ResultDC(
            params=whatprot_params,
            cmd_list=[],
            bootstrap_bounds_lower={},
            bootstrap_bounds_upper={},
            bootstrap_csv="",
        )

    wp_result.set_folder(out_path)
    return WhatprotFitBootstrapV1FlyteResult.from_inst(wp_result)
