"Generator for survey tasks."
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import flytekit
import pandas as pd
from dataclasses_json import DataClassJsonMixin
from flytekit import Resources, task, workflow
from flytekit.core.dynamic_workflow_task import dynamic
from flytekit.types.directory import FlyteDirectory

# initialize logging as early as possible in an attempt to capture logs early
from plaster import env

env.configure_logging()
import structlog

logger = structlog.get_logger()

import plaster.genv2.gen_config as gen_config
import plaster.genv2.generators.sim as sim
from plaster.genv2 import gen_utils
from plaster.run.prep_v2.prep_v2_params import PrepV2Params
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.prep_v2.prep_v2_task import prep_flyte_task
from plaster.run.sim_v3.sim_v3_params import Fret, Marker, SeqParams, SimV3Params
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult
from plaster.run.sim_v3.sim_v3_task import sim_v3_flyte_task
from plaster.run.survey_v2.survey_v2_params import (
    ProteaseLabelScheme,
    SchemePermutator,
    SurveyV2Params,
)
from plaster.run.survey_v2.survey_v2_result import SurveyV2FlyteResult, SurveyV2ResultDC
from plaster.run.survey_v2.survey_v2_task import survey_v2_flyte_task
from plaster.tools.flyte import task_configs


@dataclass
class SurveyConfig(gen_config.BaseGenConfig):
    """
    Configuration dataclass for Survey.

    A survey is essentially a collection of sims run with various
    protease/labeling schemes.  The contained SimConfig serves to
    provide a base copy of all sequencing params.

    """

    scheme_permutator: SchemePermutator = gen_config.gen_field(
        default=None,
        help="SchemePermutator class to generate permutations of proteases and labels",
    )

    sim_config: sim.SimConfig = gen_config.gen_field(
        default=None,
        help="A SimConfig object that defines all common simulation params for the schemes surveyed",
    )

    def fetch_protein_sequences(self):
        # This is only here until the generator logic is called for flyte workflows.
        # At the moment, a special hack in views.py calls this if it exists, so it
        # needs to exist in this class and we'll in turn call this on our sim_config.
        self.sim_config.fetch_protein_sequences()


# ==========================================================================================
#
# Doing this within the dynamic workflow at present.  Is there utility in it being a
# separate task?  It seems such trivial work, why spin up a pod to do this?
#
# @task(requests=Resources(cpu="2", mem="1Gi"))
# def extract_params(
#     config: sim.SimConfig,
# ) -> tuple[PrepV2Params, SimV3Params, SurveyV2Params]:
#     prep_params = PrepV2Params(
#         proteins=config.proteins,
#         proteases=config.proteases,
#         decoy_mode=config.decoy_mode,
#         shuffle_n=config.shuffle_n,
#         is_photobleaching_run=config.is_photobleaching_run,
#         photobleaching_n_cycles=config.photobleaching_n_cycles,
#         photobleaching_run_n_dye_count=config.photobleaching_run_n_dye_count,
#     )

#     sim_v3_params = SimV3Params(
#         markers=config.markers,
#         frets=config.frets,
#         seq_params=config.seq_params,
#         n_samples_train=config.n_samples_train,
#         n_samples_test=config.n_samples_test,
#         is_survey=True,
#     )

#     survey_v2_params = SurveyV2Params()

#     return prep_params, sim_v3_params, survey_v2_params


# Survey writes individual scheme results as they are processed, so we don't
# want to clear the job folder as we normally do in write_job_folder_result().
# Instead we do it at startup.
@task(
    requests=Resources(cpu="2", mem="1Gi"),
    task_config=task_configs.generate_efs_task_config(),
)
def clean_job_folder(config: SurveyConfig) -> None:
    path = gen_utils.resolve_job_folder(config.job)
    if path.exists():
        shutil.rmtree(path)


@task(
    requests=Resources(cpu="2", mem="1Gi"),
    task_config=task_configs.generate_efs_task_config(),
)
def write_job_folder_result(
    config: SurveyConfig,
    survey_flyte_result: SurveyV2FlyteResult,
):
    # Write job folder if specified
    if config.job:
        # Note clean=False because individual survey sims have written
        # their results as part of the dynamic workflow and we don't
        # want to erase all of those.
        job_path = gen_utils.write_job_folder(
            job_folder=config.job, config_dict=config.to_dict(), clean=False
        )
        survey_flyte_result.save_to_disk(job_path / "_survey_combined")

        # Write reports
        reports = [
            "survey.ipynb",
        ]
        gen_utils.write_reports(job_path=job_path, reports=reports)


@task(
    requests=Resources(cpu="2", mem="1Gi"),
    task_config=task_configs.generate_efs_task_config(),
)
def write_job_folder_scheme_result(
    config: sim.SimConfig,
    prep_flyte_result: PrepV2FlyteResult,
    sim_flyte_result: SimV3FlyteResult,
    survey_flyte_result: SurveyV2FlyteResult,
):
    # write config 'manifest' to get written to this scheme "sub-job"
    # Note that I'm causing it to be called "run_manifest.yaml" intead of
    # "job_manifest.yaml" to distinguish it as a "sub-job", which we have
    # historically referred to as a "run"
    path = gen_utils.write_job_folder(
        job_folder=config.job,
        config_dict=config.to_dict(),
        # I'm calling this anything different than job_manifest so it doesn't
        # confuse indexer - I don't want these subruns listed as individual jobs.
        # "Runs" is what they were historically called anyway.
        manifest_name="run_manifest.yaml",
    )
    # and write task results from the scheme sim+survey
    prep_flyte_result.save_to_disk(path / "prep")
    sim_flyte_result.save_to_disk(path / "sim")
    survey_flyte_result.save_to_disk(path / "survey")


@task(requests=Resources(cpu="4", mem="16Gi"))
def combine_survey_results(
    schemes: list[ProteaseLabelScheme], survey_results: list[SurveyV2FlyteResult]
) -> SurveyV2FlyteResult:
    # Combine the DataFrames from individual schemes.
    # TODO: determine if this is useful.  Historically and in my port of JobResult
    # for initial use as FlyteSurveyJobResult contains all of the meta- functionality
    # to run fns over all runs/DFs so it may be one of the two should be removed.

    dfs = []
    for scheme, survey_result in zip(schemes, survey_results):
        df = survey_result.load_result().survey
        df["scheme"] = str(scheme)  # this was historically called "run_name"
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    result = SurveyV2ResultDC(params=SurveyV2Params(), schemes=schemes, _survey=df)
    return SurveyV2FlyteResult.from_inst(result)


# ==========================================================================================
# dynamic workflow approach


@dynamic
def survey_n_schemes(survey_config: SurveyConfig) -> SurveyV2FlyteResult:
    # create protease-label schemes and compute the survey metrics for each

    schemes = survey_config.scheme_permutator.gen_schemes()

    survey_results = []
    for scheme in schemes:
        logger.info(f"Running survey scheme {scheme}", scheme=scheme)
        config = replace(
            survey_config.sim_config,
            job=f"{survey_config.job}/{scheme}",
            proteases=[scheme.protease],
            markers=[Marker(aa=label) for label in scheme.labels],
            type="sim_survey",  # currently not machine-read downstream, just for human reference.
        )

        # the extraction of these params was previously a separate task.  why?
        prep_params = PrepV2Params(
            proteins=config.proteins,
            proteases=config.proteases,
            decoy_mode=config.decoy_mode,
            shuffle_n=config.shuffle_n,
            is_photobleaching_run=config.is_photobleaching_run,
            photobleaching_n_cycles=config.photobleaching_n_cycles,
            photobleaching_run_n_dye_count=config.photobleaching_run_n_dye_count,
        )

        sim_v3_params = SimV3Params(
            markers=config.markers,
            frets=config.frets,
            seq_params=config.seq_params,
            n_samples_train=config.n_samples_train,
            n_samples_test=config.n_samples_test,
            is_survey=True,
        )

        # Now proceed with other tasks per scheme.  What I want eventually is
        # to see in my job_folder something that looks like:
        #
        # my_survey_job
        #    _reports     # reports live here as usual
        #
        #    trypsin_C_K  # a scheme
        #       prep      # prep task results
        #       sim_v3    # sim task results
        #       survey    # survey metrics computation
        #
        #    trypsin_C_Y  # another scheme, etc

        # In flyte, each task gets a unique working_directory, which is
        # where it serializes its results when converting from a resultDC
        # to a FlytResult-based class. So by the time you get back the
        # xxx_result below for each task, it has already been written
        # to disk.  Later, in write_job_folder_result() task, the contents
        # of those folders will get copied into subfolders under your
        # //jobs_folder.
        #
        # We will want to modify the write_job_folder_result() task so
        # that it puts each scheme into a subfolder.  We could choose
        # to wait until ALL schemes are done, and then call the
        # write_x task, OR we could call it here and have the results
        # copied per scheme as they are computed.

        working_dir = flytekit.current_context().working_directory
        local_dir = Path(working_dir) / str(scheme)
        local_dir.mkdir(exist_ok=True)

        prep_flyte_result = prep_flyte_task(prep_params=prep_params)

        sim_flyte_result = sim_v3_flyte_task(
            sim_params=sim_v3_params, prep_flyte_result=prep_flyte_result
        )

        survey_flyte_result = survey_v2_flyte_task(
            survey_v2_params=SurveyV2Params(),
            prep_result=prep_flyte_result,
            sim_v3_result=sim_flyte_result,
        )

        survey_results.append(survey_flyte_result)

        # cause these results to end up in a subfolder named for the scheme
        folder = gen_utils.resolve_job_folder(str(survey_config.job)) / str(scheme)
        write_job_folder_scheme_result(
            config=config,
            prep_flyte_result=prep_flyte_result,
            sim_flyte_result=sim_flyte_result,
            survey_flyte_result=survey_flyte_result,
        )

    return combine_survey_results(schemes=schemes, survey_results=survey_results)


@workflow
def survey_workflow(config: SurveyConfig) -> SurveyV2FlyteResult:
    # This is the top-level workflow which calls a dynamic task/workflow
    # that builds and executes the DAG at runtime.

    promise_0 = clean_job_folder(config=config)

    survey_result = survey_n_schemes(survey_config=config)

    # use some special Flyte syntax to say clean_job_folder needs to complete
    # before running survey_n_schemes, even though clean_job_folder does not
    # return a value that survey_n_schemes needs.
    promise_0 >> survey_result

    # All of the prep, sim, and survey results have already been written in scheme
    # subfolders during survey_n_schemes.  here we write the job_maniftest.yaml and
    # copy any reports, as well as writing the top-level combined survey result.

    write_job_folder_result(config=config, survey_flyte_result=survey_result)

    return survey_result


# ==========================================================================================


def generate(config: SurveyConfig) -> gen_config.GenerateResult:
    """Generate a survey job from a sim.SimConfig dataclass.

    Args
    ----
    config : sim.SimConfig
        A sim.SimConfig specifying information for one or more runs.

    Returns
    -------
    A gen_config.GenerateResult object describing the runs and static_reports to be executed
    for this survey job.
    """

    # NOTE this doesn't actually get run by the flyte-from-controlpanel!
    # Simplify this when ripping out all the GenV2 stuff.

    # Flyte doesn't currently have a concept of runs within a job, but this may change
    # when this generate fn takes some kind of permutation information to help users
    # create a series of related jobs (formerly called runs within a job).  Or similar.
    runs = []

    static_reports = ["survey"]

    config.sim_config.fetch_protein_sequences()

    return gen_config.GenerateResult(runs=runs, static_reports=static_reports)


generator = gen_config.Generator(SurveyConfig, generate, workflow=survey_workflow)
