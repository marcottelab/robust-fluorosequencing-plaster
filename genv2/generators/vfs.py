"Generator for VFS tasks."
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml
from flytekit import Resources, task, workflow

# initialize logging as early as possible in an attempt to capture logs early
from plaster import env

env.configure_logging()

# generators
from plaster.genv2 import gen_config, gen_utils
from plaster.genv2.generators import classify, sim

# results
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult

# run/task
from plaster.run.prep_v2.prep_v2_task import prep_flyte_task
from plaster.run.rf_train_v2.rf_train_v2_params import RFTrainV2Params
from plaster.run.rf_train_v2.rf_train_v2_result import RFTrainV2FlyteResult
from plaster.run.rf_train_v2.rf_train_v2_task import (
    rf_train_v2_big_flyte_task,
    rf_train_v2_flyte_task,
)

# params
from plaster.run.rf_v2.rf_v2_params import RFV2Params
from plaster.run.rf_v2.rf_v2_result import RFV2FlyteResult
from plaster.run.rf_v2.rf_v2_task import rf_v2_big_flyte_task, rf_v2_flyte_task
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult
from plaster.run.sim_v3.sim_v3_task import sim_v3_big_flyte_task, sim_v3_flyte_task
from plaster.tools.flyte import task_configs

# A VFS is a simulation followed by the training and testing of a classifier
# on the simulation.  At present, this classifier is always RF.
#
# VFSConfig currently inherits from SimConfig, but it may be nice to instead
# have VFSConfig instead contain a SimConfig, and optionally reference an
# external already-run simulation (using one or the other).
#
# VFS is used to analyze, ahead of actual real-data-prep-and-analysis, how
# well we can classify data given certain conditions.  For a definition, see
# https://www.notion.so/erisyon/Virtual-fluorosequencing-VFS-8cf83d47132c401da1b875b980eb49a6
#
# So we'd eventually like to extend this to trying multiple classifiers, and seeing
# how they compare.  In this scenario, you'd only want to run the simulation once,
# and then train and test the classifier(s) on that same simulated data.
#
# So it may be best to think about this in terms of workflows that train and
# test a given classifier (with reference to a simulation), and then we can
# string a handful of these together, refering to the same simulated data,
# and compare the results of the various classifiers included.
#
# So at present, VFS is what trains (and tests) a RandomForest classifier.
# But really maybe this should be called the rf_train_workflow.
# Note that we have a classify_workflow, which uses the trained rf classifier
# from this workflow to classify other simulated or real data.
# Most classifiers will fall into this scheme of separate "train" and "classfiy"
# phases.  Whatprot, notably, is a one-pass workflow -- there is no separate
# training and testing/classify phase.


@dataclass
class VFSConfig(sim.SimConfig):
    """Configuration dataclass for VFS.

    VFS, "virtual fluorosequencing" means doing a simulation and training
    and testing a classifier against the labelled peptides coming out of
    said simulation.  At present, the classifier is RF.

    Work is being done to break apart these tasks for better composability.

    That is, you can also choose to (a) do a simulation and separately (b) train
    and test various classifiers on this simulated data.

    But the VFS workflow, at present, still means sim + RF

    """

    # Current not necessary - we only support Random Forest for VFS
    #
    # classifier: Classifier = gen_config.gen_field(
    #     help=VFSConfigHelp.CLASSIFIER_HELP,
    #     default=Classifier.RF,
    #     metadata_config=config(
    #         encoder=lambda classifier: classifier.value,
    #         decoder=Classifier,
    #     ),
    # )

    # maybe you end up wanting a list of classifier params
    # for various classifiers
    rf_train_params: RFTrainV2Params = RFTrainV2Params()
    rf_test_params: RFV2Params = RFV2Params()


@task
def extract_params(
    config: VFSConfig,
) -> Tuple[sim.PrepV2Params, sim.SimV3Params, RFTrainV2Params, RFV2Params]:
    prep_params = sim.generate_prep_params(config)
    sim_params = sim.generate_sim_params(config)
    rf_train_params = config.rf_train_params
    rf_test_params = config.rf_test_params
    return prep_params, sim_params, rf_train_params, rf_test_params


@task(
    task_config=task_configs.generate_efs_task_config(),
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="8", mem="64Gi"),
)
def write_vfs_job_folder_result(
    config: VFSConfig,
    prep_flyte_result: sim.PrepV2FlyteResult,
    sim_flyte_result: sim.SimV3FlyteResult,
    rf_train_flyte_result: classify.RFTrainV2FlyteResult,
    rf_test_flyte_result: classify.RFV2FlyteResult,
) -> None:

    # Write job folder if specified
    if config.job:
        job_path = gen_utils.write_job_folder(
            job_folder=config.job, config_dict=config.to_dict()
        )

        # Write result folders
        prep_flyte_result.save_to_disk(job_path / "prep")
        sim_flyte_result.save_to_disk(job_path / "sim")
        rf_train_flyte_result.save_to_disk(job_path / "rf_train")
        rf_test_flyte_result.save_to_disk(job_path / "rf_test")

        # Write reports
        reports = [
            "vfs.ipynb",
        ]

        gen_utils.write_reports(job_path=job_path, reports=reports)


def load_vfs_results(
    path: str,
) -> Tuple[
    VFSConfig,
    PrepV2FlyteResult,
    SimV3FlyteResult,
    RFTrainV2FlyteResult,
    RFV2FlyteResult,
]:
    """
    Load VFS result from a given path.

    Args
    ----
    path: str
        Path to a VFS results folder.
    """
    path = Path(path).expanduser()
    vfs_config = VFSConfig.from_dict(
        yaml.safe_load((path / "job_manifest.yaml").read_text())["config"]
    )
    prep_flyte_result = PrepV2FlyteResult.load_from_disk(path / "prep")
    sim_flyte_result = SimV3FlyteResult.load_from_disk(path / "sim")
    rf_train_flyte_result = RFTrainV2FlyteResult.load_from_disk(path / "rf_train")
    rf_test_flyte_result = RFV2FlyteResult.load_from_disk(path / "rf_test")

    return (
        vfs_config,
        prep_flyte_result,
        sim_flyte_result,
        rf_train_flyte_result,
        rf_test_flyte_result,
    )


@workflow
def vfs_workflow(
    config: VFSConfig,
) -> Tuple[PrepV2FlyteResult, SimV3FlyteResult, RFTrainV2FlyteResult, RFV2FlyteResult]:
    """
    VFS generation workflow.

    Args
    ----
    config : VFSConfig
        Specifies VFS sim+classify configuration.

    Returns
    -------
    Tuple of Result objects containing the results of the VFS run.

    """
    prep_params, sim_params, rf_train_params, rf_params = extract_params(config=config)
    prep_flyte_result = prep_flyte_task(prep_params=prep_params)

    sim_flyte_result = sim_v3_flyte_task(
        sim_params=sim_params, prep_flyte_result=prep_flyte_result
    )
    rf_train_flyte_result = rf_train_v2_flyte_task(
        rf_train_params=rf_train_params,
        prep_flyte_result=prep_flyte_result,
        sim_flyte_result=sim_flyte_result,
    )
    rf_test_flyte_result = rf_v2_flyte_task(
        rf_params=rf_params,
        rf_train_flyte_result=rf_train_flyte_result,
        sim_flyte_result=sim_flyte_result,
        sigproc_flyte_result=None,
    )

    write_vfs_job_folder_result(
        config=config,
        prep_flyte_result=prep_flyte_result,
        sim_flyte_result=sim_flyte_result,
        rf_train_flyte_result=rf_train_flyte_result,
        rf_test_flyte_result=rf_test_flyte_result,
    )

    return (
        prep_flyte_result,
        sim_flyte_result,
        rf_train_flyte_result,
        rf_test_flyte_result,
    )


@workflow
def vfs_big_workflow(
    config: VFSConfig,
) -> Tuple[PrepV2FlyteResult, SimV3FlyteResult, RFTrainV2FlyteResult, RFV2FlyteResult]:
    """
    VFS generation workflow.

    Args
    ----
    config : SimConfig
        Specifies VFS simulation configuration.

    Returns
    -------
    Tuple of Result objects containing the results of the VFS run.

    """
    prep_params, sim_params, rf_train_params, rf_params = extract_params(config=config)
    prep_flyte_result = prep_flyte_task(prep_params=prep_params)

    sim_flyte_result = sim_v3_big_flyte_task(
        sim_params=sim_params, prep_flyte_result=prep_flyte_result
    )
    rf_train_flyte_result = rf_train_v2_big_flyte_task(
        rf_train_params=rf_train_params,
        prep_flyte_result=prep_flyte_result,
        sim_flyte_result=sim_flyte_result,
    )
    rf_test_flyte_result = rf_v2_big_flyte_task(
        rf_params=rf_params,
        rf_train_flyte_result=rf_train_flyte_result,
        sim_flyte_result=sim_flyte_result,
        sigproc_flyte_result=None,
    )

    write_vfs_job_folder_result(
        config=config,
        prep_flyte_result=prep_flyte_result,
        sim_flyte_result=sim_flyte_result,
        rf_train_flyte_result=rf_train_flyte_result,
        rf_test_flyte_result=rf_test_flyte_result,
    )

    return (
        prep_flyte_result,
        sim_flyte_result,
        rf_train_flyte_result,
        rf_test_flyte_result,
    )


def generate(config: VFSConfig) -> gen_config.GenerateResult:
    """Generate a VFS job from a VFSConfig dataclass.

    Args
    ----
    config : VFSConfig
        A VFSConfig specifying sim+classify configuration

    Returns
    -------
    A gen_config.GenerateResult object describing the runs and static_reports to be executed
    for this VFS job.
    """

    # Flyte doesn't currently have a concept of runs within a job, but this may change
    # when this generate fn takes some kind of permutation information to help users
    # create a series of related jobs (formerly called runs within a job).  Or similar.
    runs = []

    static_reports = ["vfs"]

    config.fetch_protein_sequences()

    return gen_config.GenerateResult(runs=runs, static_reports=static_reports)


generator = gen_config.Generator(VFSConfig, generate, workflow=vfs_workflow)
generator_big = gen_config.Generator(VFSConfig, generate, workflow=vfs_big_workflow)
