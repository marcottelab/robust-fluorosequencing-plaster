from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin
from flytekit import task, workflow

# initialize logging as early as possible in an attempt to capture logs early
from plaster import env

env.configure_logging()
import structlog

logger = structlog.get_logger()


from plaster.genv2 import gen_config, gen_utils
from plaster.genv2.gen_config import BaseGenConfig, gen_field
from plaster.genv2.gen_utils import resolve_job_folder, write_job_folder
from plaster.reports.helpers.report_params import FilterParams
from plaster.run.ims_import.ims_import_result import ImsImportFlyteResult
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2FlyteResult
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult
from plaster.run.whatprot_v1.whatprot_v1_params import WhatprotV1Params
from plaster.run.whatprot_v1.whatprot_v1_result import (
    WhatprotClassifyV1FlyteResult,
    WhatprotFitBootstrapV1FlyteResult,
    WhatprotFitV1FlyteResult,
    WhatprotPostClassifyV1FlyteResult,
    WhatprotPreV1FlyteResult,
)
from plaster.run.whatprot_v1.whatprot_v1_task import (
    whatprot_classify_flyte_task,
    whatprot_classify_flyte_task_xl,
    whatprot_fit_bootstrap_flyte_task,
    whatprot_fit_flyte_task,
    whatprot_post_classify_flyte_task,
    whatprot_pre_flyte_task,
    whatprot_pre_flyte_task_xl,
)
from plaster.tools.flyte import remote, task_configs


@dataclass
class WhatprotConfig(BaseGenConfig):
    # These first wp_ values are passed to the whatprot command line,
    # and can't be deduced from the related Erisyon sim/vfs.
    # At present we always use the 'hybrid' classification mode.

    # From the whatprot README.md at https://github.com/marcottelab/whatprot

    # Generate radiometry samples:
    #   -g (or --numgenerate) number of reads to simulate total. We recommend setting this to 10000. The actual
    #      number of reads will be less, because reads of unlabelable peptides or all dud-dyes will be removed
    #      from the results -- these would not be seen in real data.

    # Classify data using the hybrid classifier
    #   -k (or --neighbors) number of neighbors to use for kNN part of hybrid classifier.
    #   -s (or --sigma) sigma value for gaussian weighting function for neighbor voting.
    #   -H (or --passthrough) max-cutoff for number of peptides to forward from kNN to HMM
    #   -p (or --hmmprune) pruning cutoff for HMM (measured in sigma of fluorophore/count
    #      combination). This parameter is optional; if omitted, no pruning cutoff will be
    #      used.

    # Fit data using whatprot.
    #   -L (or --stoppingthreshold) iteration will stop when the difference between
    #      iterations is less than this value. Note that the difference between the fit
    #      value and the true value may be more than this limit.
    #   -b (or --numbootstrap) declares the number of bootstrap rounds to run.
    #   -c (or --confidenceinterval) declares the size of the confidence interval. Giving
    #      0.9 will give a 90% confidence interval using the percentile method, starting
    #      at the 5th percentile and ending at the 95th.
    #   -Y (or --results) for the path to where you want to save the bootstrapping information.

    # Classify-related
    wp_numgenerate: int = gen_field(
        default=10000,
        help="Number of reads to simulate total. We recommend setting this to 10000. The actual"
        " number of reads will be less, because reads of unlabelable peptides or all dud-dyes"
        " will be removed from the results -- these would not be seen in real data.",
    )

    wp_neighbors: int = gen_field(
        default=10000,
        help="Number of neighbors to use for kNN part of hybrid classifier.",
    )
    wp_sigma: float = gen_field(
        default=0.5,
        help="Sigma value for gaussian weighting function for neighbor voting.",
    )
    wp_passthrough: int = gen_field(
        default=1000,
        help="Max-cutoff for number of peptides to forward from kNN to HMM",
    )
    wp_hmmprune: Optional[int] = gen_field(
        default=5,
        help="pruning cutoff for HMM (measured in sigma of fluorophore/count combination)."
        " This parameter is optional; if omitted, no pruning cutoff will be used.",
    )

    # Fit-related
    wp_stoppingthreshold: float = gen_field(
        default=0.00001,
        help="Iteration will stop when the difference between iterations is less than this value."
        " Note that the difference between the fit value and the true value may be more than this limit.",
    )
    wp_maxruntime_minutes: Optional[int] = gen_field(
        default=180,
        help="Sets a time limit in minutes, after which fit will stop,"
        " even if the wp_stoppingthreshold has not been reached.",
    )
    wp_numbootstrap: Optional[int] = gen_field(
        default=0, help="declares the number of bootstrap rounds to run."
    )
    wp_confidenceinterval: Optional[float] = gen_field(
        default=0.9,
        help="declares the size of the confidence interval. Giving 0.9 will give a 90%"
        " confidence interval using the percentile method, starting at the 5th percentile"
        " and ending at the 95th.",
    )

    # Whatprot is further configured based on an Erisyon sim/vfs
    # job that gives simulation params and training database.
    #
    sim_train_flyte_result: Optional[SimV3FlyteResult] = gen_field(
        default=None,
        help="Directly provided sim flyte result for testing purposes.",
        hidden=True,
    )
    sim_train_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the sim job used to train the classifier.",
    )
    sim_train_id: Optional[str] = gen_field(
        default=None,
        help="The Flyte execution ID of the sim job used to train the classifier.",
    )

    # You can classify data from an entirely different sim, though
    # this could also point to the same sim/vfs job as above, or not
    # spec'd at all if you just want to classify sigproc data.
    #
    sim_test_flyte_result: Optional[SimV3FlyteResult] = gen_field(
        default=None,
        help="Directly provided sim flyte result for testing purposes.",
        hidden=True,
    )
    sim_test_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the sim job whose output you want to classify.",
    )
    sim_test_id: Optional[str] = gen_field(
        default=None,
        help="The Flyte execution ID of the sim job whose output you want to classify.",
    )

    # Or you can classify sigproc data with a classifier trained on the sim
    # output from the sim_train
    #
    sigproc_flyte_result: Optional[SigprocV2FlyteResult] = gen_field(
        default=None,
        help="Directly provided sigproc flyte result for testing purposes.",
        hidden=True,
    )
    sigproc_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the sigproc job whose data you want to classify.",
    )
    sigproc_id: Optional[str] = gen_field(
        default=None,
        help="The Flyte execution ID of the sigproc job whose data you want to classify.",
    )

    # Sigproc data requires filtering ahead of input to whatprot.  This filtering is
    # currently provided by values in a ReportParams object which is the result of
    # a human tweaking values after looking at radiometry reports.  So this file
    # will live on EFS, and before launching the flyte workflow the generator
    # will load this into the filter_params attribute for inclusion in WhatprotParams.
    # (Or you can spec the filter_params manually in your config)

    filter_params_yaml_path: Optional[str] = gen_field(
        default=None,
        path=True,
        help="path to yaml file holding filtering parameters.",
    )

    filter_params: Optional[FilterParams] = gen_field(
        default=None,
        help="A FilterParams object whose values are used to filter sigproc data.  You can specific this"
        " directly, or specify where to find the yaml file with filter_params_yaml_path.",
    )

    filter_reject_thresh_all_cycles: Optional[bool] = gen_field(
        default=False,
        help="Should reject/thresh filtering test signal at all cycles?  If not, only cycle 0.",
    )

    def __post_init__(self):
        sim_train_present = (
            self.sim_train_flyte_result or self.sim_train_job or self.sim_train_id
        )
        sim_test_present = (
            self.sim_test_flyte_result or self.sim_test_job or self.sim_test_id
        )
        sigproc_present = (
            self.sigproc_flyte_result or self.sigproc_job or self.sigproc_id
        )
        if not sim_train_present:
            raise ValueError("A sim_train job must be specified for input to whatprot.")
        if not (sim_test_present or sigproc_present):
            raise ValueError(
                "One of sim_test or sigproc must be specified for classification or fitting."
            )
        if sim_test_present and sigproc_present:
            raise ValueError(
                "Either sim_test or sigproc should be specified for classification/fitting, not both."
            )
        if sigproc_present and not (self.filter_params_yaml_path or self.filter_params):
            raise ValueError(
                "You've specified sigproc data but no filtering params.  Use filter_params_yaml_path or filter_params."
            )

    def load_filtering_params(self):
        if self.filter_params_yaml_path:
            path = resolve_job_folder(str(self.filter_params_yaml_path))
            if not path.exists():
                msg = f"params not found at specified filter_params_yaml_path {path}"
                logger.error(msg)
                raise FileExistsError(msg)

            self.filter_params = FilterParams.load_from_path(path)
            if not self.filter_params.check_all_present():
                msg = f"not all required filter params are present in {path}"
                logger.error(msg, filter_params=self.filter_params)
                raise ValueError(msg)

            # set this to actual path for reference in results: what path was actually loaded
            # as a result of resolve_job_folder
            self.filter_params_yaml_path = str(path)


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_sigproc_flyte_results(
    config: WhatprotConfig,
) -> tuple[Optional[ImsImportFlyteResult], Optional[SigprocV2FlyteResult]]:
    ims_import_flyte_result = None
    sigproc_flyte_result = None

    if config.sigproc_flyte_result:
        sigproc_flyte_result = config.sigproc_flyte_result
    elif config.sigproc_id:
        r = remote.fetch_flyte_remote()
        workflow_ex = r.fetch_execution(name=config.sigproc_id)
        r.sync_execution(
            workflow_ex,
        )

        # Note: it used to be that this fn only returned the result of the
        # sigproc task, but exactly analogous to the fetch_sim_train_flyte_results()
        # below, a caller sometimes needs results from the ims_import task of
        # the sigproc job as well.  Because older runs will not contain this output
        # at the expected node in the graph, I'll wrap this in a handler to
        # return only the sigproc result if that is what is available.
        # What is the best practice for dealing with changing IO signatures
        # like this? Note how this change is innocuous in the job_folder
        # version below.
        # 30 jan 2023 tfb
        #
        try:
            ims_import_flyte_result: ImsImportFlyteResult = workflow_ex.outputs.get(
                "o0", as_type=ImsImportFlyteResult
            )
            sigproc_flyte_result: SigprocV2FlyteResult = workflow_ex.outputs.get(
                "o1", as_type=SigprocV2FlyteResult
            )
        except Exception:
            # I must be able to get at the ImsImport result as well, maybe thru
            # querying the workflow for that task, and then asking for that task
            # output.  For now, set ims to None.
            ims_import_flyte_result = None

            # ims_task = workflow_ex.node_executions['n1']
            # To get this, you need to set sync_nodes=True in the call to sync_execution() above,
            # but it's still not immediately clear how to get the output.

            sigproc_flyte_result: SigprocV2FlyteResult = workflow_ex.outputs.get(
                "o0", as_type=SigprocV2FlyteResult
            )

    elif config.sigproc_job:
        ims_import_flyte_result = ImsImportFlyteResult.load_from_disk(
            resolve_job_folder(config.sigproc_job) / "ims_import"
        )
        sigproc_flyte_result = SigprocV2FlyteResult.load_from_disk(
            resolve_job_folder(config.sigproc_job) / "sigproc_v2"
        )

    return ims_import_flyte_result, sigproc_flyte_result


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_sim_train_flyte_results(
    config: WhatprotConfig,
) -> tuple[PrepV2FlyteResult, SimV3FlyteResult]:

    prep_train_flyte_result = None
    sim_train_flyte_result = None

    if config.sim_train_flyte_result:
        sim_train_flyte_result = config.sim_train_flyte_result
    elif config.sim_train_id:
        r = remote.fetch_flyte_remote()
        workflow_ex = r.fetch_execution(name=config.sim_train_id)
        r.sync_execution(workflow_ex)
        prep_train_flyte_result: PrepV2FlyteResult = workflow_ex.outputs.get(
            "o0", as_type=PrepV2FlyteResult
        )
        sim_train_flyte_result: SimV3FlyteResult = workflow_ex.outputs.get(
            "o1", as_type=SimV3FlyteResult
        )
    elif config.sim_train_job:
        prep_train_flyte_result = PrepV2FlyteResult.load_from_disk(
            resolve_job_folder(config.sim_train_job) / "prep"
        )
        sim_train_flyte_result = SimV3FlyteResult.load_from_disk(
            resolve_job_folder(config.sim_train_job) / "sim"
        )

    return prep_train_flyte_result, sim_train_flyte_result


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_sim_test_flyte_result(
    config: WhatprotConfig,
) -> Optional[SimV3FlyteResult]:

    sim_test_flyte_result = None

    if config.sim_test_flyte_result:
        sim_test_flyte_result = config.sim_test_flyte_result
    elif config.sim_test_id:
        r = remote.fetch_flyte_remote()
        workflow_ex = r.fetch_execution(name=config.sim_test_id)
        r.sync_execution(workflow_ex)
        sim_test_flyte_result: SimV3FlyteResult = workflow_ex.outputs.get(
            "o1", as_type=SimV3FlyteResult
        )
    elif config.sim_test_job:
        sim_test_flyte_result = SimV3FlyteResult.load_from_disk(
            resolve_job_folder(config.sim_test_job) / "sim"
        )

    return sim_test_flyte_result


@task(task_config=task_configs.generate_efs_task_config())
def write_classify_job_folder_result(
    config: WhatprotConfig,
    wp_pre_flyte_result: WhatprotPreV1FlyteResult,
    wp_classify_flyte_result: WhatprotClassifyV1FlyteResult,
    wp_post_classify_flyte_result: WhatprotPostClassifyV1FlyteResult,
) -> None:
    # Write job folder if specified

    logger.info("write_job_folder_result")

    if config.job:
        job_path = write_job_folder(job_folder=config.job, config_dict=config.to_dict())

        # Write result folders
        wp_pre_flyte_result.save_to_disk(job_path / "wp0_pre")
        wp_classify_flyte_result.save_to_disk(job_path / "wp1_classify")
        wp_post_classify_flyte_result.save_to_disk(job_path / "wp2_post_classify")

        # Write reports
        reports = [
            "whatprot.ipynb",
        ]
        if config.sim_test_job:
            reports.append("whatprot_poi_pr.ipynb")

        gen_utils.write_reports(job_path=job_path, reports=reports)


@task(task_config=task_configs.generate_efs_task_config())
def write_fit_job_folder_result(
    config: WhatprotConfig,
    wp_pre_flyte_result: WhatprotPreV1FlyteResult,
    wp_fit_flyte_result: WhatprotFitV1FlyteResult,
    wp_fit_bootstrap_flyte_result: WhatprotFitBootstrapV1FlyteResult,
) -> None:
    # Write job folder if specified

    logger.info("write_job_folder_result")

    if config.job:
        job_path = write_job_folder(job_folder=config.job, config_dict=config.to_dict())

        # Write result folders
        wp_pre_flyte_result.save_to_disk(job_path / "wp0_pre")
        wp_fit_flyte_result.save_to_disk(job_path / "wp1a_fit")
        wp_fit_bootstrap_flyte_result.save_to_disk(job_path / "wp1b_fit_bootstrap")

        # Write reports
        reports = [
            "whatprot_fit.ipynb",
        ]

        gen_utils.write_reports(job_path=job_path, reports=reports)


# This seems really silly, but you can't unpack the config in the workflow because
# at that point the config is a promise, not an object with referenceable attributes.
# Is this conversion step really necessary?  The WhatprotConfig is the object edited
# by a user, and the idea is that we might need to do some special processing here to
# arrive at params we're going to pass further to whatprot-related tasks.  We could
# have the fetch_sim_train_flyte_result task also return the params I suppose, which
# would save 1 pod spinning up.  As this is written, a pod is spun up to make these
# few simple assignments below!  I'm just following the pattern that has been
# established.
@task
def extract_params(
    config: WhatprotConfig,
) -> WhatprotV1Params:
    wp_params = WhatprotV1Params(
        wp_numgenerate=config.wp_numgenerate,
        wp_neighbors=config.wp_neighbors,
        wp_sigma=config.wp_sigma,
        wp_passthrough=config.wp_passthrough,
        wp_hmmprune=config.wp_hmmprune,
        wp_stoppingthreshold=config.wp_stoppingthreshold,
        wp_maxruntime_minutes=config.wp_maxruntime_minutes,
        wp_numbootstrap=config.wp_numbootstrap,
        wp_confidenceinterval=config.wp_confidenceinterval,
        filter_params=config.filter_params,
        filter_reject_thresh_all_cycles=config.filter_reject_thresh_all_cycles,
    )
    return wp_params


@workflow
def whatprot_classify_workflow(
    config: WhatprotConfig,
) -> tuple[
    WhatprotPreV1FlyteResult,
    WhatprotClassifyV1FlyteResult,
    WhatprotPostClassifyV1FlyteResult,
]:

    wp_params = extract_params(config=config)

    prep_train_flyte_result, sim_train_flyte_result = fetch_sim_train_flyte_results(
        config=config
    )
    sim_test_flyte_result = fetch_sim_test_flyte_result(config=config)
    ims_import_flyte_result, sigproc_flyte_result = fetch_sigproc_flyte_results(
        config=config
    )

    wp_pre_flyte_result = whatprot_pre_flyte_task(
        whatprot_params=wp_params,
        prep_train_flyte_result=prep_train_flyte_result,
        sim_train_flyte_result=sim_train_flyte_result,
        sim_test_flyte_result=sim_test_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
    )

    wp_classify_flyte_result = whatprot_classify_flyte_task(
        whatprot_params=wp_params, wp_pre_flyte_result=wp_pre_flyte_result
    )

    wp_post_classify_flyte_result = whatprot_post_classify_flyte_task(
        whatprot_params=wp_params,
        wp_pre_flyte_result=wp_pre_flyte_result,
        wp_classify_flyte_result=wp_classify_flyte_result,
        sim_train_flyte_result=sim_train_flyte_result,
    )

    write_classify_job_folder_result(
        config=config,
        wp_pre_flyte_result=wp_pre_flyte_result,
        wp_classify_flyte_result=wp_classify_flyte_result,
        wp_post_classify_flyte_result=wp_post_classify_flyte_result,
    )

    return wp_pre_flyte_result, wp_classify_flyte_result, wp_post_classify_flyte_result


@workflow
def whatprot_classify_workflow_xl(
    config: WhatprotConfig,
) -> tuple[
    WhatprotPreV1FlyteResult,
    WhatprotClassifyV1FlyteResult,
    WhatprotPostClassifyV1FlyteResult,
]:

    wp_params = extract_params(config=config)

    prep_train_flyte_result, sim_train_flyte_result = fetch_sim_train_flyte_results(
        config=config
    )
    sim_test_flyte_result = fetch_sim_test_flyte_result(config=config)
    ims_import_flyte_result, sigproc_flyte_result = fetch_sigproc_flyte_results(
        config=config
    )

    wp_pre_flyte_result = whatprot_pre_flyte_task_xl(
        whatprot_params=wp_params,
        prep_train_flyte_result=prep_train_flyte_result,
        sim_train_flyte_result=sim_train_flyte_result,
        sim_test_flyte_result=sim_test_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
    )

    wp_classify_flyte_result = whatprot_classify_flyte_task_xl(
        whatprot_params=wp_params, wp_pre_flyte_result=wp_pre_flyte_result
    )

    wp_post_classify_flyte_result = whatprot_post_classify_flyte_task(
        whatprot_params=wp_params,
        wp_pre_flyte_result=wp_pre_flyte_result,
        wp_classify_flyte_result=wp_classify_flyte_result,
        sim_train_flyte_result=sim_train_flyte_result,
    )

    write_classify_job_folder_result(
        config=config,
        wp_pre_flyte_result=wp_pre_flyte_result,
        wp_classify_flyte_result=wp_classify_flyte_result,
        wp_post_classify_flyte_result=wp_post_classify_flyte_result,
    )

    return wp_pre_flyte_result, wp_classify_flyte_result, wp_post_classify_flyte_result


@workflow
def whatprot_fit_workflow(
    config: WhatprotConfig,
) -> tuple[
    WhatprotPreV1FlyteResult,
    WhatprotFitV1FlyteResult,
    WhatprotFitBootstrapV1FlyteResult,
]:

    wp_params = extract_params(config=config)

    prep_train_flyte_result, sim_train_flyte_result = fetch_sim_train_flyte_results(
        config=config
    )
    sim_test_flyte_result = fetch_sim_test_flyte_result(config=config)
    ims_import_flyte_result, sigproc_flyte_result = fetch_sigproc_flyte_results(
        config=config
    )

    wp_pre_flyte_result = whatprot_pre_flyte_task(
        whatprot_params=wp_params,
        prep_train_flyte_result=prep_train_flyte_result,
        sim_train_flyte_result=sim_train_flyte_result,
        sim_test_flyte_result=sim_test_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
    )

    # Note the that next two tasks can run in parallel

    wp_fit_flyte_result = whatprot_fit_flyte_task(
        whatprot_params=wp_params, wp_pre_flyte_result=wp_pre_flyte_result
    )

    wp_fit_bootstrap_flyte_result = whatprot_fit_bootstrap_flyte_task(
        whatprot_params=wp_params, wp_pre_flyte_result=wp_pre_flyte_result
    )

    write_fit_job_folder_result(
        config=config,
        wp_pre_flyte_result=wp_pre_flyte_result,
        wp_fit_flyte_result=wp_fit_flyte_result,
        wp_fit_bootstrap_flyte_result=wp_fit_bootstrap_flyte_result,
    )

    return wp_pre_flyte_result, wp_fit_flyte_result, wp_fit_bootstrap_flyte_result


def generate(config: WhatprotConfig) -> gen_config.GenerateResult:
    """Generate a whatprot job from a WhatprotConfig dataclass.

    Args
    ----
    config : WhatprotConfig
        A WhatprotConfig specifying information for one or more runs.

    Returns
    -------
    A gen_config.GenerateResult object describing the runs and static_reports to be executed
    for this job.
    """

    # This doesn't actually even get called in the ControlPanel/Flyte codepath!

    # Flyte doesn't currently have a concept of runs within a job, but this may change
    # when this generate fn takes some kind of permutation information to help users
    # create a series of related jobs (formerly called runs within a job).  Or similar.
    runs = []

    static_reports = ["whatprot"]

    # Reads filter_params from file on EFS ahead of flyte launch.
    config.load_filtering_params()

    return gen_config.GenerateResult(runs=runs, static_reports=static_reports)


generator_classify = gen_config.Generator(
    WhatprotConfig, generate, workflow=whatprot_classify_workflow
)
generator_classify_xl = gen_config.Generator(
    WhatprotConfig, generate, workflow=whatprot_classify_workflow_xl
)
generator_fit = gen_config.Generator(
    WhatprotConfig, generate, workflow=whatprot_fit_workflow
)
