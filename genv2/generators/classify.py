from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import flytekit
from dataclasses_json import DataClassJsonMixin
from flytekit import Resources, task, workflow
from flytekit.remote.remote import FlyteRemote

# initialize logging as early as possible in an attempt to capture logs early
from plaster import env

env.configure_logging()
import structlog

logger = structlog.get_logger()

from plaster.genv2 import gen_config
from plaster.genv2.gen_config import BaseGenConfig, gen_field
from plaster.genv2.gen_utils import resolve_job_folder, write_job_folder
from plaster.reports.helpers.report_params import FilterParams
from plaster.run.ims_import.ims_import_result import ImsImportFlyteResult
from plaster.run.rf_train_v2.rf_train_v2_result import RFTrainV2FlyteResult
from plaster.run.rf_v2.rf_v2_params import RFV2Params
from plaster.run.rf_v2.rf_v2_result import RFV2FlyteResult
from plaster.run.rf_v2.rf_v2_task import rf_v2_big_flyte_task, rf_v2_flyte_task
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2FlyteResult
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult
from plaster.tools.flyte import remote, task_configs


@dataclass
class ClassifyConfig(BaseGenConfig, DataClassJsonMixin):
    rf_train_flyte_result: Optional[RFTrainV2FlyteResult] = gen_field(
        default=None,
        help="Directly provided rf_train flyte result for testing purposes.",
        hidden=True,
    )
    rf_train_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the rf_train (VFS) job.",
    )
    rf_train_id: Optional[str] = gen_field(
        default=None,
        help="The Flyte execution ID of the rf_train (VFS) job.",
    )
    sigproc_flyte_result: Optional[SigprocV2FlyteResult] = gen_field(
        default=None,
        help="Directly provided sigproc flyte result for testing purposes.",
        hidden=True,
    )
    sigproc_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the sigproc job.",
    )
    sigproc_id: Optional[str] = gen_field(
        default=None,
        help="The Flyte execution ID of the sigproc job.",
    )
    sim_flyte_result: Optional[SimV3FlyteResult] = gen_field(
        default=None,
        help="Directly provided sim flyte result for testing purposes.",
        hidden=True,
    )
    sim_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the sim job.",
    )
    sim_id: Optional[str] = gen_field(
        default=None,
        help="The Flyte execution ID of the sim job.",
    )

    # Sigproc data can be filtered ahead of input to classifier.  This filtering is
    # currently provided by values in a ReportParams object which is the result of
    # a human tweaking values after looking at radiometry reports.  So this file
    # will live on EFS, and before launching the flyte workflow the generator
    # will load this into the filter_params attribute for inclusion in RFV2Params.
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
        rf_train_present = (
            self.rf_train_flyte_result or self.rf_train_job or self.rf_train_id
        )
        sigproc_present = (
            self.sigproc_flyte_result or self.sigproc_job or self.sigproc_id
        )
        sim_present = self.sim_flyte_result or self.sim_job or self.sim_id
        if not (rf_train_present and (sigproc_present or sim_present)):
            raise ValueError(
                "At least one of the rf_train fields must not be empty, and at least one sigproc or sim field must not be empty."
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
    # task_config like this is required to mount EFS if you want to load results from e.g. job_folder
    task_config=task_configs.generate_efs_task_config(),
    # secrets like this are required if you want to load results via an existing Flyte execution ID
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_rf_train_flyte_result(
    config: ClassifyConfig,
) -> RFTrainV2FlyteResult:
    rf_train_flyte_result = None

    if config.rf_train_flyte_result:
        rf_train_flyte_result = config.rf_train_flyte_result
    elif config.rf_train_id:
        r = remote.fetch_flyte_remote()
        workflow_ex = r.fetch_execution(name=config.rf_train_id)
        r.sync_execution(workflow_ex)
        rf_train_flyte_result: RFTrainV2FlyteResult = workflow_ex.outputs.get(
            "o2", as_type=RFTrainV2FlyteResult
        )
    elif config.rf_train_job:
        rf_train_flyte_result = RFTrainV2FlyteResult.load_from_disk(
            resolve_job_folder(config.rf_train_job) / "rf_train"
        )
    else:
        raise ValueError("At least one rf_train field must not be empty.")

    return rf_train_flyte_result


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_sigproc_flyte_results(
    config: ClassifyConfig,
) -> tuple[Optional[ImsImportFlyteResult], Optional[SigprocV2FlyteResult]]:
    ims_import_flyte_result = None
    sigproc_flyte_result = None

    if config.sigproc_flyte_result:
        sigproc_flyte_result = config.sigproc_flyte_result
    elif config.sigproc_id:
        r = remote.fetch_flyte_remote()
        workflow_ex = r.fetch_execution(name=config.sigproc_id)
        r.sync_execution(workflow_ex)
        ims_import_flyte_result: ImsImportFlyteResult = workflow_ex.outputs.get(
            "o0", as_type=ImsImportFlyteResult
        )
        sigproc_flyte_result: SigprocV2FlyteResult = workflow_ex.outputs.get(
            "o1", as_type=SigprocV2FlyteResult
        )

    elif config.sigproc_job:
        sigproc_flyte_result = SigprocV2FlyteResult.load_from_disk(
            resolve_job_folder(config.sigproc_job) / "sigproc_v2"
        )
        ims_import_flyte_result = ImsImportFlyteResult.load_from_disk(
            resolve_job_folder(config.sigproc_job) / "ims_import"
        )

    return ims_import_flyte_result, sigproc_flyte_result


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_sim_flyte_result(
    config: ClassifyConfig,
) -> Optional[SimV3FlyteResult]:
    sim_flyte_result = None

    if config.sim_flyte_result:
        sim_flyte_result = config.sim_flyte_result
    elif config.sim_id:
        r = remote.fetch_flyte_remote()
        workflow_ex = r.fetch_execution(name=config.sim_id)
        r.sync_execution(workflow_ex)
        sim_flyte_result: SimV3FlyteResult = workflow_ex.outputs.get(
            "o1", as_type=SimV3FlyteResult
        )
    elif config.sim_job:
        sim_flyte_result = SimV3FlyteResult.load_from_disk(
            resolve_job_folder(config.sim_job) / "sim"
        )

    return sim_flyte_result


@task(
    task_config=task_configs.generate_efs_task_config(),
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="8", mem="64Gi"),
)
def write_job_folder_result(
    config: ClassifyConfig,
    rf_classify_flyte_result: RFV2FlyteResult,
) -> None:
    # Write job folder if specified

    if config.job:
        job_path = write_job_folder(job_folder=config.job, config_dict=config.to_dict())

        # Write result folders
        rf_classify_flyte_result.save_to_disk(job_path / "rf_classify")


@task
def extract_params(config: ClassifyConfig) -> RFV2Params:
    params = RFV2Params(
        filter_params=config.filter_params,
        filter_reject_thresh_all_cycles=config.filter_reject_thresh_all_cycles,
    )
    return params


# entities passed between tasks need to be returned into the workflow namespace
# tasks can't contain other tasks
@workflow
def classify_workflow(config: ClassifyConfig) -> RFV2FlyteResult:
    params = extract_params(config=config)
    rf_train_flyte_result = fetch_rf_train_flyte_result(config=config)
    ims_import_flyte_result, sigproc_flyte_result = fetch_sigproc_flyte_results(
        config=config
    )
    sim_flyte_result = fetch_sim_flyte_result(config=config)

    rf_classify_flyte_result = rf_v2_flyte_task(
        rf_params=params,
        rf_train_flyte_result=rf_train_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
        sim_flyte_result=sim_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
    )

    write_job_folder_result(
        config=config,
        rf_classify_flyte_result=rf_classify_flyte_result,
    )
    return rf_classify_flyte_result


@workflow
def classify_big_workflow(config: ClassifyConfig) -> RFV2FlyteResult:
    params = extract_params(config=config)
    rf_train_flyte_result = fetch_rf_train_flyte_result(config=config)
    ims_import_flyte_result, sigproc_flyte_result = fetch_sigproc_flyte_results(
        config=config
    )
    sim_flyte_result = fetch_sim_flyte_result(config=config)

    rf_classify_flyte_result = rf_v2_big_flyte_task(
        rf_params=params,
        rf_train_flyte_result=rf_train_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
        sim_flyte_result=sim_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
    )

    write_job_folder_result(
        config=config,
        rf_classify_flyte_result=rf_classify_flyte_result,
    )
    return rf_classify_flyte_result


def generate(config: ClassifyConfig) -> gen_config.GenerateResult:
    """Generate a classify job from a ClassifyConfig dataclass.

    Args
    ----
    config : ClassifyConfig
        A ClassifyConfig specifying information for one or more runs.

    Returns
    -------
    A gen_config.GenerateResult object describing the runs and static_reports to be executed
    for this job.
    """

    # Flyte doesn't currently have a concept of runs within a job, but this may change
    # when this generate fn takes some kind of permutation information to help users
    # create a series of related jobs (formerly called runs within a job).  Or similar.
    runs = []

    static_reports = ["classify"]

    return gen_config.GenerateResult(runs=runs, static_reports=static_reports)


generator = gen_config.Generator(ClassifyConfig, generate, workflow=classify_workflow)
generator_big = gen_config.Generator(
    ClassifyConfig, generate, workflow=classify_big_workflow
)
