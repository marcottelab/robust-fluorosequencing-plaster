from dataclasses import dataclass
from typing import List, Tuple

from dataclasses_json import DataClassJsonMixin
from flytekit import task, workflow
from flytekit.types.directory import FlyteDirectory
from munch import Munch

# initialize logging as early as possible in an attempt to capture logs early
from plaster import env

env.configure_logging()

from plaster.genv2 import gen_utils, help_texts
from plaster.genv2.gen_config import BaseGenConfig, GenerateResult, Generator, gen_field
from plaster.genv2.legacy_utils import task_templates
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import ImsImportFlyteResult
from plaster.run.ims_import.ims_import_task import ims_import_flyte_task
from plaster.run.sigproc_v2 import sigproc_v2_common
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2FlyteResult
from plaster.run.sigproc_v2.sigproc_v2_task import sigproc_v2_calibrate_flyte_task
from plaster.tools.flyte import task_configs


@dataclass
class SigprocCalibSource(DataClassJsonMixin):
    path: str = gen_field(help="Path to the source data", download=True, path=True)
    movie: bool = gen_field(default=False, help=help_texts.MOVIE_HELP)
    dst_ch_i_to_src_ch_i: List[int] = gen_field(
        default_factory=list, help="Mapping of destination channel to source channel"
    )


@dataclass
class SigprocV2CalibConfig(BaseGenConfig, DataClassJsonMixin):
    source: SigprocCalibSource = gen_field(help="Configuration for the source data")


def generate_ims_import_params(config: SigprocV2CalibConfig) -> ImsImportParams:
    return ImsImportParams(
        is_movie=config.source.movie,
        dst_ch_i_to_src_ch_i=config.source.dst_ch_i_to_src_ch_i,
    )


def generate_sigproc_v2_params(config: SigprocV2CalibConfig) -> SigprocV2Params:
    return SigprocV2Params(
        priors=None,
        mode=sigproc_v2_common.SIGPROC_V2_ILLUM_CALIB,
    )


@task
def extract_params(
    config: SigprocV2CalibConfig,
) -> Tuple[ImsImportParams, SigprocV2Params, FlyteDirectory, str]:
    ims_import_params = generate_ims_import_params(config)
    sigproc_v2_params = generate_sigproc_v2_params(config)
    src_dir = FlyteDirectory(config.source.path)
    return ims_import_params, sigproc_v2_params, src_dir, config.job


@task(task_config=task_configs.generate_efs_task_config())
def write_job_folder_result(
    job_folder: str,
    config: SigprocV2CalibConfig,
    ims_import_result: ImsImportFlyteResult,
    sigproc_v2_result: SigprocV2FlyteResult,
) -> None:
    # Write job folder if specified
    if job_folder:
        job_path = gen_utils.write_job_folder(
            job_folder=job_folder, config_dict=config.to_dict()
        )

        # Write result folders
        ims_import_result.save_to_disk(job_path / "ims_import")
        sigproc_v2_result.save_to_disk(job_path / "sigproc_v2")

        # Write reports
        reports = ["sigproc_calib.ipynb"]

        gen_utils.write_reports(job_path=job_path, reports=reports)


@workflow
def sigproc_v2_calibrate_workflow(
    config: SigprocV2CalibConfig,
) -> SigprocV2FlyteResult:
    """
    Sigproc V2 Calibration workflow.

    Args:
        config(SigprocV2Config): Sigproc V2 configuration.
    Returns:
        SigprocV2FlyteResult: Sigproc V2 calibration result.
    """
    ims_import_params, sigproc_v2_params, src_dir, job_folder = extract_params(
        config=config
    )

    ims_import_flyte_result = ims_import_flyte_task(
        src_dir=src_dir, ims_import_params=ims_import_params
    )
    sigproc_v2_flyte_result = sigproc_v2_calibrate_flyte_task(
        sigproc_v2_params=sigproc_v2_params,
        ims_import_flyte_result=ims_import_flyte_result,
    )

    write_job_folder_result(
        job_folder=job_folder,
        config=config,
        ims_import_result=ims_import_flyte_result,
        sigproc_v2_result=sigproc_v2_flyte_result,
    )

    return sigproc_v2_flyte_result


def generate(config: SigprocV2CalibConfig):
    runs = []

    ims_import_task = task_templates.ims_import(
        config.source.path,
        is_movie=config.source.movie,
        dst_ch_i_to_src_ch_i=config.source.dst_ch_i_to_src_ch_i,
    )

    sigproc_v2_calib_task = task_templates.sigproc_v2_calib(
        mode=sigproc_v2_common.SIGPROC_V2_ILLUM_CALIB
    )

    run = Munch(
        run_name=f"sigproc_v2_calib",
        **ims_import_task,
        **sigproc_v2_calib_task,
    )

    runs += [run]

    static_reports = ["sigproc_calib"]

    return GenerateResult(
        runs=runs,
        static_reports=static_reports,
    )


generator = Generator(
    config=SigprocV2CalibConfig,
    generate=generate,
    workflow=sigproc_v2_calibrate_workflow,
)
