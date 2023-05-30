import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import flytekit
import numpy as np
import yaml
from dataclasses_json import DataClassJsonMixin
from flytekit import Resources, task, workflow
from flytekit.types.directory import FlyteDirectory
from munch import Munch
from plumbum import local

# initialize logging as early as possible in an attempt to capture logs early
from plaster import env

env.configure_logging()

from plaster.genv2 import gen_utils, help_texts
from plaster.genv2.gen_config import (
    BaseGenConfig,
    GenerateResult,
    Generator,
    ValidationResult,
    gen_field,
)
from plaster.genv2.legacy_utils import task_templates
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import ImsImportFlyteResult
from plaster.run.ims_import.ims_import_task import ims_import_flyte_task
from plaster.run.priors import ChannelAlignPrior, Priors
from plaster.run.sigproc_v2 import sigproc_v2_common
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.run.sigproc_v2.sigproc_v2_result import (
    SigprocV2FlyteResult,
    SigprocV2ResultDC,
)
from plaster.run.sigproc_v2.sigproc_v2_task import sigproc_v2_analyze_flyte_task
from plaster.tools.flyte import remote, task_configs
from plaster.tools.utils import dataclass_utils


@dataclass
class SigprocSource(DataClassJsonMixin):
    """
    These configuration options relate to the source data for sigproc.

    As such they are largely fields related to ims_import, but may contain data used later in the
    pipeline but are still semantically related to the source data.
    """

    path: str = gen_field(help="Path to the source data", download=True, path=True)
    n_cycles_limit: int = gen_field(
        default=dataclass_utils.get_field(ImsImportParams, "n_cycles_limit").default,
        help="Limit the number of cycles to process",
    )
    start_cycle: int = gen_field(
        default=dataclass_utils.get_field(ImsImportParams, "start_cycle").default,
        help="The index of the first cycle to process",
    )
    start_field: int = gen_field(
        default=dataclass_utils.get_field(ImsImportParams, "start_field").default,
        help="The index of the first field to process",
    )
    n_fields_limit: int = gen_field(
        default=dataclass_utils.get_field(ImsImportParams, "n_fields_limit").default,
        help="The index of the numbers of fields to process starting at 'start_field'",
    )
    # TODO: improve this help text
    dst_ch_i_to_src_ch_i: List[int] = gen_field(
        default_factory=list, help="Mapping of destination channel to source channel"
    )
    movie: bool = gen_field(default=False, help=help_texts.MOVIE_HELP)


@dataclass
class SigprocV2CalibrationSettings(DataClassJsonMixin):
    """
    These configuration options relate to the calibration for sigproc.

    For the legacy pipeline, calibrations can only be specified using the calibration_job field.

    For the flyte pipeline, calibration can be specified by using exactly one of the following fields:
     - calibration_job
     - calibration_result
     - calibration_id
    """

    self_calib: bool = gen_field(
        default=False,
        help=help_texts.SELF_CALIB_HELP,
    )
    calibration_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="The path to the calibration job.",
    )
    calibration_result: Optional[SigprocV2FlyteResult] = gen_field(
        default=None,
        help="Directly provided calibration result for testing purposes.",
        hidden=True,
    )
    calibration_id: Optional[str] = gen_field(
        default=None,
        help="The Flyte execution ID of the calibration job.",
    )
    # TODO: better help text here
    calib_dst_ch_i_to_src_ch_i: List[int] = gen_field(
        default_factory=list,
        help="The mapping of calibration destination channel to source channel",
    )

    def is_calibration_referenced(self) -> bool:
        """
        Returns True if an external calibration is specified.
        """
        return any((self.calibration_job, self.calibration_result, self.calibration_id))

    def validate(self):
        if self.self_calib:
            if self.is_calibration_referenced() or self.calib_dst_ch_i_to_src_ch_i:
                return ValidationResult(
                    valid=False,
                    message="You can't specify calibration job fields when using self calibration",
                )
        else:
            if not self.is_calibration_referenced():
                return ValidationResult(
                    valid=False,
                    message="You must specify calibration_job when not using self calibration",
                )

            # There should be only one specified calibration
            calibration_references = [
                self.calibration_job,
                self.calibration_result,
                self.calibration_id,
            ]
            if sum(1 for v in calibration_references if v) != 1:
                return ValidationResult(
                    valid=False,
                    message="You must specify exactly one calibration reference",
                )

        return ValidationResult(True)


@dataclass
class SigprocV2Config(BaseGenConfig, DataClassJsonMixin):
    source: SigprocSource = gen_field(help="Configuration for the source data")
    calibration: SigprocV2CalibrationSettings = gen_field(
        help="Configuration for the calibration data"
    )

    ch_aln: List[float] = gen_field(
        default_factory=list,
        help="List of coordinates for channel alignment (x0, y0, x1, y1, ...)",
    )
    ch_for_alignment: int = gen_field(
        default=dataclass_utils.get_field(SigprocV2Params, "ch_for_alignment").default,
        help="Which channel to use for alignment",
    )
    null_hypothesis_precompute: bool = gen_field(
        default=False,
        help="Set to true to enable precomputing the null hypothesis data",
    )


def tasks_for_sigproc_v2(config: SigprocV2Config):
    tasks = {}
    if config.source.path:
        ims_import_task = task_templates.ims_import(
            config.source.path,
            is_movie=config.source.movie,
            n_cycles_limit=config.source.n_cycles_limit,
            start_cycle=config.source.start_cycle,
            dst_ch_i_to_src_ch_i=config.source.dst_ch_i_to_src_ch_i or [],
            start_field=config.source.start_field,
            n_fields_limit=config.source.n_fields_limit,
        )

        calib_priors = None
        if config.calibration.calibration_job is not None:
            calib_src_path = (
                local.path(config.calibration.calibration_job)
                / "sigproc_v2_calib/plaster_output/sigproc_v2"
            )
            calib_result = SigprocV2ResultDC.load_from_folder(
                calib_src_path, prop_list=["calib_priors"]
            )
            calib_priors = calib_result.calib_priors

            if config.calibration.calib_dst_ch_i_to_src_ch_i:
                if not config.source.dst_ch_i_to_src_ch_i:
                    raise Exception(
                        "source.dst_ch_i_to_src_ch_i must be set to use calib_dst_ch_i_to_src_ch_i"
                    )
                assert len(config.calibration.calib_dst_ch_i_to_src_ch_i) == len(
                    config.source.dst_ch_i_to_src_ch_i
                ), "calibration.calib_dst_ch_i_to_src_ch_i must be of same length as source.dst_ch_i_to_src_ch_i if both are used"

                ch_remapped_priors = Priors.copy(calib_priors)
                ch_remapped_priors.delete_ch_specific_records()

                ch_aln_prior = ch_remapped_priors.get_exact(f"ch_aln")
                if ch_aln_prior is not None:
                    ch_aln_prior = ChannelAlignPrior.ch_remap(
                        ch_aln_prior.prior,
                        config.calibration.calib_dst_ch_i_to_src_ch_i,
                    )

                for dst_ch_i, src_ch_i in enumerate(
                    config.calibration.calib_dst_ch_i_to_src_ch_i
                ):

                    def remap(src_key, dst_key):
                        prior = calib_priors.get_exact(src_key)
                        if prior is not None:
                            ch_remapped_priors.add(
                                dst_key, prior.prior, "remapped channel in gen"
                            )

                    remap(f"reg_illum.ch_{src_ch_i}", f"reg_illum.ch_{dst_ch_i}")
                    remap(f"reg_psf.ch_{src_ch_i}", f"reg_psf.ch_{dst_ch_i}")

                calib_priors = ch_remapped_priors

        ch_aln = None
        if config.ch_aln:
            ch_aln = np.array(config.ch_aln)
            assert ch_aln.shape[0] % 2 == 0
            ch_aln = ch_aln.reshape((-1, 2))

        sigproc_v2_task = task_templates.sigproc_v2_analyze(
            calib_priors=calib_priors,
            self_calib=config.calibration.self_calib,
            ch_aln=ch_aln,
            ch_for_alignment=config.ch_for_alignment,
            null_hypothesis_precompute=config.null_hypothesis_precompute,
        )

        tasks = Munch(**ims_import_task, **sigproc_v2_task)

    return tasks


def generate_ims_import_params(config: SigprocV2Config) -> ImsImportParams:
    return ImsImportParams(
        is_movie=config.source.movie,
        n_cycles_limit=config.source.n_cycles_limit,
        start_cycle=config.source.start_cycle,
        dst_ch_i_to_src_ch_i=config.source.dst_ch_i_to_src_ch_i or [],
        start_field=config.source.start_field,
        n_fields_limit=config.source.n_fields_limit,
    )


def fetch_calibration_result(execution_id: str) -> SigprocV2ResultDC:
    r = remote.fetch_flyte_remote()
    workflow_ex = r.fetch_execution(name=execution_id)
    r.sync_execution(workflow_ex)
    sigproc_v2_flyte_result: SigprocV2FlyteResult = workflow_ex.outputs.get(
        "o0", as_type=SigprocV2FlyteResult
    )
    return sigproc_v2_flyte_result.load_result()


def generate_sigproc_v2_params(config: SigprocV2Config) -> SigprocV2Params:
    calib_priors = None
    if config.calibration.is_calibration_referenced():
        if config.calibration.calibration_result:
            calib_result = config.calibration.calibration_result.load_result()
        elif config.calibration.calibration_id:
            calib_result: SigprocV2ResultDC = fetch_calibration_result(
                config.calibration.calibration_id
            )
        elif config.calibration.calibration_job:
            calib_result = SigprocV2FlyteResult.load_from_disk(
                gen_utils.resolve_job_folder(str(config.calibration.calibration_job))
                / "sigproc_v2"
            ).load_result()

        calib_priors = calib_result.calib_priors

        if config.calibration.calib_dst_ch_i_to_src_ch_i:
            if not config.source.dst_ch_i_to_src_ch_i:
                raise Exception(
                    "source.dst_ch_i_to_src_ch_i must be set to use calib_dst_ch_i_to_src_ch_i"
                )
            assert len(config.calibration.calib_dst_ch_i_to_src_ch_i) == len(
                config.source.dst_ch_i_to_src_ch_i
            ), "calibration.calib_dst_ch_i_to_src_ch_i must be of same length as source.dst_ch_i_to_src_ch_i if both are used"

            ch_remapped_priors = Priors.copy(calib_priors)
            ch_remapped_priors.delete_ch_specific_records()

            ch_aln_prior = ch_remapped_priors.get_exact(f"ch_aln")
            if ch_aln_prior is not None:
                ch_aln_prior = ChannelAlignPrior.ch_remap(
                    ch_aln_prior.prior,
                    config.calibration.calib_dst_ch_i_to_src_ch_i,
                )

            for dst_ch_i, src_ch_i in enumerate(
                config.calibration.calib_dst_ch_i_to_src_ch_i
            ):

                def remap(src_key, dst_key):
                    prior = calib_priors.get_exact(src_key)
                    if prior is not None:
                        ch_remapped_priors.add(
                            dst_key, prior.prior, "remapped channel in gen"
                        )

                remap(f"reg_illum.ch_{src_ch_i}", f"reg_illum.ch_{dst_ch_i}")
                remap(f"reg_psf.ch_{src_ch_i}", f"reg_psf.ch_{dst_ch_i}")

            calib_priors = ch_remapped_priors

    ch_aln = None
    if config.ch_aln is not None:
        ch_aln = np.array(config.ch_aln)
        assert ch_aln.shape[0] % 2 == 0
        ch_aln = ch_aln.reshape((-1, 2))
        ch_aln = ch_aln.tolist()

    if calib_priors is None:
        calib_priors = Priors()

    return SigprocV2Params(
        priors=calib_priors,
        mode=sigproc_v2_common.SIGPROC_V2_INSTRUMENT_ANALYZE,
        self_calib=config.calibration.self_calib,
        ch_aln_override=ch_aln or [],
        ch_for_alignment=config.ch_for_alignment,
        null_hypothesis_precompute=config.null_hypothesis_precompute,
    )


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def extract_params(
    config: SigprocV2Config,
) -> Tuple[ImsImportParams, SigprocV2Params, FlyteDirectory, str]:
    ims_import_params = generate_ims_import_params(config)
    sigproc_v2_params = generate_sigproc_v2_params(config)
    src_dir = FlyteDirectory(config.source.path)

    return ims_import_params, sigproc_v2_params, src_dir, config.job


@task(
    task_config=task_configs.generate_efs_task_config(),
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="8", mem="64Gi"),
)
def write_job_folder_result(
    job_folder: str,
    config: SigprocV2Config,
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

        # Get n_channels which is needed to create default parameterization
        # for reports, and we want to write this file alongside reports.
        results = yaml.safe_load(open(job_path / "sigproc_v2" / "result.yaml"))
        n_channels = results["n_channels"] if "n_channels" in results else 1

        # Write reports
        reports = [
            "sigproc_fields.ipynb",
            "sigproc_radiometry.ipynb",
            "sigproc_fret.ipynb",
            "sigproc_chcolo.ipynb",
            "params.ipynb",
        ]

        gen_utils.write_reports(job_path=job_path, reports=reports)
        gen_utils.write_default_sigproc_report_params(
            job_path=job_path, n_channels=n_channels
        )


@workflow
def sigproc_v2_analyze_workflow(
    config: SigprocV2Config,
) -> tuple[ImsImportFlyteResult, SigprocV2FlyteResult]:
    """
    Sigproc V2 Analysis workflow.

    Args:
        config(SigprocV2Config): Sigproc V2 configuration.
    Returns:
        SigprocV2FlyteResult: Sigproc V2 analysis result.
    """
    ims_import_params, sigproc_v2_params, src_dir, job_folder = extract_params(
        config=config
    )

    ims_import_flyte_result = ims_import_flyte_task(
        src_dir=src_dir, ims_import_params=ims_import_params
    )
    sigproc_v2_flyte_result = sigproc_v2_analyze_flyte_task(
        sigproc_v2_params=sigproc_v2_params,
        ims_import_flyte_result=ims_import_flyte_result,
    )

    write_job_folder_result(
        job_folder=job_folder,
        config=config,
        ims_import_result=ims_import_flyte_result,
        sigproc_v2_result=sigproc_v2_flyte_result,
    )

    return ims_import_flyte_result, sigproc_v2_flyte_result


def generate(config: SigprocV2Config):
    runs = []

    sigproc_tasks = tasks_for_sigproc_v2(config)

    run = Munch(
        run_name=f"sigproc_v2",
        **sigproc_tasks,
    )

    runs += [run]

    static_reports = []

    # if config.null_hypothesis_precompute:
    #    static_reports += ["chcolo"]

    static_reports += [
        "sigproc_fields",
        "sigproc_radiometry",
        "sigproc_fret",
        "sigproc_chcolo",
    ]

    return GenerateResult(runs=runs, static_reports=static_reports)


generator = Generator(SigprocV2Config, generate, workflow=sigproc_v2_analyze_workflow)
