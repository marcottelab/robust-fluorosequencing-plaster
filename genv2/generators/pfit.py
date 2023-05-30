from plaster import env

env.configure_logging()
import structlog

logger = structlog.get_logger()

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from dataclasses_json import DataClassJsonMixin
from flytekit import dynamic, map_task, task, workflow

from plaster.genv2.gen_config import BaseGenConfig, GenerateResult, Generator, gen_field
from plaster.genv2.gen_utils import resolve_job_folder, write_job_folder, write_reports
from plaster.genv2.generators.vfs import VFSConfig, vfs_big_workflow, vfs_workflow
from plaster.genv2.generators.whatprot import whatprot_fit_workflow
from plaster.reports.helpers.params import ReportParams
from plaster.reports.helpers.report_params import FilterParams
from plaster.run.ims_import.ims_import_result import ImsImportFlyteResult
from plaster.run.pfit_v1 import pfit_v1_params, pfit_v1_tasks
from plaster.run.pfit_v1.pfit_v1_params import ParameterEstimationParams, ParamInfo
from plaster.run.pfit_v1.pfit_v1_result import (
    FlyteParameterEstimationCheckResult,
    FlyteParameterEstimationFitResult,
    ParameterEstimationCheckOneResult,
    ParameterEstimationCheckResult,
    ParameterEstimationFitOneResult,
    ParameterEstimationFitResult,
)
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.rf_train_v2.rf_train_v2_result import RFTrainV2FlyteResult
from plaster.run.rf_v2.rf_v2_params import RFV2Params
from plaster.run.rf_v2.rf_v2_result import RFV2FlyteResult
from plaster.run.rf_v2.rf_v2_task import rf_v2_big_flyte_task, rf_v2_flyte_task
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2FlyteResult
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult
from plaster.run.whatprot_v1.whatprot_v1_result import (
    WhatprotFitBootstrapV1FlyteResult,
    WhatprotFitV1FlyteResult,
    WhatprotPreV1FlyteResult,
)
from plaster.tools.flyte import task_configs

# -------------------- modules/parameter_estimation/v1/config.py --------------------


@dataclass
class ParameterEstimationConfig(BaseGenConfig):
    # This is currently not used because our fit workflow starts with DIRECT fitter
    # which does not take starting conditions.  It might conceivably be used if we
    # decide to allow other fitters to be used for the initial fit.
    x0: Optional[list[float]] = gen_field(
        default=None,
        help="Starting point. None means to use a random starting point",
    )

    # 'defaults' is used to hold parameters fixed at specific values, or set upper
    # and lower bounds on the parameters.  It should probably be renamed to something
    # more appropriate, like "param_infos", or "param_constraints"
    #
    defaults: Optional[list[ParamInfo]] = gen_field(
        default=None,
        help="Parameter search restrictions. Default=None will search all parameters in range [0, 1] starting from Plaster's defaults",
    )

    # TODO: this is currently used for the number of 'bootstrap' rounds
    # performed on resampled data being fit. It is also used by the 'check'
    # workflow which I've not looked as closely yet, but appears to perform
    # this many fits, also on resampled data.  So check appears to be exactly
    # like bootstrapping, except that it starts at plaster defaults for x0
    # instead of the output of DIRECT.  But then it does n_reps fits on resampled
    # data using Powell, exactly like bootstrapping does.  So I'm not clear
    # without looking closely at the 'check' report what advantage it has over
    # the bootstrap info that comes out of the fit workflow.
    #
    # In any case, maybe this should become n_bootstraps.  I'm making this 100
    # because for bootstrapping at least it is done in parallel and is quite fast.
    # And we want that many to establish confidence intervals.
    #
    #
    # Precheck fits each parameter in isolation, holding all others to the Plaster
    # defaults. Each each parameter: n_reps minimize(Powell, x0=Random) calls
    # are made. If a fitted parameter has a large variance when fitted in isolation,
    # consider holding some parameters fixed.
    #
    #
    n_reps: int = gen_field(
        default=100,
        help="Number of bootstrap rounds in fitting, and number of reps in check.",
    )

    default_n_samples: int = gen_field(
        default=100_000,
        help="Number of dye tracks generated in evaluation function.",
    )

    sigproc_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the sigproc job whose data you want to fit parameters to.",
    )

    sim_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the simulation job whose data you want to fit parameters to.",
    )

    # You can specify either an already-run vfs_job, or a config in vfs_config to run
    # the vfs as part of the param_fit workflow.
    vfs_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the VFS job that trained a classifier to classify radiometry to spanning dyetracks.",
    )
    vfs_config: Optional[VFSConfig] = gen_field(
        default=None,
        help="Config to run a VFS job that trains a classifier to classify radiometry to spanning dyetracks.",
    )

    # This is used by KV's parameter_estimation_fit workflow.
    classify_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the classify job that has classified the input radiometry using the classifier trained with vfs_job.",
    )

    filter_params_yaml_path: Optional[str] = gen_field(
        default=None,
        path=True,
        help="path to yaml file holding filtering parameters if fitting sigproc data.",
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
        sim_present = self.sim_job
        sigproc_present = self.sigproc_job
        filter_present = self.filter_params_yaml_path

        if sim_present and sigproc_present:
            raise ValueError(
                "Only specify one of sim_job or sigproc_job for data to fit parameters to."
            )

        if not sim_present and not sigproc_present:
            raise ValueError(
                "You must specify one of sim_job or sigproc_job for data to fit parameters to."
            )

        if sigproc_present and not filter_present:
            raise ValueError(
                "You must specify filter_params_yaml_path when fitting sigproc data."
            )

        if self.vfs_job and self.vfs_config:
            raise ValueError("Only specifiy one of vfs_job or vfs_config")

        if not self.vfs_job and not self.vfs_config:
            raise ValueError("You must specify either a vfs_job or a vfs_config")

    def load_filtering_params(self):
        # TODO: this is identical to code in whatprot classify/fit, and will probably be used
        # elsewhere ahead of other classification.  Maybe move this somewhere to DRY
        if self.filter_params_yaml_path:
            path = resolve_job_folder(str(self.filter_params_yaml_path))
            if not path.exists():
                logger.error(
                    "params not found at specified filter_params_yaml_path", path=path
                )
                raise FileExistsError()

            self.filter_params = FilterParams.load_from_path(path)

            # set this to actual path for reference in results: what path was actually loaded
            # as a result of resolve_job_folder
            self.filter_params_yaml_path = str(path)


from plaster.genv2.generators.sim import Marker, SeqParams, SimConfig
from plaster.run.prep_v2.prep_v2_params import Protein


@dataclass
class ParamFitEasyConfig(BaseGenConfig):
    # Require the minimal possible information to do parameter fitting
    # with both Erisyon and Whatprot methods and view outputs/comparison
    # of each.

    protein: Protein = gen_field(
        default=None,
        help="The single peptide whose data is being fit.  E.g. Protein(name='jsp234',sequence='GKAGKAGAY')",
    )

    markers: list[Marker] = gen_field(
        default=None,
        help="Defines the AAs being labeled and basic params like gain_mu, gain_sigma, and bg_mu. "
        "If you are fitting sigproc data, those params will be taken from the _report_params.yaml, "
        "but you still must define the name of each AA being labeled.",
    )

    seq_params: SeqParams = gen_field(
        default=None,
        help="Defines basic sequencing params.  Only required param is n_edmans.",
    )

    sigproc_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the sigproc job whose data you want to fit parameters to. _report_params.yaml will be found here as well.",
    )

    filter_params_yaml_path: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to yaml file holding filtering parameters if fitting sigproc data.  This is optional: if not "
        "specified, the _report_params.yaml file in the sigproc job _reports folder will be used.",
    )

    filter_params: Optional[FilterParams] = gen_field(
        default=None,
        hidden=True,
        help="This is generated programmatically by loading from filter_params_yaml_path.",
    )

    sim_job: Optional[str] = gen_field(
        default=None,
        path=True,
        help="Path to the simulation job whose data you want to fit parameters to.",
    )

    # For holding params fixed or within specified bounds:
    param_constraints: Optional[list[ParamInfo]] = gen_field(
        default_factory=list,
        help="List of params to hold fixed or to specified boundaries during fit",
    )

    n_bootstrap: Optional[int] = gen_field(
        default=100,
        help="Number of bootstrap fits for both whatprot and pfit",
    )

    # I am exposing these next three so that it's easy during development to play with these.
    # It isn't expected that normal users override these.
    n_samples_default: Optional[int] = gen_field(
        default=50000,
        help="How many dyetrack samples to generate at each param location to match true distribution.",
    )

    n_samples_train: Optional[int] = gen_field(
        default=10000,
        help="How many samples per spanning-dyetrack to create in the sim used to train the classifier that classifies input radiometry to dyetracks.",
    )

    wp_maxruntime_minutes: Optional[int] = gen_field(
        default=15,
        help="Timeout value for whatprot per fit - if still fitting at this many minutes, stop and use current best-fit value.",
    )

    def set_initial_params_from_contraints(self):
        # Set the marker params and seq params based on constraints so that whatprot
        # will respect them.  Note that whatprot currently only supports holding values
        # fixed at 0 or 1, so we'll make a choice based on the value of the param.

        if not self.param_constraints:
            return

        # TODO: this only implements is_fixed constraints, which we check for
        # in __post_init__()

        d = {p.name: p.initial_value for p in self.param_constraints}
        d_wp = {k: 0.0 if v < 0.5 else 1.0 for k, v in d.items()}
        seqp_names = [
            "p_detach",
            "p_initial_block",
            "p_cyclic_block",
            "p_edman_failure",
        ]
        mark_names = ["p_dud", "p_bleach"]
        constrained_params = d.keys()
        for p in seqp_names:
            if p in constrained_params:
                logger.info(f"{p} is_fixed to {d[p]} will be {d_wp[p]} for whatprot")
                setattr(self.seq_params, p, d_wp[p])

        for i, m in enumerate(self.markers):
            for p in mark_names:
                p_ch = p + "_ch" + str(i)
                if p_ch in constrained_params:
                    logger.info(
                        f"{p_ch} is_fixed to {d[p_ch]} will be {d_wp[p_ch]} for whatprot"
                    )
                    setattr(m, p, d_wp[p_ch])

    def __post_init__(self):
        if self.sim_job and self.sigproc_job:
            raise ValueError(
                "Only specify one of sim_job or sigproc_job for data to fit parameters to."
            )

        if not self.sim_job and not self.sigproc_job:
            raise ValueError(
                "You must specify one of sim_job or sigproc_job for data to fit parameters to."
            )

        # TODO: further validate param_constraints somehow?
        for p in self.param_constraints:
            if not p.is_fixed:
                raise NotImplementedError(
                    "Bounds are not yet implemented for param_fit_easy, only is_fixed."
                )

        self.set_initial_params_from_contraints()

        if self.sigproc_job:
            # We want to do some validate here, which is what is typically occuring in __post_init__(),
            # but maybe we need to rethink this.  There are classes that have validate() fns, and
            # maybe that is better.  An issue here is that when this config object is serialized
            # and passed to tasks or subworkflows, this fn will get called *again* as the object
            # is constructed in the new context, on a k8s node.  In this case, unless the task
            # is marked with an efs config, it won't be able to access jobs_folder.  So we
            # need to take care to write logic such that any load happens only once, or
            # give those workflows/tasks access to EFS as appropriate.
            sigproc_path = resolve_job_folder(self.sigproc_job)
            if (
                not self.filter_params
            ):  # If the filter_params have not already been loaded...
                if (
                    not self.filter_params_yaml_path
                ):  # If a specific params path hasn't been given explicitly
                    self.filter_params_yaml_path = (
                        str(  # Then default to the location within the sigproc report.
                            sigproc_path / "_reports" / "_report_params.yaml"
                        )
                    )
                # Note that all of this below requires access to EFS.  The original idea was that this
                # happens once when the config object is created, either from a notebook, or from the
                # web app, in both cases EFS is avail.  But see comment above -- this fn will get called
                # each time this object is recreated via serialization, so our logic above takes care
                # to only run this code if the self.filter_params have not already been loaded.

                # This will raise if any required filtering params are not present:
                self.filter_params = FilterParams.load_from_path(
                    path=self.filter_params_yaml_path, check_all_present=True
                )
                # But we also should check that the gain_mu, gain_sigma, and bg_mu are present,
                # and we could at this time override those values in the markers if they are.
                # TODO:
                # Note that this code, as written, does not guarantee that a person has edited
                # and blessed these values.  To do that, we need to check that the param is being
                # loaded from the _report_params.yaml and not from the default setting.
                report_params = ReportParams.from_yaml(sigproc_path)
                bg_sigma = report_params.get("bg_sigma_per_channel")
                gain_mu = report_params.get("gain_mu_per_channel")
                gain_sigma = report_params.get("gain_sigma_per_channel")
                lens = [len(bg_sigma), len(gain_mu), len(gain_sigma), len(self.markers)]
                if len(set(lens)) != 1:
                    raise ValueError(
                        "The number of channels for gain_sigma and related does not match the number of channels implied in your list of markers."
                    )
                for i, m in enumerate(self.markers):
                    m.bg_sigma = float(bg_sigma[i])
                    m.gain_mu = float(gain_mu[i])
                    m.gain_sigma = (
                        float(gain_sigma[i]) * m.gain_mu
                    )  # convert from sigma as fractional mu to absolute sigma
                    # defaults are being used for p_dud and p_bleach


# -------------------- modules/parameter_estimation/v1/flyte_tasks.py --------------------


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_vfs_train_flyte_results(
    config: ParameterEstimationConfig,
) -> tuple[PrepV2FlyteResult, SimV3FlyteResult, RFTrainV2FlyteResult]:
    prep_train_flyte_result = None
    sim_train_flyte_result = None
    rf_train_flyte_result = None

    if config.vfs_job:
        prep_train_flyte_result = PrepV2FlyteResult.load_from_disk(
            resolve_job_folder(config.vfs_job) / "prep"
        )
        sim_train_flyte_result = SimV3FlyteResult.load_from_disk(
            resolve_job_folder(config.vfs_job) / "sim"
        )
        rf_train_flyte_result = RFTrainV2FlyteResult.load_from_disk(
            resolve_job_folder(config.vfs_job) / "rf_train"
        )
    return prep_train_flyte_result, sim_train_flyte_result, rf_train_flyte_result


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_sim_test_flyte_result(
    config: ParameterEstimationConfig,
) -> Optional[SimV3FlyteResult]:
    sim_test_flyte_result = None
    if config.sim_job:
        sim_test_flyte_result = SimV3FlyteResult.load_from_disk(
            resolve_job_folder(config.sim_job) / "sim"
        )
    return sim_test_flyte_result


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def fetch_sigproc_flyte_results(
    config: ParameterEstimationConfig,
) -> tuple[Optional[ImsImportFlyteResult], Optional[SigprocV2FlyteResult]]:
    ims_import_flyte_result = None
    sigproc_flyte_result = None
    if config.sigproc_job:
        ims_import_flyte_result = ImsImportFlyteResult.load_from_disk(
            resolve_job_folder(config.sigproc_job) / "ims_import"
        )
        sigproc_flyte_result = SigprocV2FlyteResult.load_from_disk(
            resolve_job_folder(config.sigproc_job) / "sigproc_v2"
        )
    return ims_import_flyte_result, sigproc_flyte_result


@task(task_config=task_configs.generate_efs_task_config())
def write_check_job_folder_result(
    config: ParameterEstimationConfig,
    check_flyte_result: FlyteParameterEstimationCheckResult,
) -> None:
    if config.job:
        job_path = write_job_folder(job_folder=config.job, config_dict=config.to_dict())

        check_flyte_result.save_to_disk(job_path)

        reports = [
            "parameter_estimation_precheck.ipynb",
        ]

        write_reports(job_path=job_path, reports=reports)


@task(task_config=task_configs.generate_efs_task_config())
def write_fit_job_folder_result(
    config: ParameterEstimationConfig,
    fit_flyte_result: FlyteParameterEstimationFitResult,
    clean: bool = True,
) -> None:
    if config.job:
        job_path = write_job_folder(
            job_folder=config.job, config_dict=config.to_dict(), clean=clean
        )

        fit_flyte_result.save_to_disk(job_path)

        reports = [
            "param_fit_easy.ipynb",
        ]

        write_reports(job_path=job_path, reports=reports)


# param_fit_easy uses subworkflows that write to the job_folder
# as they do their work, so we want to clean the job folder up front.
# TODO: survey does this same thing, refactor that to take a path
# and DRY.
import shutil

from plaster.genv2 import gen_utils


@task(task_config=task_configs.generate_efs_task_config())
def clean_job_folder(config: BaseGenConfig) -> None:
    path = gen_utils.resolve_job_folder(config.job)
    logger.info("cleaning job_folder", job_folder=path, t=type(path))
    if path.exists():
        shutil.rmtree(path)


@task(task_config=task_configs.generate_efs_task_config())
def write_param_fit_easy_job_folder_result(
    config: ParamFitEasyConfig,
    wp_fit_flyte_result: WhatprotFitV1FlyteResult,
    paramfit_flyte_result: FlyteParameterEstimationFitResult,
) -> None:
    if config.job:
        job_path = write_job_folder(
            job_folder=config.job, config_dict=config.to_dict(), clean=False
        )
        # Note: the results passed are just to indicate dependency for the DAG builder,
        # those results have already been written in this case by the sub-workflows that
        # we call, which is also why we do clean=False above.
        reports = [
            "param_fit_easy.ipynb",
        ]
        if config.sigproc_job:
            reports.append("paramfit_sigproc_qc.ipynb")

        write_reports(job_path=job_path, reports=reports)


@task(
    task_config=task_configs.generate_efs_task_config(),
    secret_requests=task_configs.generate_secret_requests(),
)
def extract_params(
    config: ParameterEstimationConfig,
) -> ParameterEstimationParams:
    rv = pfit_v1_params.import_plaster(
        pfit_v1_params.PlasterPaths(
            job=config.job,
            vfs_job=config.vfs_job,
            classify_job=config.classify_job,
            sim_job=config.sim_job,
            sigproc_job=config.sigproc_job,
            filter_params_yaml_path=config.filter_params_yaml_path,
        )
    )

    # Defaults for all parameters used for simulator.
    defaults: dict[str, ParamInfo] = {}
    if config.defaults:
        defaults = {x.name: x for x in config.defaults}
    # Ignore specifications for unknown parameters.
    defaults = {k: v for (k, v) in defaults.items() if k in rv.plaster_x0}
    # For all fitted parameters...
    for k, v in rv.plaster_x0.items():
        try:
            # Check specified parameter.
            v2 = defaults[k]
            assert (
                v2.initial_value is None
                or v2.is_fixed
                or v2.bound_lower <= v2.initial_value <= v2.bound_upper
            )
        except KeyError:
            # Add unspecified parameter.
            defaults[k] = ParamInfo(
                # Use default for unspecified parameters.
                name=k
                # initial_value: Optional[float] = None
                # is_fixed: bool = False
                # bound_lower: float = 0.0
                # bound_upper: float = 1.0
            )
    # At this point, defaults contains all and only plaster_x0 keys.

    rv.x0 = config.x0
    rv.defaults = defaults
    rv.n_reps = config.n_reps
    rv.default_n_samples = config.default_n_samples

    return rv


@task
def extract_rf_filter_params(
    config: ParameterEstimationConfig,
) -> RFV2Params:
    rf_params = RFV2Params(
        filter_params=config.filter_params,
        filter_reject_thresh_all_cycles=config.filter_reject_thresh_all_cycles,
    )
    return rf_params


def de_floatify(params: ParameterEstimationParams):
    params.n_ch = int(params.n_ch)
    params.n_pres = int(params.n_pres)
    params.n_edmans = int(params.n_edmans)
    params.n_mocks = int(params.n_mocks)
    params.n_span_labels_list = [int(x) for x in params.n_span_labels_list]
    params.true_dyetracks_count = [int(x) for x in params.true_dyetracks_count]
    params.n_reps = int(params.n_reps)
    params.default_n_samples = int(params.default_n_samples)


@dataclass
class CheckArgs(DataClassJsonMixin):
    params: ParameterEstimationParams
    param_ind: int
    rep: int


@task
def parameter_estimation_check_flyte_task(
    args: CheckArgs,
) -> ParameterEstimationCheckOneResult:
    params = args.params
    de_floatify(params)

    param_ind = int(args.param_ind)
    rep = int(args.rep)

    return pfit_v1_tasks.param_fit_v1_check(params, param_ind, rep)


@task
def parameter_estimation_check_coalesce_flyte_task(
    params: ParameterEstimationParams,
    res_list: list[ParameterEstimationCheckOneResult],
) -> FlyteParameterEstimationCheckResult:
    res = ParameterEstimationCheckResult(params=params, res=res_list)
    return FlyteParameterEstimationCheckResult.from_inst(res)


@task
def parameter_estimation_prep_check_flyte_task(
    params: ParameterEstimationParams,
) -> list[CheckArgs]:
    lst = []
    for param_ind in range(len(params.plaster_x0)):
        for rep in range(params.n_reps):
            lst += [CheckArgs(params, param_ind, rep)]

    return lst


@dataclass
class FitArgs0(DataClassJsonMixin):
    # FitArgs0 was needed to avoid the Flyte univariate list error message.
    method: str
    resample: bool  # should this fit be done on a resampling of the input data?
    x0: list[float]


@dataclass
class FitArgs(DataClassJsonMixin):
    params: ParameterEstimationParams
    fit_params: FitArgs0


@task
def parameter_estimation_prep_fit_flyte_task(
    params: ParameterEstimationParams,
    method: str,
    initial_fit: ParameterEstimationFitOneResult,
    resample: bool,
) -> list[FitArgs]:
    x0 = initial_fit.x
    return [
        FitArgs(params, FitArgs0(method, resample, x0)) for rep in range(params.n_reps)
    ]


@task
def parameter_estimation_fit_coalesce_flyte_task(
    params: ParameterEstimationParams,
    fit_list: list[ParameterEstimationFitOneResult],
    bootstrap_list: list[ParameterEstimationFitOneResult],
) -> FlyteParameterEstimationFitResult:
    res = ParameterEstimationFitResult(
        params=params, res=fit_list, bootstrap_fits=bootstrap_list
    )
    return FlyteParameterEstimationFitResult.from_inst(res)


@task
def parameter_estimation_fit_flyte_task(
    params: ParameterEstimationParams,
    method: str,
    x0: Optional[list[float]],
) -> ParameterEstimationFitOneResult:
    de_floatify(params)
    return pfit_v1_tasks.param_fit_v1_fit(params, method, x0, False)


@task
def parameter_estimation_refit_flyte_task(
    params: ParameterEstimationParams,
    method: str,
    fit_one_result: ParameterEstimationFitOneResult,
) -> ParameterEstimationFitOneResult:
    """
    Peforms a new fit starting at the x returned by a previous fit in fit_one_result
    """
    de_floatify(params)
    return pfit_v1_tasks.param_fit_v1_fit(params, method, fit_one_result.x, False)


# TODO: this is used for the bootstrapping - and a map task is designed to
# run lots of pods in parallel on a single node.  Does that mean this task
# needs to requeset the number of cpus to execute e.g. 100 in parallel,
# or is it correct to have the task just ask for 1 or 2 cpu?
# For some reason the CPU utilization is not showing up on DD for this task.
@task
def parameter_estimation_fit_flyte_mappable_task(
    args: FitArgs,
) -> ParameterEstimationFitOneResult:
    params = args.params
    de_floatify(params)

    method = args.fit_params.method
    resample = args.fit_params.resample
    x0 = args.fit_params.x0

    return pfit_v1_tasks.param_fit_v1_fit(params, method, x0, resample)


# -------------------- modules/parameter_estimation/v1/flyte_workflows.py --------------------


@workflow
def parameter_estimation_precheck(
    config: ParameterEstimationConfig,
) -> FlyteParameterEstimationCheckResult:
    params = extract_params(config=config)

    arg_list = parameter_estimation_prep_check_flyte_task(params=params)
    res_list = map_task(parameter_estimation_check_flyte_task)(args=arg_list)

    check_flyte_result = parameter_estimation_check_coalesce_flyte_task(
        params=params, res_list=res_list
    )

    write_check_job_folder_result(config=config, check_flyte_result=check_flyte_result)

    return check_flyte_result


@task
def paramfit_prep(
    config: ParameterEstimationConfig,
    prep_train_flyte_result: PrepV2FlyteResult,  # to get at the peptide sequence
    sim_train_flyte_result: SimV3FlyteResult,  # to get at the seqparams, markers, etc
    rf_flyte_result: RFV2FlyteResult,  # to get at the "true" dyeseqs & counts
) -> ParameterEstimationParams:
    # This replaces the extract_params and import_plaster() fn it called.

    # These are the prep and sim that were part of the VFS job that trained
    # the classifier.  So they were done with spanning_dyetrack option to generate
    # all possible dyetracks.
    prep_result = prep_train_flyte_result.load_result()
    sim_result = sim_train_flyte_result.load_result()

    # dyeseqs
    # TODO: why do we need the form "...K...K"?  Can we just use the sequence?  The flu?
    marker_labels = [x.aa for x in sim_result.params.markers]
    peptides = prep_result.peps__pepstrs()["seqstr"][
        1:
    ].values  # skip null peptide at 0
    dyeseqs = [
        "".join([x if x in marker_labels else "." for x in seq]) for seq in peptides
    ]

    # Other stuff from sim
    n_channels = sim_result.n_channels
    n_pres = sim_result.params.seq_params.n_pres
    n_edmans = sim_result.params.seq_params.n_edmans
    n_mocks = sim_result.params.seq_params.n_mocks
    n_span_labels_list = sim_result.params.n_span_labels_list

    # This doesn't appear to be used downstream in the default fit workflow
    from plaster.run.sim_v3.sim_v3_params import Marker, SeqParams

    plaster_x0 = {
        "p_initial_block": SeqParams.p_initial_block,
        "p_cyclic_block": SeqParams.p_cyclic_block,
        "p_detach": SeqParams.p_detach,
        "p_edman_failure": SeqParams.p_edman_failure,
        **{
            k: v
            for i, x in enumerate(marker_labels)
            for (k, v) in zip(
                [f"p_bleach_ch{i}", f"p_dud_ch{i}"], [Marker.p_bleach, Marker.p_dud]
            )
        },
    }

    # Defaults for all parameters used for simulator.
    defaults: dict[str, ParamInfo] = {}
    if config.defaults:
        defaults = {x.name: x for x in config.defaults}
    # Ignore specifications for unknown parameters.
    defaults = {k: v for (k, v) in defaults.items() if k in plaster_x0}
    # For all fitted parameters...
    for k, v in plaster_x0.items():
        try:
            # Check specified parameter.
            v2 = defaults[k]
            assert (
                v2.initial_value is None
                or v2.is_fixed
                or v2.bound_lower <= v2.initial_value <= v2.bound_upper
            )
        except KeyError:
            # Add unspecified parameter.
            defaults[k] = ParamInfo(
                # Use default for unspecified parameters.
                name=k
                # initial_value: Optional[float] = None
                # is_fixed: bool = False
                # bound_lower: float = 0.0
                # bound_upper: float = 1.0
            )
    # At this point, defaults contains all and only plaster_x0 keys.

    # Get the the dyetracks and counts from the sim and classifier.
    # I am very tired; there is a more efficient way to do this!
    rf_result = rf_flyte_result.load_result()

    dyetracks_str = ["".join(map(str, r)) for r in sim_result.train_dytmat]
    pred_pep_iz = rf_result.pred_pep_iz.get()
    dyetracks_all = [dyetracks_str[i] for i in pred_pep_iz]
    rf_df = pd.DataFrame(
        zip(pred_pep_iz, dyetracks_all, rf_result.scores.get()),
        columns=["dyetrack_i", "dyetrack", "score"],
    )
    df_counts = rf_df.groupby("dyetrack").size().reset_index(name="dyetrk_count")
    true_dyetracks = list(df_counts.dyetrack.values)
    true_dyetracks_count = list(map(int, df_counts.dyetrk_count.values))

    return ParameterEstimationParams(
        exp_name="unused",  # Currently unused
        dyeseqs=dyeseqs,  # dyeseqs of the form "...K..K..Y.." (only labeled AAs are non-dot)
        n_ch=n_channels,  # seqparams
        n_pres=n_pres,
        n_edmans=n_edmans,
        n_mocks=n_mocks,
        n_span_labels_list=n_span_labels_list,  # max labels, dye-span sim
        marker_labels=marker_labels,  # e.g. [K,Y] (list of AAs labeled) how is this used downstream?
        true_dyetracks=true_dyetracks,  # list of dyetrack strings the classifier made calls to
        true_dyetracks_count=true_dyetracks_count,  # counts of calls to dyetrack strings
        plaster_x0=plaster_x0,  # plaster defaults that could be used to initialize fitter
        x0=config.x0,
        defaults=defaults,
        n_reps=config.n_reps,
        default_n_samples=config.default_n_samples,
    )


# Alternate workflow by tfb.  Direct followed by Powell refinement, and optional
# separate bootstrapping fits.
@workflow
def param_fit(
    config: ParameterEstimationConfig,
) -> FlyteParameterEstimationFitResult:
    # Load trained classifier and prep/sims going into it
    (
        prep_train_flyte_result,
        sim_train_flyte_result,
        rf_train_flyte_result,
    ) = fetch_vfs_train_flyte_results(config=config)

    # Load simulation data to fit to, if available
    sim_test_flyte_result = fetch_sim_test_flyte_result(config=config)

    # Load sigproc data to fit to, if available
    (
        ims_import_flyte_result,
        sigproc_flyte_result,
    ) = fetch_sigproc_flyte_results(config=config)

    # Get filtering params to be applied ahead of classification, if any
    rf_params = extract_rf_filter_params(config=config)

    # Classify either the sim or the sigproc data with trained classifier
    rf_classify_flyte_result = rf_v2_flyte_task(
        rf_params=rf_params,
        rf_train_flyte_result=rf_train_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
        sim_flyte_result=sim_test_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
    )

    # Extract information from various results to build the params for the fitter
    params = paramfit_prep(
        config=config,
        prep_train_flyte_result=prep_train_flyte_result,  # to get at the peptide sequence
        sim_train_flyte_result=sim_train_flyte_result,  # to get at the seqparams, markers, etc
        rf_flyte_result=rf_classify_flyte_result,  # to get at the "true" dyeseqs & counts
    )

    # Fit
    fit_direct_result = parameter_estimation_fit_flyte_task(
        params=params,
        method="Direct",
        x0=None,
    )
    fit_powell_result = parameter_estimation_refit_flyte_task(
        params=params, method="SciPyMinimizePowell", fit_one_result=fit_direct_result
    )

    # Bootstrap /resampled fits
    arg_list = parameter_estimation_prep_fit_flyte_task(
        params=params,
        method="SciPyMinimizePowell",
        initial_fit=fit_direct_result,
        resample=True,
    )
    bootstrap_res_list = map_task(parameter_estimation_fit_flyte_mappable_task)(
        args=arg_list
    )

    # Collect results
    fit_results = parameter_estimation_fit_coalesce_flyte_task(
        params=params,
        fit_list=[fit_direct_result, fit_powell_result],
        bootstrap_list=bootstrap_res_list,
    )

    # Write results
    write_fit_job_folder_result(config=config, fit_flyte_result=fit_results, clean=True)

    return fit_results


# Experimental workflow to use a provided vfs_config and call the vfs_workflow
# from without our workflow to do train the classifier needed during param_fit.
# I think the most straightforward way to accomplish the logic required is to
# make this a dynamic workflow, so that it can either load existing results
# or run a subworkflow.
@dynamic
def get_vfs_results(
    config: ParameterEstimationConfig,
) -> tuple[PrepV2FlyteResult, SimV3FlyteResult, RFTrainV2FlyteResult]:
    # If an already-trained classifier was specified, load the required results.
    if config.vfs_job:
        return fetch_vfs_train_flyte_results(config=config)

    # Otherwise run a vfs_workflow on the provided vfs_config.  It may be
    # we need to run the vfs_big_workflow depending on memory requirements.
    # Let's assert that the vfs job_folder is not the same as the top-level
    # job_folder, so that the vfs only cleans and writes to it's own folder.
    assert config.job != config.vfs_config.job

    # multi-channel requires considerably more memory for the RF train.
    # vfs_big_workflow, which asks for 1T of ram, is probably overkill,
    # but there is currently not a middle ground.  The [2,1] span-dyetrack
    # classify took ~120G memory.
    n_channels = len(config.vfs_config.n_span_labels_list)
    vfs_fn = vfs_workflow if n_channels == 1 else vfs_big_workflow
    (
        prep_flyte_result,
        sim_flyte_result,
        rf_train_flyte_result,
        rf_test_flyte_result,
    ) = vfs_fn(config=config.vfs_config)
    return prep_flyte_result, sim_flyte_result, rf_train_flyte_result


@workflow
def param_fit_vfs(
    config: ParameterEstimationConfig,
) -> FlyteParameterEstimationFitResult:
    # I want to rethink the whole "cleaning the job_folder" as part of writing
    # the job_folder.  This gets complicated when launching subworkflows, because
    # you most often don't then want the final task in that subworkflow to clean
    # the job folder.  Actually, maybe it is ok, IF the subworkflow is
    # working in a subfolder of the top-level job_folder.  But it still means that
    # as the final task in the top-level workflow, you don't want to clean the
    # top-level job folder, otherwise you'll lose the results that were written
    # as part of subworkflows (which for organizational purposes I set those
    # job_folders to be inside the top-level job_folder).  So I think we'll
    # often want to clean the job folder before the work begins - and maybe this
    # should get done before launch, even as part of the initial config setup,
    # or it could be a member of BaseGenonfigDC - clean_job_folder() and this
    # can get called either via your notebook, or by the webapp that launches
    # the job.

    clean_job_folder(config=config)

    # Either get existing VFS results, or potentially run a full VFS workflow.
    # Note if a vfs_workflow is run, we expect that the caller has taken care
    # to set the vfs_config.job_folder to be under the top-level job_folder,
    # so it will only clean it's own folder before writing results.
    (
        prep_train_flyte_result,
        sim_train_flyte_result,
        rf_train_flyte_result,
    ) = get_vfs_results(config=config)

    # Load simulation data to fit to, if available
    sim_test_flyte_result = fetch_sim_test_flyte_result(config=config)

    # Load sigproc data to fit to, if available
    (
        ims_import_flyte_result,
        sigproc_flyte_result,
    ) = fetch_sigproc_flyte_results(config=config)

    # Get filtering params to be applied ahead of classification, if any
    rf_params = extract_rf_filter_params(config=config)

    # Classify either the sim or the sigproc data with trained classifier
    rf_classify_flyte_result = rf_v2_flyte_task(
        rf_params=rf_params,
        rf_train_flyte_result=rf_train_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
        sim_flyte_result=sim_test_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
    )

    # Extract information from various results to build the params for the fitter
    params = paramfit_prep(
        config=config,
        prep_train_flyte_result=prep_train_flyte_result,  # to get at the peptide sequence
        sim_train_flyte_result=sim_train_flyte_result,  # to get at the seqparams, markers, etc
        rf_flyte_result=rf_classify_flyte_result,  # to get at the "true" dyeseqs & counts
    )

    # Fit
    fit_direct_result = parameter_estimation_fit_flyte_task(
        params=params,
        method="Direct",
        x0=None,
    )
    fit_powell_result = parameter_estimation_refit_flyte_task(
        params=params, method="SciPyMinimizePowell", fit_one_result=fit_direct_result
    )

    # Bootstrap /resampled fits
    arg_list = parameter_estimation_prep_fit_flyte_task(
        params=params,
        method="SciPyMinimizePowell",
        initial_fit=fit_direct_result,
        resample=True,
    )
    bootstrap_res_list = map_task(parameter_estimation_fit_flyte_mappable_task)(
        args=arg_list
    )

    # Collect results
    fit_results = parameter_estimation_fit_coalesce_flyte_task(
        params=params,
        fit_list=[fit_direct_result, fit_powell_result],
        bootstrap_list=bootstrap_res_list,
    )

    # Write results
    # We need to ensure to NOT clean the job folder if we want to see the results of
    # the VFS job that ran as the first subworkflow above!
    write_fit_job_folder_result(
        config=config, fit_flyte_result=fit_results, clean=False
    )

    return fit_results


# Identical to the above workflow but calls the rf_v2_big... workflow
@workflow
def param_fit_big_vfs(
    config: ParameterEstimationConfig,
) -> FlyteParameterEstimationFitResult:
    clean_job_folder(config=config)
    (
        prep_train_flyte_result,
        sim_train_flyte_result,
        rf_train_flyte_result,
    ) = get_vfs_results(config=config)
    sim_test_flyte_result = fetch_sim_test_flyte_result(config=config)
    (
        ims_import_flyte_result,
        sigproc_flyte_result,
    ) = fetch_sigproc_flyte_results(config=config)
    rf_params = extract_rf_filter_params(config=config)
    rf_classify_flyte_result = rf_v2_big_flyte_task(
        rf_params=rf_params,
        rf_train_flyte_result=rf_train_flyte_result,
        sigproc_flyte_result=sigproc_flyte_result,
        sim_flyte_result=sim_test_flyte_result,
        ims_import_flyte_result=ims_import_flyte_result,
    )
    params = paramfit_prep(
        config=config,
        prep_train_flyte_result=prep_train_flyte_result,  # to get at the peptide sequence
        sim_train_flyte_result=sim_train_flyte_result,  # to get at the seqparams, markers, etc
        rf_flyte_result=rf_classify_flyte_result,  # to get at the "true" dyeseqs & counts
    )
    fit_direct_result = parameter_estimation_fit_flyte_task(
        params=params,
        method="Direct",
        x0=None,
    )
    fit_powell_result = parameter_estimation_refit_flyte_task(
        params=params, method="SciPyMinimizePowell", fit_one_result=fit_direct_result
    )
    arg_list = parameter_estimation_prep_fit_flyte_task(
        params=params,
        method="SciPyMinimizePowell",
        initial_fit=fit_direct_result,
        resample=True,
    )
    bootstrap_res_list = map_task(parameter_estimation_fit_flyte_mappable_task)(
        args=arg_list
    )
    fit_results = parameter_estimation_fit_coalesce_flyte_task(
        params=params,
        fit_list=[fit_direct_result, fit_powell_result],
        bootstrap_list=bootstrap_res_list,
    )
    write_fit_job_folder_result(
        config=config, fit_flyte_result=fit_results, clean=False
    )
    return fit_results


# Can this just be a regular task, or does it need to be a "dynamic" workflow
# since it is going to call another sub-workflow?  dynamic workflows seem just
# like tasks to me...
from plaster.genv2.generators.sim import SimConfig, sim_only_workflow
from plaster.genv2.generators.whatprot import WhatprotConfig
from plaster.run.rf_train_v2.rf_train_v2_params import RFTrainV2Params


@dynamic
def run_whatprot_fit(
    config: ParamFitEasyConfig,
) -> tuple[
    WhatprotPreV1FlyteResult,
    WhatprotFitV1FlyteResult,
    WhatprotFitBootstrapV1FlyteResult,
]:
    job_folder = resolve_job_folder(config.job) / "whatprot"

    # First create a super-simple simulation because this is the easiest
    # way to parameterize whatprot.
    sim_config = SimConfig(
        type="sim_only_workflow",
        job=str(job_folder / "sim_train_whatprot"),
        markers=config.markers,
        proteins=[config.protein],
        seq_params=config.seq_params,
        seed=0,
        n_samples_train=1,
        n_samples_test=1,
    )

    prep_flyte_result, sim_flyte_result = sim_only_workflow(config=sim_config)

    wp_config = WhatprotConfig(
        type="whatprot_fit",
        job=str(job_folder),
        sim_train_job=sim_config.job,
        sim_test_job=config.sim_job,
        sigproc_job=config.sigproc_job,
        # Note that in the __post_init__ of the EasyConfig, the filter_params
        # have already been loaded/validated.  We set those below, but we also
        # set the yaml_path so that the WhatprotConfig won't complain it's missing,
        # and so that the WhatprotConfig as written in the results will reference
        # the yaml file the params were loaded from.
        filter_params_yaml_path=config.filter_params_yaml_path,
        filter_reject_thresh_all_cycles=True,
        filter_params=config.filter_params,  # note these params have already been loaded
        # Some special whatprot params, there are others that could be used as well.
        wp_maxruntime_minutes=config.wp_maxruntime_minutes,  # per fit
        wp_numbootstrap=config.n_bootstrap,
    )

    (
        wp_pre_flyte_result,
        wp_fit_flyte_result,
        wp_bootstrap_flyte_result,
    ) = whatprot_fit_workflow(config=wp_config)

    # use some special Flyte syntax to indicate the sim needs to complete before
    # the whatprot_workflow is run even though we don't directly pass outputs.
    prep_flyte_result >> wp_pre_flyte_result

    return wp_pre_flyte_result, wp_fit_flyte_result, wp_bootstrap_flyte_result


@dynamic
def run_param_fit(config: ParamFitEasyConfig) -> FlyteParameterEstimationFitResult:
    job_folder = resolve_job_folder(config.job) / "param_fit"

    # We need to calculate how many labels are on the peptide in question for
    # input to the spanning-dyetrack sim/vfs.
    n_span_labels_list = []
    for m in config.markers:
        n = 0
        for aa in m.aa:  # because m.aa can really be multiple, e.g. "DE"
            n += config.protein.sequence.count(aa)
        n_span_labels_list.append(n)
    logger.info(
        "compute n_span_labels_list",
        n_span_labels_list=n_span_labels_list,
        protein=config.protein.sequence,
    )

    pfit_config = ParameterEstimationConfig(
        type="param_fit_vfs",
        job=str(job_folder),
        sim_job=config.sim_job,
        sigproc_job=config.sigproc_job,
        # Note that in the __post_init__ of the EasyConfig, the filter_params
        # have already been loaded/validated.  We set those below, but we also
        # set the yaml_path so that the ParameterEstimationConfig won't complain it's missing,
        # and so that this config as written in the results will reference
        # the yaml file the params were loaded from.
        filter_params_yaml_path=config.filter_params_yaml_path,
        filter_reject_thresh_all_cycles=True,
        filter_params=config.filter_params,  # note these params have already been loaded
        defaults=config.param_constraints,
        # Spec the classifier
        vfs_config=VFSConfig(
            type="vfs",
            job=str(job_folder / "vfs"),
            span_dyetracks=True,
            n_span_labels_list=n_span_labels_list,
            markers=config.markers,
            seq_params=config.seq_params,  # note all params at default values
            proteins=[config.protein],  # explicitly list our one peptide
            seed=0,
            n_samples_train=config.n_samples_train,  # this is per spanning-dyetrack
            rf_train_params=RFTrainV2Params(
                classify_dyetracks=True,  # we will classify to dyetracks, not peptides.
            ),
            rf_test_params=RFV2Params(),
        ),
        # and whatever params to the fitter:
        default_n_samples=config.n_samples_default,  # This is how many dyetracks to gen via dyetrack-sim at the param location.
        # You'd like this to be large enough to really sample the distribution.  It will
        # be compared to the distribution of the classification to dyetracks of the input
        # radiometry.  So for simulated input data that you are testing with, create sims
        # with largeish numbers of samples for best results (that also reasonably capture
        # the dyetrack distribution for the single peptide)
        n_reps=config.n_bootstrap,
    )

    # Now run the workflow
    if len(n_span_labels_list) == 1:
        return param_fit_vfs(config=pfit_config)
    else:
        return param_fit_big_vfs(config=pfit_config)


@workflow
def param_fit_easy(
    config: ParamFitEasyConfig,
) -> tuple[
    FlyteParameterEstimationFitResult,
    WhatprotPreV1FlyteResult,
    WhatprotFitV1FlyteResult,
    WhatprotFitBootstrapV1FlyteResult,
]:
    # Clean the job folder, since we'll be telling write_job_folder to NOT do this
    # since we have multiple workflows writing their results during this process.
    promise_0 = clean_job_folder(config=config)

    paramfit_flyte_result = run_param_fit(config=config)

    promise_0 >> paramfit_flyte_result  # Tell Flyte that clean needs to finish first

    (
        wp_pre_flyte_result,
        wp_fit_flyte_result,
        wp_bootstrap_flyte_result,
    ) = run_whatprot_fit(config=config)

    promise_0 >> wp_pre_flyte_result  # Tell Flyte that clean needs to finish first

    # Those separate workflows have already written their respective results
    # into the job folder, but we want to write a comprehensive report as well.

    write_param_fit_easy_job_folder_result(
        config=config,
        wp_fit_flyte_result=wp_fit_flyte_result,
        paramfit_flyte_result=paramfit_flyte_result,
    )

    return (
        paramfit_flyte_result,
        wp_pre_flyte_result,
        wp_fit_flyte_result,
        wp_bootstrap_flyte_result,
    )


# The original parameter_estimation_fit workflow by KV (mods by tfb)
@workflow
def parameter_estimation_fit(
    config: ParameterEstimationConfig,
) -> FlyteParameterEstimationFitResult:
    params = extract_params(config=config)

    # Initial Fit(s)
    fit_direct_res = parameter_estimation_fit_flyte_task(
        params=params, method="Direct", x0=None
    )

    fit_powell_res = parameter_estimation_refit_flyte_task(
        params=params, method="SciPyMinimizePowell", fit_one_result=fit_direct_res
    )

    fit_list = [fit_direct_res, fit_powell_res]

    # Bootstrap (resampled) fits
    arg_list = parameter_estimation_prep_fit_flyte_task(
        params=params,
        method="SciPyMinimizePowell",
        initial_fit=fit_direct_res,
        resample=True,
    )
    bootstrap_list = map_task(parameter_estimation_fit_flyte_mappable_task)(
        args=arg_list
    )

    fit_flyte_result = parameter_estimation_fit_coalesce_flyte_task(
        params=params, fit_list=fit_list, bootstrap_list=bootstrap_list
    )

    write_fit_job_folder_result(
        config=config, fit_flyte_result=fit_flyte_result, clean=True
    )

    return fit_flyte_result


# -------------------- modules/parameter_estimation/v1/gen.py --------------------


def generate_precheck(config: ParameterEstimationConfig) -> GenerateResult:
    """Generate a job from a ParameterEstimationConfig dataclass.

    Args
    ----
    config : ParameterEstimationConfig
        A ParameterEstimationConfig specifying information for one or more runs.

    Returns
    -------
    A gen_config.GenerateResult object describing the runs and static_reports to be executed
    for this job.
    """

    # This doesn't actually even get called in the ControlPanel/Flyte codepath!

    static_reports = ["parameter_estimation_precheck"]

    # Reads filter_params from file on EFS ahead of flyte launch.
    config.load_filtering_params()

    return GenerateResult(runs=[], static_reports=static_reports)


generator_precheck = Generator(
    ParameterEstimationConfig, generate_precheck, workflow=parameter_estimation_precheck
)


def generate_fit(config: ParameterEstimationConfig) -> GenerateResult:
    """Generate a job from a ParameterEstimationConfig dataclass.

    Args
    ----
    config : ParameterEstimationConfig
        A ParameterEstimationConfig specifying information for one or more runs.

    Returns
    -------
    A gen_config.GenerateResult object describing the runs and static_reports to be executed
    for this job.
    """

    # This doesn't actually even get called in the ControlPanel/Flyte codepath!

    static_reports = ["parameter_estimation_fit"]

    # Reads filter_params from file on EFS ahead of flyte launch.
    config.load_filtering_params()

    return GenerateResult(runs=[], static_reports=static_reports)


generator_fit = Generator(
    ParameterEstimationConfig, generate_fit, workflow=parameter_estimation_fit
)
generator_param_fit = Generator(
    ParameterEstimationConfig, generate_fit, workflow=param_fit
)
generator_param_fit_vfs = Generator(
    ParameterEstimationConfig, generate_fit, workflow=param_fit_vfs
)
generator_param_fit_easy = Generator(
    ParamFitEasyConfig, generate_fit, workflow=param_fit_easy
)
