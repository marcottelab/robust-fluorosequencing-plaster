from pathlib import Path
from typing import Optional

import flytekit
import structlog
from flytekit import Resources, task
from pandas import DataFrame

logger = structlog.get_logger()

from plaster.reports.helpers.report_params import FilterParams
from plaster.reports.helpers.report_params_filtering import (
    get_classify_default_filtering_df,
)
from plaster.run.ims_import.ims_import_result import (
    ImsImportFlyteResult,
    ImsImportResult,
)
from plaster.run.rf_train_v2.rf_train_v2_result import RFTrainV2FlyteResult
from plaster.run.rf_v2.rf_v2_params import RFV2Params
from plaster.run.rf_v2.rf_v2_result import RFV2FlyteResult, RFV2ResultDC
from plaster.run.rf_v2.rf_v2_worker import rf_classify
from plaster.run.sigproc_v2.sigproc_v2_result import (
    SigprocV2FlyteResult,
    SigprocV2ResultDC,
)
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult


def get_sigproc_filter_df(
    filter_params: FilterParams,
    filter_all_cycles: bool,
    sigproc_result: SigprocV2ResultDC,
    ims_import_result: ImsImportResult,
) -> DataFrame:
    assert filter_params.check_all_present()
    filter_df = get_classify_default_filtering_df(
        sigproc=sigproc_result,
        ims_import=ims_import_result,
        params=filter_params,
        all_cycles=filter_all_cycles,
    )
    n_pass = filter_df.pass_all.sum()
    logger.info(
        "Filtering sigproc radmat ahead of classification",
        filter_shape=filter_df.shape,
        n_pass=n_pass,
    )
    return filter_df


@task(requests=Resources(cpu="16", mem="16Gi"), limits=Resources(cpu="48", mem="128Gi"))
def rf_v2_flyte_task(
    rf_params: RFV2Params,
    rf_train_flyte_result: RFTrainV2FlyteResult,
    *,
    sigproc_flyte_result: Optional[SigprocV2FlyteResult] = None,
    sim_flyte_result: Optional[SimV3FlyteResult] = None,
    ims_import_flyte_result: Optional[ImsImportFlyteResult] = None,
) -> RFV2FlyteResult:

    radmat = None
    folder = None
    assert not (sigproc_flyte_result is None and sim_flyte_result is None)

    # sigproc radmat takes precedence over sim radmat
    if sigproc_flyte_result is not None:
        sigproc_result = sigproc_flyte_result.load_result()
        radmat = sigproc_result.sig(flat_chcy=True)
        folder = Path(flytekit.current_context().working_directory) / "rf_classify"

        # If we've been passed filtering params AND an ims_import result, we'll construct
        # a filtering DF and use that to filter the radmat ahead of classification, which
        # can save a lot of memory and cpu time.
        if rf_params.filter_params and ims_import_flyte_result:
            ims_import_result = ims_import_flyte_result.load_result()
            filter_df = get_sigproc_filter_df(
                filter_params=rf_params.filter_params,
                filter_all_cycles=rf_params.filter_reject_thresh_all_cycles,
                sigproc_result=sigproc_result,
                ims_import_result=ims_import_result,
            )
            assert len(filter_df) == radmat.shape[0]
            radmat = radmat[filter_df.pass_all]
            logger.info("New radmat shape", shape=radmat.shape)
        else:
            logger.info("Classify sigproc radmat with no pre-filtering.")

    elif sim_flyte_result is not None:
        radmat = sim_flyte_result.load_result().flat_test_radmat()
        folder = Path(flytekit.current_context().working_directory) / "rf_test"

    else:
        raise ValueError(
            "Both sim_result and sigproc_result are None. Nothing to classify!"
        )

    folder.mkdir(parents=True, exist_ok=True)

    assert len(radmat.shape) == 2

    rf_result = rf_classify(
        rf_params,
        rf_train_flyte_result.load_result(),
        radmat,
        result_class=RFV2ResultDC,
        folder=folder,
    )

    if sim_flyte_result is not None:
        # Stuff the true value into the results to simplify downstream processing
        rf_result.true_pep_iz.set(sim_flyte_result.load_result().test_true_pep_iz.arr())
        rf_result.train_pep_recalls.set(
            sim_flyte_result.load_result().train_pep_recalls
        )

    return RFV2FlyteResult.from_inst(rf_result)


@task(requests=Resources(cpu="64", mem="950Gi"), limits=Resources(cpu="128", mem="1Ti"))
def rf_v2_big_flyte_task(
    rf_params: RFV2Params,
    rf_train_flyte_result: RFTrainV2FlyteResult,
    *,
    sigproc_flyte_result: Optional[SigprocV2FlyteResult] = None,
    sim_flyte_result: Optional[SimV3FlyteResult] = None,
    ims_import_flyte_result: Optional[ImsImportFlyteResult] = None,
) -> RFV2FlyteResult:

    radmat = None
    folder = None
    assert not (sigproc_flyte_result is None and sim_flyte_result is None)

    # sigproc radmat takes precedence over sim radmat
    if sigproc_flyte_result is not None:
        sigproc_result = sigproc_flyte_result.load_result()
        radmat = sigproc_result.sig(flat_chcy=True)
        folder = Path(flytekit.current_context().working_directory) / "rf_classify"

        # If we've been passed filtering params AND an ims_import result, we'll construct
        # a filtering DF and use that to filter the radmat ahead of classification, which
        # can save a lot of memory and cpu time.
        if rf_params.filter_params and ims_import_flyte_result:
            ims_import_result = ims_import_flyte_result.load_result()
            filter_df = get_sigproc_filter_df(
                filter_params=rf_params.filter_params,
                filter_all_cycles=rf_params.filter_reject_thresh_all_cycles,
                sigproc_result=sigproc_result,
                ims_import_result=ims_import_result,
            )
            assert len(filter_df) == radmat.shape[0]
            radmat = radmat[filter_df.pass_all]
            logger.info("New radmat shape", shape=radmat.shape)
        else:
            logger.info("Classify sigproc radmat with no pre-filtering.")

    elif sim_flyte_result is not None:
        radmat = sim_flyte_result.load_result().flat_test_radmat()
        folder = Path(flytekit.current_context().working_directory) / "rf_test"

    else:
        raise ValueError(
            "Both sim_result and sigproc_result are None. Nothing to classify!"
        )

    folder.mkdir(parents=True, exist_ok=True)

    assert len(radmat.shape) == 2

    rf_result = rf_classify(
        rf_params,
        rf_train_flyte_result.load_result(),
        radmat,
        result_class=RFV2ResultDC,
        folder=folder,
    )

    if sim_flyte_result is not None:
        # Stuff the true value into the results to simplify downstream processing
        rf_result.true_pep_iz.set(sim_flyte_result.load_result().test_true_pep_iz.arr())
        rf_result.train_pep_recalls.set(
            sim_flyte_result.load_result().train_pep_recalls
        )

    return RFV2FlyteResult.from_inst(rf_result)
