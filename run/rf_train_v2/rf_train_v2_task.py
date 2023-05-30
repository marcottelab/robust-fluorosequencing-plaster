from pathlib import Path

import flytekit
from flytekit import Resources, task

from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.rf_train_v2.rf_train_v2_params import RFTrainV2Params
from plaster.run.rf_train_v2.rf_train_v2_result import (
    RFTrainV2FlyteResult,
    RFTrainV2ResultDC,
)
from plaster.run.rf_train_v2.rf_train_v2_worker import rf_train
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult


@task(requests=Resources(cpu="16", mem="16Gi"), limits=Resources(cpu="48", mem="128Gi"))
def rf_train_v2_flyte_task(
    rf_train_params: RFTrainV2Params,
    prep_flyte_result: PrepV2FlyteResult,
    sim_flyte_result: SimV3FlyteResult,
) -> RFTrainV2FlyteResult:

    folder = Path(flytekit.current_context().working_directory) / "rf_train"

    folder.mkdir(parents=True, exist_ok=True)

    result = rf_train(
        rf_train_params,
        prep_flyte_result.load_result(),
        sim_flyte_result.load_result(),
        folder,
        result_class=RFTrainV2ResultDC,
    )
    return RFTrainV2FlyteResult.from_inst(result)


@task(requests=Resources(cpu="64", mem="950Gi"), limits=Resources(cpu="128", mem="1Ti"))
def rf_train_v2_big_flyte_task(
    rf_train_params: RFTrainV2Params,
    prep_flyte_result: PrepV2FlyteResult,
    sim_flyte_result: SimV3FlyteResult,
) -> RFTrainV2FlyteResult:

    folder = Path(flytekit.current_context().working_directory) / "rf_train"

    folder.mkdir(parents=True, exist_ok=True)

    result = rf_train(
        rf_train_params,
        prep_flyte_result.load_result(),
        sim_flyte_result.load_result(),
        folder,
        result_class=RFTrainV2ResultDC,
    )
    return RFTrainV2FlyteResult.from_inst(result)
