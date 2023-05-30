from pathlib import Path

import flytekit
from flytekit import Resources, task

from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.sim_v3 import sim_v3_worker
from plaster.run.sim_v3.sim_v3_params import SimV3Params
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult


@task(requests=Resources(cpu="16", mem="32Gi"), limits=Resources(cpu="16", mem="32Gi"))
def sim_v3_flyte_task(
    sim_params: SimV3Params, prep_flyte_result: PrepV2FlyteResult
) -> SimV3FlyteResult:

    folder = Path(flytekit.current_context().working_directory) / "sim"

    folder.mkdir(parents=True, exist_ok=True)

    return SimV3FlyteResult.from_inst(
        sim_v3_worker.sim_v3(
            sim_params=sim_params,
            prep_result=prep_flyte_result.load_result(),
            folder=folder,
        )
    )


@task(
    requests=Resources(cpu="64", mem="128Gi"), limits=Resources(cpu="64", mem="128Gi")
)
def sim_v3_big_flyte_task(
    sim_params: SimV3Params, prep_flyte_result: PrepV2FlyteResult
) -> SimV3FlyteResult:

    folder = Path(flytekit.current_context().working_directory) / "sim"

    folder.mkdir(parents=True, exist_ok=True)

    return SimV3FlyteResult.from_inst(
        sim_v3_worker.sim_v3(
            sim_params=sim_params,
            prep_result=prep_flyte_result.load_result(),
            folder=folder,
        )
    )


@task(
    requests=Resources(cpu="128", mem="950Gi"), limits=Resources(cpu="128", mem="950Gi")
)
def sim_v3_xlarge_flyte_task(
    sim_params: SimV3Params, prep_flyte_result: PrepV2FlyteResult
) -> SimV3FlyteResult:

    folder = Path(flytekit.current_context().working_directory) / "sim"

    folder.mkdir(parents=True, exist_ok=True)

    return SimV3FlyteResult.from_inst(
        sim_v3_worker.sim_v3(
            sim_params=sim_params,
            prep_result=prep_flyte_result.load_result(),
            folder=folder,
        )
    )
