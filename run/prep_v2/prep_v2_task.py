from pathlib import Path

import flytekit
from flytekit import Resources, task

from plaster.run.prep_v2.prep_v2_params import PrepV2Params
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.prep_v2.prep_v2_worker import prep_v2


@task(requests=Resources(cpu="2", mem="8Gi"), limits=Resources(cpu="4", mem="16Gi"))
def prep_flyte_task(prep_params: PrepV2Params) -> PrepV2FlyteResult:

    folder = Path(flytekit.current_context().working_directory) / "prep"
    folder.mkdir(parents=True, exist_ok=True)

    return PrepV2FlyteResult.from_inst(prep_v2(prep_params=prep_params, folder=folder))
