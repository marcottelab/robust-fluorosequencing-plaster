import pathlib

import flytekit
from flytekit import Resources, task

from plaster.run.ims_import.ims_import_result import ImsImportFlyteResult
from plaster.run.priors import Priors
from plaster.run.sigproc_v2 import sigproc_v2_common as common
from plaster.run.sigproc_v2 import sigproc_v2_worker as worker
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.run.sigproc_v2.sigproc_v2_result import (
    SigprocV2FlyteResult,
    SigprocV2ResultDC,
)


@task(
    requests=Resources(cpu="64", mem="128Gi"), limits=Resources(cpu="128", mem="192G")
)
def sigproc_v2_analyze_flyte_task(
    sigproc_v2_params: SigprocV2Params, ims_import_flyte_result: ImsImportFlyteResult
) -> SigprocV2FlyteResult:
    folder = pathlib.Path(flytekit.current_context().working_directory) / "sigproc_v2"

    folder.mkdir(parents=True, exist_ok=True)

    ims_import_result = ims_import_flyte_result.load_result()

    result = worker.analyze(
        sigproc_v2_params=sigproc_v2_params,
        ims_import_result=ims_import_result,
        result_class=SigprocV2ResultDC,
        folder=folder,
    )

    return SigprocV2FlyteResult.from_inst(result)


@task(
    requests=Resources(cpu="64", mem="128Gi"), limits=Resources(cpu="128", mem="192G")
)
def sigproc_v2_calibrate_flyte_task(
    sigproc_v2_params: SigprocV2Params, ims_import_flyte_result: ImsImportFlyteResult
) -> SigprocV2FlyteResult:
    folder = pathlib.Path(flytekit.current_context().working_directory) / "sigproc_v2"

    folder.mkdir(parents=True, exist_ok=True)

    result = worker.calibrate(
        sigproc_v2_params,
        ims_import_flyte_result.load_result(),
        result_class=SigprocV2ResultDC,
        folder=folder,
    )

    return SigprocV2FlyteResult.from_inst(result)
