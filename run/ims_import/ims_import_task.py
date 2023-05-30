import pathlib

import flytekit
from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from plumbum import local

from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import (
    ImsImportFlyteResult,
    ImsImportResult,
)
from plaster.run.ims_import.ims_import_worker import ims_import


@task(
    requests=Resources(cpu="64", mem="128Gi"), limits=Resources(cpu="128", mem="192G")
)
def ims_import_flyte_task(
    src_dir: FlyteDirectory, ims_import_params: ImsImportParams
) -> ImsImportFlyteResult:
    folder = pathlib.Path(flytekit.current_context().working_directory) / "ims_import"

    folder.mkdir(parents=True, exist_ok=True)

    result = ims_import(
        local.path(src_dir.download()),
        ims_import_params,
        result_class=ImsImportResult,
        folder=folder,
    )

    result.save()

    return ImsImportFlyteResult.from_inst(result)
