from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.rad_filter.rad_filter_params import RadFilterParams
from plaster.run.rad_filter.rad_filter_worker import rad_filter
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.tools.pipeline.pipeline import PipelineTask


class RadFilterTask(PipelineTask):
    def start(self):
        rad_filter_params = RadFilterParams(**self.config.parameters)

        ims_import_result = ImsImportResult.load_from_folder(self.inputs.ims_import)
        sigproc_v2_result = SigprocV2Result.load_from_folder(self.inputs.sigproc)

        rad_filter_result = rad_filter(
            rad_filter_params,
            ims_import_result,
            sigproc_v2_result,
        )

        rad_filter_result.save()
