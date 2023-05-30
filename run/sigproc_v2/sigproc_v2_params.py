import logging
from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import DataClassJsonMixin

from plaster.run.priors import ParamsDCWithPriors
from plaster.run.sigproc_v2 import sigproc_v2_common as common
from plaster.tools.utils import utils

log = logging.getLogger(__name__)


@dataclass
class SigprocV2Params(
    ParamsDCWithPriors, utils.DataclassUnpickleFromMunchMixin, DataClassJsonMixin
):
    """
    About Calibration:
        The long term goal of the calibration files is to dissociate
        the name of the file from the records (subjects) in the file.
        For now, we're going to load all records from the calibration file

    Warning: Flyte doesn't like Optional/Noneable ints currently (Apr 2022),
    so all int fields in this class with defaults must not be set to None.
    """

    mode: Optional[str] = None
    calibration_file: Optional[str] = None
    instrument_identity: Optional[str] = None
    divs: int = 5
    peak_mea: int = 11
    n_fields_limit: int = 0
    run_regional_balance: bool = True
    run_analysis_gauss2_fitter: bool = False
    run_aligner: bool = True
    run_per_cycle_peakfinder: bool = False
    low_inflection: float = 0.03
    low_sharpness: float = 50.0
    high_inflection: float = 0.50
    high_sharpness: float = 50.0
    self_calib: bool = False
    no_calib: bool = False
    save_full_signal_radmat_npy: bool = True
    n_cycles_limit: int = 0
    # ch_aln_override allows for a temporarily needed hack to bypass the calibration system
    ch_aln_override: List[List[float]] = field(default_factory=list)
    ch_for_alignment: int = -1
    run_fast_peak_finder: bool = False
    run_minimal_analysis_gauss2_fitter: bool = True
    null_hypothesis_precompute: bool = False
    source: Optional[str] = None

    def validate(self):
        if self.mode == common.SIGPROC_V2_ILLUM_CALIB:
            pass
            # ZBS: At the moment these checks are more trouble than they are worth
            # if local.path(self.calibration_file).exists():
            #     if not log.confirm_yn(
            #         f"\nCalibration file '{self.calibration_file}' already exists "
            #         "when creating a SIGPROC_V2_PSF_CALIB. Overwrite?",
            #         "y",
            #     ):
            #         raise SchemaValidationFailed(
            #             f"Not overwriting calibration file '{self.calibration_file}'"
            #         )

        else:
            # Analyzing
            if self.self_calib:
                assert (
                    self.calibration_file is None
                ), "In self-calibration mode you may not specify a calibration file"
                assert (
                    self.instrument_identity is None
                ), "In self-calibration mode you may not specify an instrument identity"
                assert (
                    self.no_calib is not True
                ), "In self-calibration mode you may not specify the no_calib option"

            # elif (
            #     not self.no_calib
            #     and self.calibration_file != ""
            #     and self.calibration_file is not None
            # ):
            #     self.calibration = Calib.load_file(
            #         self.calibration_file, self.instrument_identity
            #     )

            elif self.no_calib:
                assert (
                    self.no_calib_psf_sigma is not None
                ), "In no_calib mode you must specify an estimated no_calib_psf_sigma"

        return True
