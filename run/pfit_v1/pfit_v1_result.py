from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin

from plaster.run.base_result_dc import BaseResultDC
from plaster.run.base_result_dc import generate_flyte_result_class as genf
from plaster.run.pfit_v1.pfit_v1_params import ParameterEstimationParams


@dataclass
class ParameterEstimationCheckOneResult(DataClassJsonMixin):
    param_ind: int
    x: list[float]
    fun: float


@dataclass
class ParameterEstimationCheckResult(BaseResultDC, DataClassJsonMixin):
    params: ParameterEstimationParams
    res: list[ParameterEstimationCheckOneResult]


FlyteParameterEstimationCheckResult = genf(ParameterEstimationCheckResult)


@dataclass
class ParameterEstimationFitOneResult(DataClassJsonMixin):
    # TODO: it would be nice to either hold a list of ParamInfo here
    # or otherwise provide more functionality to connect these generic
    # lists of floats to what params they are.  At present, this relies
    # on special knowledge of the order of params in the fitter implementation,
    # which makes it difficult to deal with in reports, especially when
    # there are some parameters that have been held fixed, such that these
    # lists are a subset of the parameters.
    #
    # For now, I have added some functionality to the top of the
    # param_fit_easy report that sets bounds on the ObjectiveFunction
    # class so that it may be used to expand_x during the report.

    x0: list[float]
    x: list[float]
    fun: float
    method: str  # the name of the minimization/optimization method used, e.g. "Direct"
    resample: bool  # was the input data set resampled before the fit?


@dataclass
class ParameterEstimationFitResult(BaseResultDC, DataClassJsonMixin):
    params: ParameterEstimationParams
    res: list[ParameterEstimationFitOneResult]
    bootstrap_fits: list[
        ParameterEstimationFitOneResult
    ]  # fits done on resampled input data for bootstrapping


FlyteParameterEstimationFitResult = genf(ParameterEstimationFitResult)
