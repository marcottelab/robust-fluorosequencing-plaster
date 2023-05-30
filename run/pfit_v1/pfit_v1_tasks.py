from typing import Optional

import structlog

logger = structlog.get_logger()

from plaster.run.pfit_v1.paramopt.opt_routines import (
    ObjectiveFunction,
    OptimizeDirect,
    OptimizeSciPyMinimizePowell,
)
from plaster.run.pfit_v1.pfit_v1_params import ParameterEstimationParams
from plaster.run.pfit_v1.pfit_v1_result import (
    ParameterEstimationCheckOneResult,
    ParameterEstimationFitOneResult,
)

param_fit_v1_known_opts = {
    "Direct": OptimizeDirect,
    "SciPyMinimizePowell": OptimizeSciPyMinimizePowell,
}


def _build_opt_obj(
    params: ParameterEstimationParams, precheck_ind: Optional[int] = None
):
    dyeseqs = params.dyeseqs
    n_edmans = params.n_edmans
    marker_labels = params.marker_labels
    true_dyetracks = params.true_dyetracks
    true_dyetracks_count = params.true_dyetracks_count
    default_n_samples = params.default_n_samples

    obj_fnc_class = ObjectiveFunction(
        true_dyetracks,
        true_dyetracks_count,
        marker_labels,
        n_edmans,
        dyeseqs,
        None,
        default_n_samples,
    )

    plaster_x0 = [params.plaster_x0[k] for k in obj_fnc_class.pnms]

    if params.defaults is not None:
        bounds = []
        if precheck_ind is not None:
            bounds = plaster_x0[:]
            bounds[precheck_ind] = None
        else:
            for k in [x for x in params.defaults if x not in params.plaster_x0]:
                print("Unknown ParamInfo default:", k)

            for k in obj_fnc_class.pnms:
                try:
                    kk = params.defaults[k]
                    if kk.is_fixed:
                        bounds += [kk.initial_value]
                    else:
                        bounds += [(kk.bound_lower, kk.bound_upper)]
                except KeyError:
                    bounds += [None]

        for i, (a, b) in enumerate(zip(obj_fnc_class.pnms, bounds)):
            print(f"Constructed bounds {i + 1}/{len(bounds)}: {a} = {b}")

        obj_fnc_class.set_bounds(bounds)

    return obj_fnc_class, plaster_x0


def param_fit_v1_check(
    params: ParameterEstimationParams, param_ind: int, rep: int
) -> ParameterEstimationCheckOneResult:
    obj_fnc_class, plaster_x0 = _build_opt_obj(params, param_ind)
    obj_fnc = obj_fnc_class.objective_func

    x0 = obj_fnc_class.mask_x(plaster_x0)

    obj_fnc_class.resample()

    opt = OptimizeSciPyMinimizePowell(obj_fnc, obj_fnc_class, obj_fnc_class.bounds, x0)

    opt.go()
    opt.polish()

    return ParameterEstimationCheckOneResult(param_ind=param_ind, x=opt.x, fun=opt.fun)


def param_fit_v1_fit(
    params: ParameterEstimationParams,
    method: str,
    x0: list[float],
    resample: bool,
) -> ParameterEstimationFitOneResult:
    obj_fnc_class, plaster_x0 = _build_opt_obj(params)
    obj_fnc = obj_fnc_class.objective_func

    if not x0:
        x0 = obj_fnc_class.mask_x(plaster_x0)

    if resample:
        obj_fnc_class.resample()

    logger.info(f"Launching fitter {method}", bounds=obj_fnc_class.bounds)
    opt = param_fit_v1_known_opts[method](
        obj_fnc, obj_fnc_class, obj_fnc_class.bounds, x0
    )
    opt.go()
    opt.polish()

    result = ParameterEstimationFitOneResult(
        x0=x0, x=opt.x, fun=opt.fun, method=method, resample=resample
    )

    return result
