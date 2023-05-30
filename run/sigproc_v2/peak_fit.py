import numpy as np

from plaster.run.priors import RegPSFPrior
from plaster.run.sigproc_v2.c_gauss2_fitter import gauss2_fitter
from plaster.run.sigproc_v2.c_gauss2_fitter.gauss2_fitter import (
    AugmentedGauss2Params,
    Gauss2Params,
)


def fit_peaks_one_im(im, locs, reg_psf: RegPSFPrior):
    """
    Run the fast Gauss2 fitter

    Arguments:
        im: A single image
        locs: The locations where to run the fitter
        reg_psf: RegionalPSF (contains the PSF for all channels)

    Returns:
        fit_params: ndarray(n_locs, , AugmentedGauss2Params.N_FULL_PARAMS)
    """
    assert isinstance(reg_psf, RegPSFPrior)

    # reg_yx = np.clip(
    #     np.floor(reg_psf.hyper_n_divs * locs / im.shape[0]).astype(int),
    #     a_min=0,
    #     a_max=reg_psf.hyper_n_divs - 1,
    # )

    n_locs = len(locs)
    guess_params = np.zeros((n_locs, AugmentedGauss2Params.N_FULL_PARAMS))

    # COPY over parameters by region for each peak. use [0, 0] as a guess
    # no need to be more specific since the point is to fit
    guess_params[:, Gauss2Params.SIGMA_X] = reg_psf.sigma_x[0, 0]
    guess_params[:, Gauss2Params.SIGMA_Y] = reg_psf.sigma_y[0, 0]
    guess_params[:, Gauss2Params.RHO] = reg_psf.rho[0, 0]

    # CENTER
    guess_params[:, Gauss2Params.CENTER_X] = reg_psf.hyper_peak_mea / 2
    guess_params[:, Gauss2Params.CENTER_Y] = reg_psf.hyper_peak_mea / 2

    # Pass zero to amp and offset to force the fitter to make its own guess
    guess_params[:, Gauss2Params.AMP] = 0.0
    guess_params[:, Gauss2Params.OFFSET] = 0.0

    fit_params, _ = gauss2_fitter.fit_image(
        im, locs, guess_params, reg_psf.hyper_peak_mea
    )
    return fit_params
