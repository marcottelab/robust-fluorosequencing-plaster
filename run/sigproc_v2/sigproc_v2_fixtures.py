import numpy as np
from scipy.stats import norm

from plaster.run.priors import Priors
from plaster.run.sigproc_v2 import sigproc_v2_common
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result
from plaster.run.sim_v2.sim_v2_result import SimV2Result
from plaster.tools.c_common.c_common_tools import DytType
from plaster.tools.schema import check


def _sample_gaussian(mu, sigma, n_samples):
    return norm(mu, sigma).rvs(n_samples)


def synthetic_radmat_from_dytmat(
    dytmat, priors, n_channels, n_samples, remove_dark=True
):
    check.t(priors, Priors)
    check.array_t(dytmat, ndim=2, dtype=DytType)
    n_dyts, n_cols = dytmat.shape
    n_cycles = n_cols // n_channels
    assert n_cycles * n_channels == n_cols
    radmat = np.zeros((n_dyts * n_samples, n_cols))
    true_dyt_iz = np.zeros((n_dyts * n_samples,), dtype=int)

    for dyt_i, dyt in enumerate(dytmat):
        dyt_radmat = np.zeros((n_samples, n_cols))
        for ch_i in range(n_channels):
            mu = priors.get_mle(f"gain_mu.ch_{ch_i}")
            sigma = priors.get_mle(f"gain_sigma.ch_{ch_i}")
            bg_sigma = priors.get_mle(f"bg_sigma.ch_{ch_i}")

            assert sigma < 1.0
            sigma = sigma * mu

            for cy_i in range(n_cycles):
                col_i = ch_i * n_cycles + cy_i
                dyt_count = dyt[col_i]

                dyt_radmat[:, cy_i] = _sample_gaussian(
                    mu * dyt_count,
                    np.sqrt(dyt_count * sigma**2 + bg_sigma**2),
                    n_samples,
                )

        radmat[dyt_i * n_samples : (dyt_i + 1) * n_samples, :] = dyt_radmat

        true_dyt_iz[dyt_i * n_samples : (dyt_i + 1) * n_samples] = dyt_i

    n_radrows = radmat.shape[0]
    true_ks = np.ones((n_radrows,))
    row_k_sigma = priors.get_mle(f"row_k_sigma")
    if row_k_sigma > 0.0:
        true_ks = _sample_gaussian(1.0, row_k_sigma, n_radrows)
        radmat = radmat * true_ks[:, None]

    if remove_dark:
        good_mask = np.any(dytmat[true_dyt_iz] > 0, axis=1)
        radmat = radmat[good_mask]
        true_dyt_iz = true_dyt_iz[good_mask]
        true_ks = true_ks[good_mask]

    return radmat, true_dyt_iz, true_ks


class SigprocV2ResultFixture(SigprocV2Result):
    def _load_field_prop(self, field_i, prop):
        if prop == "signal_radmat":
            sim_v2_result = SimV2Result.from_prep_fixture(self.prep_result)
            radmat, true_dyt_iz, true_ks = synthetic_radmat_from_dytmat(
                sim_v2_result.train_dytmat,
                GainModel.test_fixture(),
                n_samples=100,
            )

        else:
            raise NotImplementedError(
                f"SigprocV2ResultFixture request un-handled prop {prop}"
            )

    @property
    def n_fields(self):
        return 1


def simple_sigproc_v2_result_fixture(prep_result):
    params = SigprocV2Params(
        calibration_file=None, mode=sigproc_v2_common.SIGPROC_V2_INSTRUMENT_ANALYZE
    )
    return SigprocV2ResultFixture(
        params=params,
        n_input_channels=1,
        n_channels=1,
        n_cycles=4,
        prep_result=prep_result,
    )
