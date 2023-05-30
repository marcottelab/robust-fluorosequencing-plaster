"""
Priors are parameters that are used by our models for
simulation, classification, and fitting.

Each prior has a distribution from which it can sample.

Some priors are not yet well modelled (as of May 2021) and
are simply constants based on old NBT paper data.

Some priors are based on calibration data from a specific instrument
while others are based on analysis from non-instrument devices
(ie mass-spec or other non-Erisyon devices.)

Some of priors are sampled from C code to speed and must be
exposed both with a Pythonic "sample()" function as well as
"parameters()" so that C code can implement the same function.

Priors are requested by name such that:
    1. If a calibration file is inline and contains the key, that is used
    2. If the name is in a task block of plaster_run.yaml that is used
    3. Otherwise, the defaults are used.

There's a variety of helpers that convert priors of certain kinds
into other useful objects. For example, the Regional Illumination
is parameterized with a few scalars but a helper can render that into
an image of the illumination plane.

Similarly, there are estimate functions which take data of some sort
and generate priors.

Naming Convention
-----------------

Prior names are hierarchical. Eg, suppose "atto647" is the name of
a dye.  It might have a known "p_bleach" which would
be called: "p_bleach.atto647"; but if that value is requested
and is not available then it will use the default for "p_bleach"
instead. This means that the reports must to be able to list the
ways in which a prior was found. For example:

    "p_bleach.foo123":
        Exact match not found; used "p_bleach" from defaults

To this end, the following are tracked within an instance of a prior:
    request_name:
        The name that calling code requested. Eg "p_bleach.foo123"
    matched_name:
        The name the prior lookup resolved. Eg "p_bleach"
    source_name:
        Where the lookup resolver got the value. Eg "defaults"

Conventions:
    .ch_{i}:  Channel

Hyper-parameters
----------------

Some Priors contain hyper-parameters which are not sampled but
are stored in order to inform the sampling; for example the pixel dimensions
of a camera when a calibration was taken. These start with "hyper_".

Fixtures
----------------

Fixtures are helpers that are used for testing.
Generally test want a single known value as opposed to drawing
from a distribution to the PriorsMLEFixtures create MLEPriors
and overloads the .get()
"""

import base64
import copy
import functools
import importlib
import pickle
import sys
from dataclasses import dataclass, field
from itertools import product

import marshmallow.fields
import numpy as np
from dataclasses_json import DataClassJsonMixin, config
from munch import Munch
from plumbum import local
from scipy import interpolate

from plaster.run.sigproc_v2.c_gauss2_fitter.gauss2_fitter import Gauss2Params
from plaster.tools.image import coord, imops
from plaster.tools.schema import check
from plaster.tools.schema.schema import Params
from plaster.tools.schema.schema import Schema
from plaster.tools.schema.schema import Schema as s
from plaster.tools.utils import utils

default_hyper_peak_mea = 15


class Prior:
    """
    Base class for any Prior.

    Typically a superclass so that class-specific helpers methods
    may be added.
    """

    def __init__(self):
        self.source = None

    def sample(self):
        """Implemented in subclasses"""
        raise NotImplementedError

    def parameters(self):
        """Implemented in subclasses"""
        raise NotImplementedError

    def _serialize(self, hyper_params=None, **params):
        """Convert into a dict form called a 'priors_desc'"""
        return dict(
            class_name=self.__class__.__name__, hyper_params=hyper_params, params=params
        )

    def set(self, **kwargs):
        """Implemented in subclasses"""
        raise NotImplementedError

    def deserialize(self, params):
        """
        For the case where there's no munging
        """
        self.set(**params)


class MLEPrior(Prior):
    """
    Used for parameters where we have a single point estimate of the value.
    These should be replaced over time with higher quality priors
    based on control or calibration experiments.
    """

    def __init__(self):
        super().__init__()
        self.value = None

    def set(self, value):
        self.value = value
        return self

    def serialize(self):
        return self._serialize(value=self.value)

    def sample(self):
        return self.value

    def parameters(self):
        return self.value


class GaussianPrior(Prior):
    """
    Parameters with a Gaussian prior.

    Remember, here we refer to the fact that the Bayesian prior
    (the model of our knowledge of the parameter) is Gaussian --
    not necessarily that the parameter ITSELF represents a Gaussian.
    """

    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None

    def set(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        return self

    def serialize(self):
        return self._serialize(
            mu=self.mu,
            sigma=self.sigma,
        )

    def sample(self, size=1):
        return np.random.normal(self.mu, self.sigma, size=size)

    def parameters(self):
        return Munch(
            mu=self.mu,
            sigma=self.sigma,
        )


class RegIllumPrior(Prior):
    @staticmethod
    def _reg_bal_func_exp_1(im_mea, X, cen_x, cen_y, falloff, gain):
        """
        Single exponential with a centering position
        """
        xx, yy = X
        return gain * np.exp(
            -falloff
            * (
                (yy / im_mea - cen_y / im_mea) ** 2
                + (xx / im_mea - cen_x / im_mea) ** 2
            )
        )

    fit_func = _reg_bal_func_exp_1

    def __init__(self, hyper_im_mea):
        super().__init__()
        self.hyper_im_mea = hyper_im_mea
        self.cen_x = None
        self.cen_y = None
        self.falloff = None

    def set(self, cen_x, cen_y, falloff):
        self.cen_x = cen_x
        self.cen_y = cen_y
        self.falloff = falloff
        return self

    def sample(self):
        raise NotImplementedError

    def serialize(self):
        return self._serialize(
            hyper_params=dict(hyper_im_mea=self.hyper_im_mea),
            cen_x=float(self.cen_x),
            cen_y=float(self.cen_y),
            falloff=float(self.falloff),
        )

    def parameters(self):
        return Munch(
            hyper_im_mea=self.hyper_im_mea,
            cen_x=self.cen_x,
            cen_y=self.cen_y,
            falloff=self.falloff,
        )

    def estimate(self, samples):
        """
        samples is an array spot radiometries of 3 columns: (y, x, val)
        """

        check.array_t(samples, ndim=2)
        assert samples.shape[1] == 3

        from scipy.optimize import curve_fit

        ys = samples[:, 0]
        xs = samples[:, 1]
        vals = samples[:, 2]
        cen = self.hyper_im_mea / 2

        def _fit_wrapper(*args, **kwargs):
            return self.fit_func(self.hyper_im_mea, *args, **kwargs)

        # I had previously seeded the initial_falloff parameter with 0.4 and found that
        # it failed to converge in some cases, for example, jim/jhm2021_06_17_01_tetraspec_3channel.
        # It seems happier to start at 0.0 so I'm setting it there for a while
        # but I would not be shocked if that causes it to fail in some other case.
        # That said, the jim/jhm2021_06_17_01_tetraspec_3channel is not even a single-count
        # experiment and is only being used in a self-calibration mode so it should
        # definitely not be setting the standard for this.
        initial_falloff = 0.0
        popt, pcov = curve_fit(
            _fit_wrapper,
            (xs, ys),
            vals,
            p0=(cen, cen, initial_falloff, np.nanmean(vals)),
        )
        self.cen_x, self.cen_y, self.falloff, _ = popt

    def render(self):
        xx, yy = np.meshgrid(
            np.arange(self.hyper_im_mea),
            np.arange(self.hyper_im_mea),
        )
        return self.fit_func(
            self.hyper_im_mea, (xx, yy), self.cen_x, self.cen_y, self.falloff, gain=1.0
        )

    @classmethod
    def fixture_uniform(cls, hyper_im_mea):
        self = cls(hyper_im_mea)
        self.cen_x = hyper_im_mea / 2
        self.cen_y = hyper_im_mea / 2
        self.falloff = 0.0
        return self


class ChannelAlignPrior(Prior):
    def __init__(self, hyper_n_channels):
        super().__init__()
        self.hyper_n_channels = hyper_n_channels
        self.ch_aln = None

    def set(self, ch_aln):
        check.array_t(ch_aln, ndim=2)
        assert ch_aln.shape[1] == 2
        assert np.all(
            ch_aln[0, :] == 0.0
        )  # Everything is calibrated relative to channel 0
        self.ch_aln = ch_aln
        return self

    def sample(self):
        return self.ch_aln

    def parameters(self):
        return Munch(hyper_n_channels=self.hyper_n_channels, ch_aln=self.ch_aln)

    def serialize(self):
        return self._serialize(
            hyper_params=dict(hyper_n_channels=self.hyper_n_channels),
            ch_aln=self.ch_aln.tolist(),
        )

    def deserialize(self, params):
        self.set(np.array(params.ch_aln))

    @classmethod
    def fixture_uniform(cls, hyper_n_channels):
        self = cls(hyper_n_channels)
        self.ch_aln = np.zeros((hyper_n_channels, 2))
        return self

    @classmethod
    def ch_remap(cls, src, dst_ch_i_to_src_ch_i):
        """
        Copy from src and remap channels.

        This involves some shifting around.
        All alignment is done relative to channel zero.
        So all of the other channels now need to be shifted to be
        relative to the NEW channel zero.

        Example:
            src_ch_aln = [
                [0, 0],
                [10, 20],
                [30, 40],
            ]

            dst_ch_i_to_src_ch_i = [2, 1]  # src ch 2 is becoming dst channel 0

            dst_ch_aln = [
                [0, 0],
                [30-10, 40-20]
            ]

        """
        src_new_zero_ch_i = dst_ch_i_to_src_ch_i[0]  # The src of the new channel 0

        n_dst_channels = len(dst_ch_i_to_src_ch_i)
        ch_aln = np.zeros((n_dst_channels, 2))
        for dst_ch_i, src_ch_i in enumerate(dst_ch_i_to_src_ch_i):
            ch_aln[dst_ch_i, :] = (
                src.ch_aln[src_ch_i, :] - src.ch_aln[src_new_zero_ch_i, :]
            )

        self = ChannelAlignPrior(n_dst_channels)
        self.set(ch_aln)
        return self


class RegPSFPrior(Prior):
    def __init__(self, hyper_im_mea, hyper_peak_mea, hyper_n_divs):
        super().__init__()
        self.hyper_im_mea = hyper_im_mea
        self.hyper_peak_mea = hyper_peak_mea
        self.hyper_n_divs = hyper_n_divs
        self.sigma_x = None
        self.sigma_y = None
        self.rho = None

    def set(self, sigma_x, sigma_y, rho):
        check.array_t(sigma_x, ndim=2, is_square=True)
        check.array_t(sigma_y, ndim=2, is_square=True)
        check.array_t(rho, ndim=2, is_square=True)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho
        return self

    def deserialize(self, params):
        self.sigma_x = np.array(params["sigma_x"])
        self.sigma_y = np.array(params["sigma_y"])
        self.rho = np.array(params["rho"])

    def serialize(self):
        return self._serialize(
            hyper_params=dict(
                hyper_im_mea=self.hyper_im_mea,
                hyper_peak_mea=self.hyper_peak_mea,
                hyper_n_divs=self.hyper_n_divs,
            ),
            sigma_x=self.sigma_x.tolist(),
            sigma_y=self.sigma_y.tolist(),
            rho=self.rho.tolist(),
        )

    def sample(self):
        raise NotImplementedError

    def parameters(self):
        return Munch(
            hyper_im_mea=self.hyper_im_mea,
            hyper_peak_mea=self.hyper_peak_mea,
            hyper_n_divs=self.hyper_n_divs,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            rho=self.rho,
        )

    def estimate(self, psf_ims):
        check.array_t(psf_ims, ndim=4)
        n_divs_y, n_divs_x, peak_mea_h, peak_mea_w = psf_ims.shape
        assert n_divs_y == n_divs_x and self.hyper_n_divs == n_divs_y
        assert peak_mea_h == peak_mea_w and self.hyper_peak_mea == peak_mea_h

        self.sigma_x = np.zeros((self.hyper_n_divs, self.hyper_n_divs))
        self.sigma_y = np.zeros((self.hyper_n_divs, self.hyper_n_divs))
        self.rho = np.zeros((self.hyper_n_divs, self.hyper_n_divs))
        for y in range(n_divs_y):
            for x in range(n_divs_x):
                im = psf_ims[y, x]
                if np.sum(im) > 0:
                    fit_params, _ = imops.fit_gauss2(im)
                    self.sigma_x[y, x] = fit_params[Gauss2Params.SIGMA_X]
                    self.sigma_y[y, x] = fit_params[Gauss2Params.SIGMA_Y]
                    self.rho[y, x] = fit_params[Gauss2Params.RHO]

    @functools.lru_cache()
    def _init_interpolation(self):
        center = self.hyper_im_mea / self.hyper_n_divs / 2.0
        coords = np.linspace(center, self.hyper_im_mea - center, self.hyper_n_divs)
        xx, yy = np.meshgrid(coords, coords)
        return (
            interpolate.interp2d(xx, yy, self.sigma_x, kind="cubic"),
            interpolate.interp2d(xx, yy, self.sigma_y, kind="cubic"),
            interpolate.interp2d(xx, yy, self.rho, kind="cubic"),
        )

    def render_at_loc(self, loc, amp=1.0, const=0.0, focus=1.0):
        interp_sigma_x_fn, interp_sigma_y_fn, interp_rho_fn = self._init_interpolation()
        loc_x = loc[1]
        loc_y = loc[0]
        sigma_x = interp_sigma_x_fn(loc_x, loc_y)[0]
        sigma_y = interp_sigma_y_fn(loc_x, loc_y)[0]
        rho = interp_rho_fn(loc_x, loc_y)[0]

        half_mea = self.hyper_peak_mea / 2.0
        corner_x = np.floor(loc_x - half_mea + 0.5)
        corner_y = np.floor(loc_y - half_mea + 0.5)
        center_x = loc_x - corner_x
        center_y = loc_y - corner_y

        im = imops.gauss2_rho_form(
            amp=amp,
            std_x=sigma_x,
            std_y=sigma_y,
            pos_x=center_x,
            pos_y=center_y,
            rho=rho,
            const=const,
            mea=self.hyper_peak_mea,
        )

        return im, (corner_y, corner_x)

    def render_into_im(self, im, loc, amp):
        psf_im, accum_to_loc = self.render_at_loc(loc, amp=1.0, const=0.0)
        imops.accum_inplace(im, amp * psf_im, loc=coord.YX(accum_to_loc), center=False)
        return psf_im

    def render_one_reg(self, div_y, div_x, frac_y=0.0, frac_x=0.0, const=0.0):
        assert 0 <= div_y < self.hyper_n_divs
        assert 0 <= div_x < self.hyper_n_divs
        assert 0 <= frac_x <= 1.0
        assert 0 <= frac_y <= 1.0

        im = imops.gauss2_rho_form(
            amp=1.0,
            std_x=self.sigma_x[div_y, div_x],
            std_y=self.sigma_y[div_y, div_x],
            # Note that the following must be integer divide because the
            # fractional component is relative to the lower-left corner (origin)
            pos_x=self.hyper_peak_mea // 2 + frac_x,
            pos_y=self.hyper_peak_mea // 2 + frac_y,
            rho=self.rho[div_y, div_x],
            const=const,
            mea=self.hyper_peak_mea,
        )

        # NORMALIZE to get an AUC exactly equal to 1.0
        return im / np.sum(im)

    def render(self):
        psf_ims = np.zeros(
            (
                self.hyper_n_divs,
                self.hyper_n_divs,
                self.hyper_peak_mea,
                self.hyper_peak_mea,
            )
        )
        for y, x in product(range(self.hyper_n_divs), range(self.hyper_n_divs)):
            psf_ims[y, x] = self.render_one_reg(y, x)
        return psf_ims

    def sample_params_grid(self, n_divs):
        # TODO: Optimize to avoid the python double loop. Numpy
        #   Something is wrong because when I try this in a notebook it is instant
        #   but here is taking almost 0.5 sec?
        samples = np.zeros((n_divs, n_divs, 3))

        interp_sigma_x_fn, interp_sigma_y_fn, interp_rho_fn = self._init_interpolation()
        space = np.linspace(0, self.hyper_im_mea, n_divs)
        for yi, y in enumerate(space):
            for xi, x in enumerate(space):
                sig_x = interp_sigma_x_fn(x, y)[0]
                sig_y = interp_sigma_y_fn(x, y)[0]
                rho = interp_rho_fn(x, y)[0]
                samples[yi, xi, :] = (sig_x, sig_y, rho)

        return samples

    @classmethod
    def fixture_uniform(
        cls, hyper_im_mea, hyper_peak_mea, hyper_n_divs, peak_width=1.8
    ):
        self = cls(hyper_im_mea, hyper_peak_mea, hyper_n_divs)
        self.sigma_x = peak_width * np.ones((self.hyper_n_divs, self.hyper_n_divs))
        self.sigma_y = peak_width * np.ones((self.hyper_n_divs, self.hyper_n_divs))
        self.rho = np.zeros((self.hyper_n_divs, self.hyper_n_divs))
        return self

    @classmethod
    def from_psf_ims(cls, im_mea, psf_ims):
        """
        Fit to a Gaussian for one-channel
        """
        check.array_t(psf_ims, ndim=4)
        divs_y, divs_x, peak_mea_h, peak_mea_w = psf_ims.shape
        assert divs_y == divs_x
        assert peak_mea_h == peak_mea_w
        reg_psf = cls(im_mea, peak_mea_h, divs_y)
        reg_psf.estimate(psf_ims)

        return reg_psf


class PriorsIncludeFile:
    """
    A yaml file that follows the priors_schema.
    This is the wrapper around what is callef "calibration" by gen.
    """

    def __init__(self, path):
        self.path = path
        self.priors_desc = utils.yaml_load_munch(local.path(path))
        Schema(Priors.priors_desc_schema).validate(self.priors_desc)


class Priors:
    """
    See above docs
    """

    priors_desc_schema = s.is_dict(
        elems=dict(
            class_name=s.is_str(),
            hyper_params=s.is_dict(),
            params=s.is_dict(required=False),
        ),
        required=False,
    )

    def add(self, name, prior, source=None):
        check.t(prior, Prior)
        if source is not None:
            prior.source = source
        assert prior.source is not None
        self.priors[name] = prior

    def delete_ch_specific_records(self):
        remove_keys = []
        for key in self.priors.keys():
            parts = key.split(".")
            if len(parts) > 1:
                if parts[1].startswith("ch_"):
                    remove_keys += [key]
        self.priors = {
            key: val for key, val in self.priors.items() if key not in remove_keys
        }

    def _instanciate_prior(
        self, source, name, class_name, hyper_params, params, overwrite=False
    ):
        """
        ADD a prior instance of class_name from source into name
        """
        parts = class_name.split(".")
        if len(parts) == 1:
            # Use this module
            module = sys.modules[__name__]
        else:
            module = importlib.import_module(".".join(parts[0:-1]))

        klass = getattr(module, parts[-1])
        instance = klass(**(hyper_params or {}))
        if isinstance(instance, PriorsIncludeFile):
            # RECURSE for include file
            self._instanciate_priors_desc(
                f"Include file '{instance.path}'", instance.priors_desc
            )
        else:
            instance.source = source
            instance.deserialize(params)
            if name in self.priors and not overwrite:
                raise ValueError(f"Duplicate prior name '{name}'")
            self.add(name, instance)

    def _instanciate_priors_desc(self, source, priors_desc):
        for name, desc in priors_desc.items():
            self._instanciate_prior(
                source,
                name,
                desc["class_name"],
                desc.get("hyper_params"),
                desc.get("params"),
            )

    @classmethod
    def from_priors_desc(cls, priors_desc, source=""):
        """
        Create Priors from a desc, for example from a task parameters block
        These can include "PriorsIncludeFile" which recursively add
        """
        priors = Priors()
        priors._instanciate_priors_desc(source, priors_desc)
        return priors

    def serialize(self):
        descs = []
        for prior_name, prior_instance in self.priors.items():
            s = prior_instance.serialize()
            s["name"] = prior_name
            descs += [s]
        return descs

    def __init__(self, hyper_im_mea=None, hyper_n_channels=None):
        self._default_reg_illum = None
        self._default_reg_psf = None
        self._default_ch_aln = None

        if hyper_im_mea is not None:
            self._default_reg_illum = RegIllumPrior(hyper_im_mea).set(
                hyper_im_mea / 2, hyper_im_mea / 2, 0.0
            )

            hyper_n_divs = 5
            sigma_x = np.full((hyper_n_divs, hyper_n_divs), 1.8)
            sigma_y = np.full((hyper_n_divs, hyper_n_divs), 1.8)
            rho = np.full((hyper_n_divs, hyper_n_divs), 0.0)
            self._default_reg_psf = RegPSFPrior(
                hyper_im_mea,
                hyper_peak_mea=default_hyper_peak_mea,
                hyper_n_divs=hyper_n_divs,
            ).set(sigma_x=sigma_x, sigma_y=sigma_y, rho=rho)

        if hyper_n_channels is not None:
            self._default_ch_aln = ChannelAlignPrior(hyper_n_channels).set(
                np.zeros((hyper_n_channels, 2))
            )

        self._defaults = {
            "reg_illum": self._default_reg_illum,
            "reg_psf": self._default_reg_psf,
            "ch_aln": self._default_ch_aln,
            "gain_mu": MLEPrior().set(value=7500.0),
            "gain_sigma": MLEPrior().set(value=0.0),
            "bg_mu": MLEPrior().set(value=0.0),
            "bg_sigma": MLEPrior().set(value=100.0),
            "row_k_sigma": MLEPrior().set(value=0.15),
            "p_edman_failure": MLEPrior().set(value=0.06),
            "p_detach": MLEPrior().set(value=0.05),
            "p_bleach": MLEPrior().set(value=0.05),
            "p_non_fluorescent": MLEPrior().set(value=0.07),
        }

        self.priors = {}

    def get_exact(self, request_name):
        """
        Look up exact match or return None
        """
        if request_name in self.priors:
            return Munch(
                request_name=request_name,
                matched_name=request_name,
                prior=self.priors[request_name],
            )

        return None

    def get(self, request_name):
        """
        Search for a request_name looking up the naming tree for a match if needed
        """
        parts = request_name.split(".")
        for i in range(len(parts), 0, -1):
            # SEARCH up the hierarchy
            matched_name = ".".join(parts[0:i])
            if matched_name in self.priors:
                return Munch(
                    request_name=request_name,
                    matched_name=matched_name,
                    prior=self.priors[matched_name],
                )

        # Not found, look in defaults
        matched_name = parts[0]
        if matched_name in self._defaults:
            default = self._defaults[parts[0]]
            if default is not None:
                default.source = "defaults"
                return Munch(
                    request_name=request_name,
                    matched_name=matched_name,
                    prior=default,
                )

        raise KeyError(f"Prior '{request_name}' not resolved")

    def enumerate_names(self):
        return list(self.priors.keys())

    def remove(self, name):
        del self.priors[name]

    def update(self, other, source):
        for key, value in other.priors.items():
            value = copy.deepcopy(value)
            value.source = source
            self.priors[key] = value

    def get_distr(self, request_name):
        """
        Like get() but returns the distribution object only (without the other metadata)
        """
        p = self.get(request_name)
        return p.prior

    def get_sample(self, request_name):
        """
        Like get() but returns a sample from the distribution
        """
        p = self.get(request_name)
        return p.prior.sample()

    def get_mle(self, request_name, **kwargs):
        """
        Like get() but returns the MLE.
        For now, this means that the prior must come from an MLEPrior
        """
        p = self.get(request_name)
        assert isinstance(p.prior, MLEPrior)
        return p.prior.sample()

    def helper_illum_model(self, n_channels):
        """
        Used by nn_v2 to load the C context with cols: gain_mu, gain_sigma, bg_mu, bg_sigma
        One row per channel.

        Note that this is still providing the older functionality of returning
        a MLE value for the parameters. Eventually this is going to be changed
        so that the nn_v2 will have a parametric description of the the prior
        so that it can draw samples instead of using the MLE.
        """

        illum_model = np.zeros((n_channels, 4))
        for ch_i in range(n_channels):
            prior = self.get(f"gain_mu.ch_{ch_i}")
            assert isinstance(prior.prior, MLEPrior)
            gain_mu = prior.prior.sample()
            illum_model[ch_i, 0] = gain_mu

            prior = self.get(f"gain_sigma.ch_{ch_i}")
            assert isinstance(prior.prior, MLEPrior)
            gain_sigma = prior.prior.sample()
            illum_model[ch_i, 1] = gain_sigma

            illum_model[ch_i, 2] = 0.0

            prior = self.get(f"bg_sigma.ch_{ch_i}")
            assert isinstance(prior.prior, MLEPrior)
            bg_sigma = prior.prior.sample()
            illum_model[ch_i, 3] = bg_sigma

        return illum_model

    @classmethod
    def copy(cls, src):
        self = cls()
        self.priors = copy.deepcopy(src.priors)
        return self


class PriorsMLEFixtures(Priors):
    """
    Uses MLE priors for all
    """

    def __init__(self, hyper_im_mea=None, hyper_n_channels=None, **mle_priors):
        super().__init__(hyper_im_mea, hyper_n_channels)

        for name, val in mle_priors.items():
            self._instanciate_prior(
                f"fixture('{name}')", name, "MLEPrior", None, dict(value=val)
            )

    def replace_mle_fixture(self, name, val):
        self.remove(name)
        self._instanciate_prior(
            f"fixture('{name}')", name, "MLEPrior", None, dict(value=val)
        )

    def get_sample(self, request_name):
        found_prior = super().get(request_name)
        prior = found_prior.prior
        check.t(prior, MLEPrior)
        return prior.sample()

    @classmethod
    def illumination(
        cls, row_k_sigma=0.15, gain_mu=7000.0, gain_sigma=0.14, bg_sigma=200.0
    ):
        return PriorsMLEFixtures(
            row_k_sigma=row_k_sigma,
            gain_mu=gain_mu,
            gain_sigma=gain_sigma,
            bg_sigma=bg_sigma,
        )

    @classmethod
    def illumination_lognormal(
        cls, row_k_sigma=0.15, gain_mu=7000.0, gain_sigma=0.20, bg_sigma=200.0
    ):
        return PriorsMLEFixtures(
            row_k_sigma=row_k_sigma,
            gain_mu=gain_mu,
            gain_sigma=gain_sigma,
            bg_sigma=bg_sigma,
        )

    @classmethod
    def fixture_no_errors(
        cls, hyper_im_mea=None, hyper_n_channels=None, reg_psf=None, **kwargs
    ):
        priors = PriorsMLEFixtures(
            hyper_im_mea=hyper_im_mea,
            hyper_n_channels=hyper_n_channels,
            row_k_sigma=0.0,
            gain_mu=7500.0,
            gain_sigma=0.0,
            bg_sigma=0.0,
            p_edman_failure=0.0,
            p_bleach=0.0,
            p_detach=0.0,
            p_non_fluorescent=0.0,
        )

        if reg_psf is not None:
            priors.add("reg_psf", reg_psf, source="fixture_no_errors")

        for key, val in kwargs.items():
            priors._instanciate_prior(
                "fixture_no_errors",
                key,
                "MLEPrior",
                None,
                dict(value=val),
                overwrite=True,
            )

        return priors

    @classmethod
    def illumination_multi_channel(
        cls, gain_mus_ch, gain_sigmas_ch, bg_sigmas_ch, row_k_sigma=0.15
    ):
        kws = {}
        for ch_i, (gain_mu, gain_sigma, bg_sigma) in enumerate(
            zip(gain_mus_ch, gain_sigmas_ch, bg_sigmas_ch)
        ):
            kws[f"gain_mu.ch_{ch_i}"] = gain_mu
            kws[f"gain_sigma.ch_{ch_i}"] = gain_sigma
            kws[f"bg_sigma.ch_{ch_i}"] = bg_sigma

        return PriorsMLEFixtures(row_k_sigma=row_k_sigma, **kws)

    @classmethod
    def nbt_defaults(cls):
        """
        Based on values from the NBT paper.
        """
        return PriorsMLEFixtures(
            row_k_sigma=0.15,
            gain_mu=7500.0,
            gain_sigma=0.16,
            bg_sigma=700.0,
            p_edman_failure=0.06,
            p_bleach=0.05,
            p_detach=0.05,
            p_non_fluorescent=0.07,
        )

    @classmethod
    def val_defaults(cls):
        """
        Based on values from the NBT paper plus (normal as opposed to lognormal)
        illumnation vaguely similar to val
        """
        return PriorsMLEFixtures(
            row_k_sigma=0.15,
            gain_mu=10_000.0,
            gain_sigma=1000.0,
            bg_sigma=500.0,
            p_edman_failure=0.06,
            p_bleach=0.05,
            p_detach=0.05,
            p_non_fluorescent=0.07,
        )

    @classmethod
    def reg_psf_uniform(
        cls,
        hyper_im_mea=512,
        hyper_peak_mea=default_hyper_peak_mea,
        hyper_n_divs=5,
        hyper_peak_sigma=1.8,
    ):
        reg_psf = RegPSFPrior(hyper_im_mea, hyper_peak_mea, hyper_n_divs)
        reg_psf.set(
            sigma_x=np.full((hyper_n_divs, hyper_n_divs), hyper_peak_sigma),
            sigma_y=np.full((hyper_n_divs, hyper_n_divs), hyper_peak_sigma),
            rho=np.full((hyper_n_divs, hyper_n_divs), 0.0),
        )
        return reg_psf

    @classmethod
    def reg_psf_variable(
        cls,
        hyper_im_mea=512,
        hyper_peak_mea=default_hyper_peak_mea,
        hyper_n_divs=5,
        hyper_peak_sigma=1.8,
        sigma_mag=0.0,
        rho_mag=1.0,
    ):
        reg_psf = RegPSFPrior(hyper_im_mea, hyper_peak_mea, hyper_n_divs)

        sigma_x = np.full((hyper_n_divs, hyper_n_divs), hyper_peak_sigma)
        # sigma_y = np.full((hyper_n_divs, hyper_n_divs), hyper_peak_sigma)
        rho = np.full((hyper_n_divs, hyper_n_divs), 0.0)

        center = hyper_im_mea / 2.0
        yy, xx = np.meshgrid(
            (np.linspace(0, hyper_im_mea, hyper_n_divs) - center) / hyper_im_mea,
            (np.linspace(0, hyper_im_mea, hyper_n_divs) - center) / hyper_im_mea,
        )

        sigma_x = sigma_x * (1 + sigma_mag * np.abs(xx * yy))
        sigma_y = sigma_x

        rho = rho_mag * xx * yy

        reg_psf.set(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rho=rho,
        )
        return reg_psf

    @classmethod
    def reg_illum_uniform(cls, hyper_im_mea=512):
        reg_illum = RegIllumPrior(hyper_im_mea)
        reg_illum.set(
            cen_x=hyper_im_mea / 2.0,
            cen_y=hyper_im_mea / 2.0,
            falloff=0.0,
        )
        return reg_illum

    @classmethod
    def reg_illum_variable(cls, hyper_im_mea=512, falloff=0.5, rand_offset=0.0):
        reg_illum = RegIllumPrior(hyper_im_mea)
        offset = np.random.uniform(0.0, rand_offset * hyper_im_mea, size=2)
        reg_illum.set(
            cen_y=hyper_im_mea / 2.0 + offset[0],
            cen_x=hyper_im_mea / 2.0 + offset[1],
            falloff=falloff,
        )
        return reg_illum


class ParamsAndPriors(Params):
    def __init__(self, **kwargs):
        assert "priors_list" not in kwargs
        source = kwargs.pop("source", None)

        # Remove the priors before validating
        priors = kwargs.pop("priors", None)
        super().__init__(**kwargs)
        self.priors = None

        priors_desc = kwargs.pop("priors_desc", None)
        if priors and priors_desc:
            raise TypeError(
                "priors and priors_desc specified. At most one must be passed."
            )

        if isinstance(priors, Priors):
            self.priors = priors
        elif isinstance(priors_desc, dict):
            # Put it back in and let the super init take care of converting it
            self.priors = Priors.from_priors_desc(priors_desc, source)
        elif priors_desc is None and priors is None:
            self.priors = Priors()
        else:
            # priors_list must be None or a valid list of encoded priors
            raise TypeError(f"Incorrect priors initialization")


def dc_encode_priors(priors: Priors) -> str:
    # I don't like it either
    return base64.b64encode(pickle.dumps(priors)).decode("utf-8")


def dc_decode_priors(priors_value: str) -> Priors:
    return pickle.loads(base64.b64decode(priors_value.encode("utf-8")))


@dataclass
class ParamsDCWithPriors(DataClassJsonMixin):
    priors: Priors = field(
        default_factory=Priors,
        metadata=config(
            mm_field=marshmallow.fields.String(),
            encoder=dc_encode_priors,
            decoder=dc_decode_priors,
        ),
    )
