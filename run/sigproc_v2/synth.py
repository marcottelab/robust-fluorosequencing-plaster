import copy
import logging
import math
import random
from typing import List

import cv2
import numpy as np
import structlog

from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.ims_import.ims_import_worker import OUTPUT_NP_TYPE
from plaster.run.priors import RegIllumPrior, RegPSFPrior
from plaster.run.sigproc_v2.c_gauss2_fitter import gauss2_fitter
from plaster.tools.image import imops
from plaster.tools.image.coord import XY
from plaster.tools.schema import check
from plaster.tools.utils import utils
from plaster.tools.utils.tmp import tmp_folder

logger = structlog.get_logger()

# see comment below, above "PeaksModelPSF" regarding why this is commented out
# from plaster.run.sigproc_v2.psf_sample import psf_sample


class Synth:
    """
    Generate synthetic images for testing.

    This system is organized so that synthetic image(s) is
    delayed until the render() command is called. This allows
    for "reaching in" to the state and messing with it.

    Example, suppose that in some strange test you need to
    have a position of a certain peak location in very specific
    places for the test. To prevent a proliferation of one-off
    methods in this class, the idea is that you can use the
    method that creates two peaks and then "reach in" to
    tweak the positions directly before render.

    Examples:
        with Synth() as s:
            p = PeaksModelGaussian()
            p.locs_randomize()
            CameraModel(100, 2)
            s.render_chcy()

    """

    synth = None

    def __init__(
        self,
        n_fields=1,
        n_channels=1,
        n_cycles=1,
        dim=(512, 512),
        save_as=None,
    ):
        self.n_fields = n_fields
        self.n_channels = n_channels
        self.n_cycles = n_cycles
        self.dim = dim
        self.models = []
        self.aln_offsets = np.random.uniform(-20, 20, size=(self.n_cycles, 2))
        self.aln_offsets[0] = (0, 0)
        self.ch_aln = None
        Synth.synth = self

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        Synth.synth = None
        if exception_type is not None:
            raise exception_type(exception_value)

    def zero_aln_offsets(self):
        self.aln_offsets = np.zeros((self.n_cycles, 2))

    def add_model(self, model):
        self.models += [model]

    def _save_debug(self):
        path = "/erisyon/plaster/_synth_debug.npy"
        np.save(path, self.render_flchcy())
        logger.info("debug written", path=path)

    def render_chcy(self, fl_i=0):
        """
        Returns only chcy_ims (first field)
        """
        ims = np.zeros((self.n_channels, self.n_cycles, *self.dim))
        for ch_i in np.arange(self.n_channels):
            ch_aln_offset = 0.0
            if self.ch_aln is not None:
                ch_aln_offset = self.ch_aln[ch_i]

            for cy_i in np.arange(self.n_cycles):
                im = ims[ch_i, cy_i]
                for model in self.models:
                    model.render(
                        im, fl_i, ch_i, cy_i, self.aln_offsets[cy_i] + ch_aln_offset
                    )
                ims[ch_i, cy_i] = im

        return ims

    def render_flchcy(self):
        flchcy_ims = np.zeros(
            (self.n_fields, self.n_channels, self.n_cycles, *self.dim)
        )
        for fl_i in range(self.n_fields):
            flchcy_ims[fl_i] = self.render_chcy()
        return flchcy_ims

    def scale_peaks_by_max(self):
        """
        For some tests it is nice to know that the max brightness of a peak
        instead of the area under the curve.
        """
        self.peak_ims = [peak_im / np.max(peak_im) for peak_im in self.peak_ims]

    def channel_aln_offsets(self, ch_aln):
        """
        TODO: This probably should move to Peaks so that it conforms
              to similar pattern of channel_scale_factor()
        """
        check.array_t(ch_aln, shape=(self.n_channels, 2))
        self.ch_aln = ch_aln


class BaseSynthModel:
    def __init__(self):
        self.dim = Synth.synth.dim
        Synth.synth.add_model(self)

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        pass


class PeaksModel(BaseSynthModel):
    def __init__(self, n_peaks=1000):
        super().__init__()
        self.n_channels = Synth.synth.n_channels
        self.n_cycles = Synth.synth.n_cycles
        self.n_fields = Synth.synth.n_fields
        self.n_peaks = n_peaks
        self.locs = np.zeros((n_peaks, 2))
        self.row_k = np.ones((n_peaks,))
        self.counts = np.ones((n_peaks, self.n_cycles, self.n_channels), dtype=int)
        self._amps = np.ones((n_peaks, self.n_cycles))
        self._channel_scale_factor = None
        self._channel_offset = None

    # locs related
    # ------------------------------------------------------------------------
    def locs_randomize(self):
        self.locs = np.random.uniform(0, self.dim, (self.n_peaks, 2))
        return self

    def locs_randomize_no_subpixel(self):
        self.locs = np.floor(np.random.uniform(0, self.dim, (self.n_peaks, 2))).astype(
            float
        )
        return self

    def locs_randomize_away_from_edges(self, dist=15):
        self.locs = np.random.uniform(
            [dist, dist], np.array(self.dim) - dist, (self.n_peaks, 2)
        )
        return self

    def locs_grid(self, pad=10):
        step = self.dim[0] // math.floor(math.sqrt(self.n_peaks))
        y = np.arange(pad, self.dim[0] - pad, step)
        x = np.arange(pad, self.dim[1] - pad, step)
        self.locs = np.array(np.meshgrid(x, y)).T.reshape(-1, 2).astype(float)
        return self

    def locs_center(self):
        # Good for one peak only
        assert self.n_peaks == 1
        self.locs = [
            (
                self.dim[0] / 2,
                self.dim[1] / 2,
            )
        ]
        return self

    def locs_add_random_subpixel(self):
        self.locs += np.random.uniform(-1, 1, self.locs.shape)
        return self

    def remove_near_edges(self, dist=20):
        self.locs = np.array(
            [
                loc
                for loc in self.locs
                if dist < loc[0] < self.dim[0] - dist
                and dist < loc[1] < self.dim[1] - dist
            ]
        )
        return self

    # count related. Use this preferentially over direct amps assignment
    # ------------------------------------------------------------------------
    def counts_uniform(self, cnt):
        self.counts = cnt * np.ones(
            (self.n_peaks, self.n_cycles, self.n_channels), dtype=int
        )
        return self

    def bleach(self, p_bleach):
        r = np.random.uniform(
            0.0, 1.0, size=(self.n_peaks, self.n_cycles, self.n_channels)
        )
        decrement = np.where(r < p_bleach, -1, 0)
        self.counts[:, 1:, :] = decrement[:, 1:, :]
        self.counts = np.clip(np.cumsum(self.counts, axis=1), a_min=0, a_max=None)
        return self

    def lognormal(self, beta, sigma):
        # Convert cnt to _amps with lognormal
        with utils.np_no_warn():
            self._amps = np.nan_to_num(
                np.random.lognormal(
                    np.log(beta * self.counts), sigma, size=self.counts.shape
                )
            )
        return self

    def gain_constant(self, gain):
        self._amps = gain * self.counts
        return self

    # dyt related
    # ------------------------------------------------------------------------
    def dyt_uniform(self, dyt):
        dyt = np.array(dyt)
        self.counts = np.repeat(dyt[:, None], self.n_peaks, axis=(1, 2))
        return self

    def dyt_random_choice(self, dyts, probs):
        """
        dyts is like:
            [
                [3, 2, 2, 1],
                [2, 1, 1, 0],
                [1, 0, 0, 0],
            ]

        and each row of dyts has a probability for it
            probs: [0.5, 0.3, 0.2]
        """
        dyts = np.array(dyts)
        check.array_t(dyts, ndim=2)
        assert dyts.shape[0] == len(probs)
        self.dyt_iz = np.random.choice(len(dyts), size=self.n_peaks, p=probs)
        self.counts = dyts[self.dyt_iz]
        return self

    def multichannel_dyt_random_choice(self, dyts, probs):
        """
        dyts: (n_options, n_channels, n_cycles)
            [
                [[1, 1, 1],  [0, 0, 0]],  # On in ch 0
                [[0, 0, 0],  [1, 1, 1]],  # On in ch 1
                [[1, 1, 1],  [1, 1, 1]],  # Shared
            ],

        probs: (n_options)
            [0.80, 0.18, 0.02],  # Choices

        """

        dyts = np.array(dyts)
        check.array_t(dyts, ndim=3)
        _n_options, _n_channels, _n_cycles = dyts.shape

        assert _n_channels == self.n_channels
        assert _n_cycles == self.n_cycles

        probs = np.array(probs)
        check.array_t(probs, ndim=1)
        assert probs.shape[0] == _n_options

        self.counts = np.zeros((self.n_peaks, self.n_cycles, self.n_channels))
        self.dyt_iz = np.random.choice(_n_options, size=self.n_peaks, p=probs)
        for ch_i in range(self.n_channels):
            self.counts[:, :, ch_i] = dyts[self.dyt_iz, ch_i]

        return self

    def multichannel_independent_random(self, n_peaks_per_ch):
        """
        n_peaks_per_ch: the number of peaks on each channel, thus
        the self.n_peaks will end up n_channels * n_peaks_per_ch
        """
        self.n_peaks = n_peaks_per_ch * self.n_channels
        self.counts = np.zeros((self.n_peaks, self.n_cycles, self.n_channels))
        for ch_i in range(self.n_channels):
            self.counts[
                ch_i * n_peaks_per_ch : (ch_i + 1) * n_peaks_per_ch, :, ch_i
            ] = np.ones((n_peaks_per_ch, self.n_cycles))

        self.locs = np.random.uniform(0, self.dim, (self.n_peaks, 2))
        self.row_k = np.ones((self.n_peaks,))

        return self

    # amps related (Prefer the above over these direct manipulations)
    # ------------------------------------------------------------------------
    def amps_constant(self, val):
        """
        Set amps directly. If you want to use dyts, use gain_constant
        """
        self._amps = val * np.ones((self.n_peaks,))
        return self

    def amps_randomize(self, mean=1000, std=10):
        self._amps = mean + std * np.random.normal(size=(self.n_peaks, self.n_cycles))
        return self

    def channel_scale_factor(self, ch_scale):
        self._channel_scale_factor = ch_scale
        return self

    # row_k related
    # ------------------------------------------------------------------------
    def row_k_randomize(self, mean=1.0, std=0.2):
        self.row_k = np.random.normal(loc=mean, scale=std, size=(self.n_peaks,))
        return self


class PeaksModelPSF(PeaksModel):
    """Sample from a RegPSF"""

    def __init__(self, reg_psf: RegPSFPrior, **kws):
        check.t(reg_psf, (type(None), RegPSFPrior))
        self.reg_psf = reg_psf
        super().__init__(**kws)

    def render(self, im, fl_i, ch_i, cy_i, aln_offset, reg_psf=None):
        if reg_psf is None:
            reg_psf = self.reg_psf

        super().render(im, fl_i, ch_i, cy_i, aln_offset)

        ch_scale = 1.0
        if self._channel_scale_factor is not None:
            ch_scale = self._channel_scale_factor[ch_i]

        for peak_i, (loc, amp, k) in enumerate(zip(self.locs, self._amps, self.row_k)):
            loc = loc + aln_offset

            if isinstance(amp, np.ndarray):
                if amp.ndim == 1:
                    amp = amp[cy_i]
                elif amp.ndim == 2:
                    amp = amp[cy_i, ch_i]
                else:
                    raise ValueError("unknown amp arg")

            # TODO: Use     def render_into_im(self, im, loc, amp):
            reg_psf.render_into_im(im, loc, amp=ch_scale * k * amp)


class ChPeaksModelPSF(PeaksModelPSF):
    """Multi-channel version of PeaksModelPSF"""

    def __init__(self, ch_reg_psfs: List[RegPSFPrior], **kws):
        check.list_t(ch_reg_psfs, RegPSFPrior)
        self.ch_reg_psfs = ch_reg_psfs
        super().__init__(None, **kws)

    def render(self, im, fl_i, ch_i, cy_i, aln_offset, reg_psf=None):
        assert reg_psf is None
        super().render(im, fl_i, ch_i, cy_i, aln_offset, reg_psf=self.ch_reg_psfs[ch_i])


class PeaksModelGaussian(PeaksModel):
    def __init__(self, **kws):
        self.mea = kws.pop("mea", 11)
        super().__init__(**kws)
        self.std = None
        self.std_x = None
        self.std_y = None
        self._channel_peak_width_scale_factor = None

    def uniform_width_and_heights(self, width=1.5, height=1.5):
        self.std_x = [width for _ in self.locs]
        self.std_y = [height for _ in self.locs]
        return self

    def channel_peak_width_scale_factor(self, channel_factor):
        self._channel_peak_width_scale_factor = np.array(channel_factor)
        return self

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        n_locs = len(self.locs)
        assert n_locs == len(self.std_x)
        assert n_locs == len(self.std_y)

        super().render(im, fl_i, ch_i, cy_i, aln_offset)

        locs = self.locs + aln_offset

        _amps = np.array(self._amps)
        if _amps.ndim == 3:
            amps = _amps[:, cy_i, ch_i]
        elif _amps.ndim == 2:
            amps = _amps[:, cy_i]
        else:
            amps = _amps

        std_xs = np.array(self.std_x)
        std_ys = np.array(self.std_y)

        ch_peak_width = 1.0
        if self._channel_peak_width_scale_factor is not None:
            ch_peak_width = self._channel_peak_width_scale_factor[ch_i]

        ch_scale = 1.0
        if self._channel_scale_factor is not None:
            ch_scale = self._channel_scale_factor[ch_i]

        gauss2_fitter.synth_image(
            im,
            self.mea,
            locs,
            ch_scale * self.row_k * amps,
            ch_peak_width * std_xs,
            ch_peak_width * std_ys,
        )


class PeaksModelGaussianCircular(PeaksModelGaussian):
    def __init__(self, **kws):
        super().__init__(**kws)
        self.std = 1.0

    def widths_uniform(self, width=1.5):
        self.std_x = [width for _ in self.locs]
        self.std_y = copy.copy(self.std_x)
        return self

    def widths_variable(self, width=1.5, scale=0.1):
        self.std_x = [random.gauss(width, scale) for _ in self.locs]
        self.std_y = copy.copy(self.std_x)
        return self

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        # self.covs = np.array([(std ** 2) * np.eye(2) for std in self.stds])
        super().render(im, fl_i, ch_i, cy_i, aln_offset)


class PeaksModelGaussianAstigmatism(PeaksModelGaussian):
    def __init__(self, strength, **kws):
        raise DeprecationWarning
        super().__init__(**kws)
        self.strength = strength
        center = np.array(self.dim) / 2
        d = self.dim[0]
        for loc_i, loc in enumerate(self.locs):
            loc = loc + aln_offset
            delta = center - loc
            a = np.sqrt(np.sum(delta**2))
            r = 1 + strength * a / d
            pc0 = delta / np.sqrt(delta.dot(delta))
            pc1 = np.array([-pc0[1], pc0[0]])
            cov = np.eye(2)
            cov[0, 0] = r * pc0[1]
            cov[1, 0] = r * pc0[0]
            cov[0, 1] = pc1[1]
            cov[1, 1] = pc1[0]
            self.covs[loc_i, :, :] = cov


class IlluminationQuadraticFalloffModel(BaseSynthModel):
    def __init__(self, center=(0.5, 0.5), width=1.2):
        super().__init__()
        self.center = center
        self.width = width

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        super().render(im, fl_i, ch_i, cy_i, aln_offset)
        if isinstance(self.width, np.ndarray):
            width = self.width[ch_i]
        else:
            width = self.width
        yy, xx = np.meshgrid(
            (np.linspace(0, 1, im.shape[0]) - self.center[0]) / width,
            (np.linspace(0, 1, im.shape[1]) - self.center[1]) / width,
        )
        self.regional_scale = np.exp(-(xx**2 + yy**2))
        im *= self.regional_scale


class RegIllumModel(BaseSynthModel):
    def __init__(self, reg_ill: RegIllumPrior):
        super().__init__()
        check.t(reg_ill, RegIllumPrior)
        self.reg_ill = reg_ill

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        super().render(im, fl_i, ch_i, cy_i, aln_offset)
        self.reg_scale = self.reg_ill.render()
        im *= self.reg_scale


class ChRegIllumModel(BaseSynthModel):
    # Multichannel version of RegIllumModel
    def __init__(self, reg_ills: List[RegIllumPrior]):
        super().__init__()
        check.list_t(reg_ills, RegIllumPrior)
        self.reg_ills = reg_ills

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        super().render(im, fl_i, ch_i, cy_i, aln_offset)
        reg_scale = self.reg_ills[ch_i].render()
        im *= reg_scale


class CameraModel(BaseSynthModel):
    def __init__(self, bg_mean=100, bg_std=10):
        super().__init__()
        self.bg_mean = bg_mean
        self.bg_std = bg_std

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        super().render(im, fl_i, ch_i, cy_i, aln_offset)
        bg = np.random.normal(loc=self.bg_mean, scale=self.bg_std, size=self.dim)
        imops.accum_inplace(im, bg, XY(0, 0), center=False)


class ChCameraModel(BaseSynthModel):
    # Multichannel version of CameraModel
    def __init__(self, ch_bg_means, ch_bg_stds):
        super().__init__()
        self.ch_bg_means = ch_bg_means
        self.ch_bg_stds = ch_bg_stds

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        super().render(im, fl_i, ch_i, cy_i, aln_offset)
        bg = np.random.normal(
            loc=self.ch_bg_means[ch_i], scale=self.ch_bg_stds[ch_i], size=self.dim
        )
        imops.accum_inplace(im, bg, XY(0, 0), center=False)


class HaloModel(BaseSynthModel):
    def __init__(self, std=20, scale=2):
        super().__init__()
        self.std = std
        self.scale = scale

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        super().render(im, fl_i, ch_i, cy_i, aln_offset)
        size = int(self.std * 2.5)
        size += 1 if size % 2 == 0 else 0
        bg_mean = np.median(im) - 1
        blur = cv2.GaussianBlur(im, (size, size), self.std) - bg_mean - 1
        imops.accum_inplace(im, self.scale * blur, XY(0, 0), center=False)


class BlobModel(BaseSynthModel):
    def __init__(self, size=15, amp=1000):
        super().__init__()
        self.size = (size & ~1) + 1
        self.amp = amp

    def render(self, im, fl_i, ch_i, cy_i, aln_offset):
        super().render(im, fl_i, ch_i, cy_i, aln_offset)

        blob = imops.generate_circle_mask(self.size, size=self.size * 3)
        imops.accum_inplace(
            im, self.amp * blob, XY(0.25 * im.shape[0], 0.25 * im.shape[0]), center=True
        )


def synth_to_ims_import_result(synth: Synth):
    chcy_ims = synth.render_chcy()

    with tmp_folder(remove=False) as folder:
        # A tmp folder is needed here because tests can run
        # multi-threaded and we need to avoid collisions
        # It can't be removed because the file will be opened
        # later outside of this scope so we assume that
        # tmp will be garbage collected outside of the
        # test system.

        ims_import_params = ImsImportParams()
        ims_import_result = ImsImportResult(
            folder=folder,
            params=ims_import_params,
            tsv_data=None,
            n_fields=synth.n_fields,
            n_channels=synth.n_channels,
            n_cycles=synth.n_cycles,
            dim=synth.dim[0],
            dtype=np.dtype(OUTPUT_NP_TYPE).name,
            src_dir="",
            flchcy_i_to_nd2path={},
            is_nd2=False,
        )

        for fl_i in range(synth.n_fields):
            field_chcy_arr = ims_import_result.allocate_field(
                fl_i,
                (synth.n_channels, synth.n_cycles, synth.dim[0], synth.dim[1]),
                OUTPUT_NP_TYPE,
            )
            field_chcy_ims = field_chcy_arr.arr()

            field_chcy_ims[:, :, :, :] = chcy_ims

            ims_import_result.save_field(fl_i, field_chcy_arr, None, None)

        ims_import_result.save()

    return ims_import_result
