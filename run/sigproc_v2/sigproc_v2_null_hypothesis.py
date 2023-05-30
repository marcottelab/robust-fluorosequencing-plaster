from dataclasses import dataclass
from typing import List

import numpy as np
from munch import Munch

from plaster.run.priors import PriorsMLEFixtures
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.utils import utils
from plaster.tools.zap import zap


@dataclass
class SynthParams:
    im_mea: int
    n_channels: int
    bg_mean: int = 100
    bg_std: int = 50


def analyze_field_from_synthetic_peak_counts(
    peaks_per_channel: List[int],
    synth_params: SynthParams,
    sigproc_v2_params: SigprocV2Params,
):
    # HACK: These modules aren't organized well, so we're getting circular imports. Deferring import to execution time to fix :/
    from plaster.run.sigproc_v2 import sigproc_v2_worker, synth

    assert synth_params.n_channels == len(peaks_per_channel)
    n_peaks = sum(peaks_per_channel)
    mu = np.random.normal(10_000, 1000)
    sigma = np.random.normal(0.16, 0.05)
    halo_size = 20
    halo_inensity = 2
    n_cycles = 1
    n_channels = synth_params.n_channels
    hyper_peak_mea = 16
    hyper_peak_sigma = 1.8
    hyper_n_divs = 4

    x1 = np.arange(0, hyper_peak_mea // 2)
    x2 = np.arange(-hyper_peak_mea // 2, 0, 1)
    phase_j, phase_i = np.meshgrid(np.hstack((x1, x2)), np.hstack((x1, x2)))

    reg_psf = PriorsMLEFixtures.reg_psf_uniform(
        hyper_im_mea=synth_params.im_mea,
        hyper_peak_mea=hyper_peak_mea,
        hyper_n_divs=hyper_n_divs,
        hyper_peak_sigma=hyper_peak_sigma,
    )

    reg_illum = PriorsMLEFixtures.reg_illum_variable(
        hyper_im_mea=synth_params.im_mea, falloff=0.5
    )

    with synth.Synth(
        dim=(synth_params.im_mea, synth_params.im_mea),
        n_cycles=n_cycles,
        n_channels=n_channels,
    ) as s:
        s.channel_aln_offsets(np.random.rand(n_channels, 2))
        # s.zero_aln_offsets()

        dyts = [
            [[1 if n == i else 0] * n_cycles for n in range(n_channels)]
            for i in range(n_channels)
        ]

        probs = [float(n) / n_peaks for n in peaks_per_channel]

        peaks = (
            synth.PeaksModelPSF(n_peaks=n_peaks, reg_psf=reg_psf)
            # .bleach(0.3)
            .locs_randomize()
            # .locs_randomize_no_subpixel()
            # TODO: do we want a channel scale factor here? How to generate argument?
            # .channel_scale_factor(
            #     (
            #         1.0,
            #         0.6,
            #     )
            # )
            .multichannel_dyt_random_choice(dyts=dyts, probs=probs).gain_constant(mu)
        )

        synth.CameraModel(bg_mean=synth_params.bg_mean, bg_std=synth_params.bg_std)
        synth.HaloModel(halo_size, halo_inensity)
        synth.RegIllumModel(reg_illum)
        chcy_ims = s.render_chcy()

        # Offset and force everything positive and store as uint 16 like the camera does
        bias = np.min(chcy_ims)
        cam_chcy_ims = np.ascontiguousarray((chcy_ims - bias).astype(np.uint16))

    field_analysis: sigproc_v2_worker.AnalyzeFieldResult = (
        sigproc_v2_worker._analyze_field(cam_chcy_ims, sigproc_v2_params)
    )

    return NullHypothesisSynth(peaks_per_channel, cam_chcy_ims, field_analysis)


@dataclass
class NullHypothesisSynth:
    peaks_per_channel: List[int]
    chcy_im: np.ndarray
    field_analysis: "sigproc_v2_worker.AnalyzeFieldResult"  # Quoted to avoid circular import


@dataclass
class NullHypothesisPrecompute:
    source_peaks_space: np.ndarray
    synths: List[NullHypothesisSynth]


def precompute_null_hypothesis(
    actual_discovered_peaks,
    synth_params: SynthParams,
    sigproc_v2_params: SigprocV2Params,
    progress=None,
    n_steps=10,
    n_samples_per_step=3,
):
    source_peaks_space = utils.ispace(
        actual_discovered_peaks * 0.3, actual_discovered_peaks * 2.0, n_steps
    )

    samples_per_source_peak = n_samples_per_step

    peaks = np.repeat(source_peaks_space, samples_per_source_peak)

    with zap.Context(trap_exceptions=False, progress=progress):
        values = zap.work_orders(
            [
                Munch(
                    fn=analyze_field_from_synthetic_peak_counts,
                    args=[
                        [peak] * synth_params.n_channels,
                        synth_params,
                        sigproc_v2_params,
                    ],
                )
                for peak in peaks
            ]
        )

    return NullHypothesisPrecompute(source_peaks_space, values)
