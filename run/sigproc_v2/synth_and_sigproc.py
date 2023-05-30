import numpy as np
from munch import Munch
from plumbum import local

from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_worker import ims_import
from plaster.run.priors import PriorsMLEFixtures
from plaster.run.run import RunResult
from plaster.run.sigproc_v2 import sigproc_v2_worker, synth
from plaster.run.sigproc_v2.sigproc_v2_params import SigprocV2Params
from plaster.tools.c_common.c_common_tools import DytType
from plaster.tools.utils import tmp
from plaster.tools.zap import zap


def independent_channels(bg_std, gain, reg_psf, n_peaks, n_channels):
    """
    Synthesize a field at a given density with each channel being independent (different spots)
    TODO: Convert to take priors instance and move into mainline code
    """
    im_mea = reg_psf.hyper_im_mea
    with synth.Synth(n_channels=n_channels, n_cycles=1, dim=(im_mea, im_mea)) as s:
        peaks = (
            synth.PeaksModelPSF(reg_psf)
            .multichannel_independent_random(n_peaks_per_ch=n_peaks)
            .gain_constant(gain)
        )
        synth.CameraModel(bg_mean=0.0, bg_std=bg_std)
        ims_import_result = synth.synth_to_ims_import_result(s)

    sigproc_v2_params = SigprocV2Params(
        divs=reg_psf.hyper_n_divs,
        peak_mea=reg_psf.hyper_peak_mea,
        mode="analyze",
        priors=PriorsMLEFixtures.fixture_no_errors(
            hyper_im_mea=im_mea,
            hyper_n_channels=n_channels,
            gain_mu=gain,
            reg_psf=reg_psf,
        ),
    )

    # The results can not be allowed to overwrite each other.
    # The save needs some kinds of flag or we have to change local folder
    # TODO: DRY with below
    with tmp.tmp_folder(remove=False) as folder:
        folder = folder / "plaster_output/sigproc_v2"
        folder.mkdir()
        sigproc_v2_result = sigproc_v2_worker.analyze(
            sigproc_v2_params, ims_import_result, folder=folder
        )

    return sigproc_v2_result


def do_combined_channels_synth_one_field(
    field_i,
    bg_std,
    gain,
    reg_psf,
    n_peaks,
    n_channels,
    n_cycles,
    dyts,
    probs,
):
    im_mea = reg_psf.hyper_im_mea

    with synth.Synth(
        n_channels=n_channels, n_cycles=n_cycles, dim=(im_mea, im_mea)
    ) as s:
        peaks = (
            synth.PeaksModelPSF(reg_psf, n_peaks=n_peaks)
            .multichannel_dyt_random_choice(dyts, probs)
            .gain_constant(gain)
            .locs_randomize()
        )
        synth.CameraModel(bg_mean=0.0, bg_std=bg_std)
        chcy_ims = s.render_chcy()
        for ch_i, ch_ims in enumerate(chcy_ims):
            for cy_i, im in enumerate(ch_ims):
                np.save(f"test_fl{field_i:04d}_ch{ch_i:01d}_cy{cy_i:03d}.npy", im)


def combined_channels(bg_std, gain, reg_psf, n_peaks, n_channels, n_cycles, n_fields=1):
    """
    Synthesize a field at a given density with related channels
    where there are an equal number of peaks in the combination
        * singles (only in one channel) -- n_channels
        * all (in all channels) -- 1
        # * pairs (in each pair of channels) -- (n_channels-1)*n_channels / 2
    Thus n_categories = n_channels + 1 # + (n_channels-1)*n_channels / 2
    """
    n_categories = n_channels + 1
    dyts = np.zeros((n_categories, n_channels, n_cycles), dtype=DytType)

    # Singletons
    for ch_i in range(n_channels):
        dyts[ch_i, ch_i, :] = 1

    # All channels
    dyts[n_channels, :, :] = 1

    probs = np.ones((n_categories,)) / n_categories

    im_mea = reg_psf.hyper_im_mea

    sigproc_v2_params = SigprocV2Params(
        divs=reg_psf.hyper_n_divs,
        peak_mea=reg_psf.hyper_peak_mea,
        mode="analyze",
        priors=PriorsMLEFixtures.fixture_no_errors(
            hyper_im_mea=im_mea,
            hyper_n_channels=n_channels,
            gain_mu=gain,
            reg_psf=reg_psf,
        ),
    )

    with tmp.tmp_folder(remove=False) as job_folder:
        job_folder = local.path(job_folder)

        ims_import_folder = job_folder / "plaster_output/ims_import"
        ims_import_folder.mkdir()

        sigproc_v2_folder = job_folder / "plaster_output/sigproc_v2"
        sigproc_v2_folder.mkdir()

        with local.cwd(ims_import_folder):
            zap.work_orders(
                [
                    dict(
                        fn=do_combined_channels_synth_one_field,
                        field_i=field_i,
                        bg_std=bg_std,
                        gain=gain,
                        reg_psf=reg_psf,
                        n_peaks=n_peaks,
                        n_channels=n_channels,
                        n_cycles=n_cycles,
                        dyts=dyts,
                        probs=probs,
                    )
                    for field_i in range(n_fields)
                ]
            )

            ims_import_result = ims_import(
                job_folder / "plaster_output/ims_import",
                ImsImportParams(),
            )

            ims_import_result.save()

        with local.cwd(sigproc_v2_folder):
            sigproc_v2_result = sigproc_v2_worker.analyze(
                sigproc_v2_params, ims_import_result, folder=sigproc_v2_folder
            )
            sigproc_v2_result.save()

    return sigproc_v2_result


def run_from_sigproc_v2_result(sigproc_v2_result):
    """
    Mock a RunResult with the only task being the sythen sigproc_v2_result made above
    """
    return RunResult(
        run_folder=sigproc_v2_result._folder.parent.parent,
        include_manifest=False,
        mock_config=Munch(sigproc_v2=sigproc_v2_result),
    )
