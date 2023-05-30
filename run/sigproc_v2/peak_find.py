import warnings
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional

import numpy as np
import pandas as pd
from skimage import draw

from plaster.run.sigproc_v2.c_peak_find import peak_find as c_peak_find
from plaster.tools.image import imops
from plaster.tools.image.coord import HW, YX
from plaster.tools.schema import check
from plaster.tools.utils.stats import half_nanstd

log = getLogger(__name__)


def pixel_peak_find_one_im(im, approx_psf):
    """
    Peak find on a single image with 1 pixel accuracy.
    (compare to subpixel_peak_find_one_im)

    Arguments:
        im: the image to peak find
        approx_psf: An estimated PSF search kernel

    Returns:
        locs: ndarray (n_peaks_found, 2) where the 2 is in (y,x) order
    """
    from skimage.feature import peak_local_max  # Defer slow import

    std = half_nanstd(im.flatten())

    # This is assert is to catch bad tests that pass-in noise free
    # background that mean that the bg std can not be estimated
    # and therefore will cause many false peaks.
    assert std > 0.1, "The std is suspiciously small on a pixel_peak_find_one_im"

    # Tuning thresh:
    #  I initially had it set at 2 * std.
    #  Later, in synthetic 2 count data without a bandpass filter
    #  I found that I was losing too many 1 counts so I tried 1 * std
    #  but I found that when I put the band-pass back in that 2 * std
    #  seemed right again.  We probably should find an objective way
    #  to determine this.

    thresh = 2 * std

    cim = imops.convolve(np.nan_to_num(im, nan=float(np.nanmedian(im))), approx_psf)

    # CLEAN the edges
    # ZBS: Added because there were often edge effect from the convolution
    # that created false stray edge peaks.
    imops.edge_fill(cim, approx_psf.shape[0])

    # The background is well-described by the the histogram centered
    # around zero thanks to the fact that im and kern are expected
    # to be roughly zero-centered. Therefore we estimate the threshold
    # by using the samples less than zero cim[cim<0]
    if (cim < 0).sum() > 0:
        cim[cim < thresh] = 0
        return peak_local_max(cim, min_distance=2, threshold_abs=thresh)
    else:
        return np.zeros((0, 2))


def _pixel_to_subpixel_one_im(im, peak_dim, locs):
    """
    This is a subtle calculation.

    locs is given as an *integer* position (only has pixel accuracy).
    We then extract out a sub-image using an *integer* half width.
    Peak_dim is typically odd. Suppose it is (11, 11)
    That makes half_peak_mea_i be 11 // 2 = 5

    Suppose that a peak is at (17.5, 17.5).

    Suppose that peak was found a (integer) location (17, 17)
    which is within 1 pixel of its center as expected.

    We extract the sub-image at (17 - 5, 17 - 5) = (12:23, 12:23)

    The Center-of-mass calculation should return (5.5, 5.5) because that is
    relative to the sub-image which was extracted

    We wish to return (17.5, 17.5). So that's the lower left
    (17 - 5) of the peak plus the COM found.
    """
    check.array_t(locs, dtype=int)
    assert peak_dim[0] == peak_dim[1]
    half_peak_mea_i = peak_dim[0] // 2
    lower_left_locs = locs - half_peak_mea_i
    com_per_loc = np.zeros(locs.shape)
    for loc_i, loc in enumerate(lower_left_locs):
        peak_im = imops.crop(im, off=YX(loc), dim=peak_dim, center=False)
        com_per_loc[loc_i] = imops.com(peak_im**2)
    return lower_left_locs + com_per_loc


@dataclass
class PeakFindChCyImsResult:
    loc_df: pd.DataFrame
    locs_per_channel: Optional[List[np.ndarray]] = None

    @property
    def locs(self) -> np.ndarray:
        return self.loc_df[self.loc_df.ambiguous == 0][["loc_x", "loc_y"]].to_numpy()

    def ch_locs(self, ch_i, include_ambiguous=False):
        if include_ambiguous:
            df = self.loc_df
        else:
            df = self.loc_df[self.loc_df.ambiguous == 0]
        return df[[f"loc_ch_{ch_i}_x", f"loc_ch_{ch_i}_y"]].to_numpy()


def peak_find_chcy_ims(chcy_ims, approx_psf, cycle_i, subpixel=True):
    """
    Previous version of this code depended on the channels being
    balanced relative to one another. But this early-stage channel
    balancing turned out to be problematic.

    This new code instead peak finds on each channel independently
    and then reconciles the peak locations by unioning the
    lists of peaks and de-duping them using an approximate measure
    of distance as the de-dupe key.

    If subpixel is not True uses the faster pixel-only accuracy.

    There are two location "spaces" we are working in:
      1. The locations of the peaks found per channel using
         peak_local_max + the subpixel correction.
         The peak orders in this space are randomlized relative to
         each channel.
      2. The locations found through c_peak_find.peak_find_on_peak_label_ims
         which is called the "canonical peaks" which is where the
         peaks over all channels have been unifed.

    Returns:
        locs: ndarray (n_peaks, 2)  where the second dim is in (y, x) order
    """

    n_channels = chcy_ims.shape[0]
    im_dim = chcy_ims.shape[2:]
    locs_per_channel = []
    for ch_i in range(n_channels):
        im = chcy_ims[ch_i, cycle_i, :, :]
        try:
            locs = pixel_peak_find_one_im(im, approx_psf)
            if subpixel:
                locs = _pixel_to_subpixel_one_im(
                    im, HW(approx_psf.shape), locs.astype(int)
                )
        except Exception as e:
            # Failure during peak find, no peaks recorded for this frame.
            locs = np.zeros((0, 2))

        # RESERVE locs[0] with a nan.
        locs = np.vstack(([np.nan, np.nan], locs))

        locs_per_channel += [locs]

    # At this point locs_per_channel is a list of arrays and each of those
    # arrays may be a different length and the ordering of peaks in
    # each channel has nothing to do with the others.

    # CREATE a new set images call the "peak_labels" (one per channel).
    # These are zero-valued in the background and contain an peak_id
    # refering to the original peak index (in the original channel)
    # for each pixel. This allows us to create the final "canonical"
    # set of peaks by combining all the channel peak lists WITHOUT
    # brightness bias (ie SNR can vary wildly between channels.)

    radius = 2
    peak_label_ims_per_ch = np.zeros(
        (n_channels, im_dim[0], im_dim[1]), dtype=np.uint16
    )
    for ch_i, locs in enumerate(locs_per_channel):
        for loc_i, loc in enumerate(locs):
            # SLAT a circle into the peak_label_im

            # TODO: Likely faster to imops.generate_circle_mask once and then
            # splat that into the label_im instead of regenerating the disk
            # each loc.

            if loc_i > 0:
                circle = draw.disk(loc, radius)
                peak_label_ims_per_ch[ch_i][circle] = loc_i

            # Note that that peak [0] was reserved above
            # and therefore there will not be a peak for it.

    # RUN the peak reconciler on the peak_id_mask_ims_per_ch
    # to generate a "canonical" list of peaks.
    # Some channels will contribute more peaks than others and there are
    # edge cases called "ambiguous" where it could not be determined if
    # there was a single peak at that location.
    canonical_loc_i_to_ch_loc_i, ambiguous = c_peak_find.peak_find_on_peak_label_ims(
        peak_label_ims_per_ch
    )
    if canonical_loc_i_to_ch_loc_i.shape[0] == 0:
        # Handle empty case
        canonical_loc_i_to_ch_loc_i = np.zeros((0, n_channels))
    else:
        check.array_t(canonical_loc_i_to_ch_loc_i, dtype=int, ndim=2)

    n_canonical_locs = canonical_loc_i_to_ch_loc_i.shape[0]
    assert canonical_loc_i_to_ch_loc_i.shape[1] == n_channels
    assert n_canonical_locs == ambiguous.shape[0]

    # canonical_loc_i_to_ch_loc_i: a mapping from the canonical index back to
    # the original per-channel indicies. Ie there is one index PER CHANNEL
    # thus the shape is (n_canonical_locs, n_channels)

    # NOTE: A value of 0 in this array means that the peak was NOT
    # found in the given channel's original image and this agrees
    # with the fact that locs_per_channel reserved the zero peak
    # and put nan's into its location thus simplifying all of the following lookup

    # CREATE a full nan array over which we're going to nanmean
    canonical_locs_per_ch = np.full((n_channels, n_canonical_locs, 2), np.nan)

    for ch_i in range(n_channels):
        # LOOKUP each peak's location in the original per channel locs.
        # This is a numpy indirect lookup. Ie canonical_loc_i_to_ch_loc_i is
        # indexing to the locs_per_channel
        # Remember: locs_per_channel was initialized above with NAN in it's [0]
        # position fo that where canonical_loc_i_to_ch_loc_i is zero the
        # lookup will result in a nan

        if canonical_loc_i_to_ch_loc_i[:, ch_i].shape[0] == 0:
            canonical_locs_per_ch[ch_i] = np.zeros((0, 2))
        else:
            canonical_locs_per_ch[ch_i] = locs_per_channel[ch_i][
                canonical_loc_i_to_ch_loc_i[:, ch_i]
            ]

    # Each canonical peak may have come MORE THEN ONE
    # channel and there can be small noise-induced differences in each channel's
    # estimate of the center of the peak -- even for well behaved peaks.
    # Therfore we now use a nanmean to combine these slightly differing location estimates.
    # over all channels from which the peak was originally found.

    with warnings.catch_warnings():
        # AVOID nanmean runtime warning if you take a mean an all-nan group which correctly returns nan
        warnings.simplefilter("ignore", category=RuntimeWarning)

        canonical_locs = np.nanmean(canonical_locs_per_ch, axis=0)
        assert canonical_locs.shape == (n_canonical_locs, 2)

    # BUILD a DataFrame with the canoncial location called "loc_x" & "loc_y"
    # And for debugging purposes we also write in the original locations
    # per channel into their own columns.
    loc_df = pd.DataFrame()

    loc_df["loc_x"] = canonical_locs[:, 0]
    loc_df["loc_y"] = canonical_locs[:, 1]

    for ch_i in range(n_channels):
        loc_df[f"loc_ch_{ch_i}_x"] = canonical_locs_per_ch[ch_i, :, 0]
        loc_df[f"loc_ch_{ch_i}_y"] = canonical_locs_per_ch[ch_i, :, 1]

    loc_df[f"ambiguous"] = ambiguous

    return PeakFindChCyImsResult(
        loc_df=loc_df,
        locs_per_channel=locs_per_channel,
    )


def peak_find_chcy_ims_fast(chcy_ims, approx_psf, cycle_i, subpixel=True):
    """
    Unlike the above this assumes that channel balance is working well
    and that we can just mean over the channels
    """

    n_channels = chcy_ims.shape[0]

    # np.save(f"/erisyon/internal/_chcy_ims_{cycle_i}.npy", chcy_ims[:, cycle_i, 0, 0])
    im = np.mean(chcy_ims[:, cycle_i, :, :], axis=0)
    # np.save(f"/erisyon/internal/_mean_im_{cycle_i}.npy", im)
    try:
        locs = pixel_peak_find_one_im(im, approx_psf)
        if subpixel:
            locs = _pixel_to_subpixel_one_im(im, HW(approx_psf.shape), locs.astype(int))
    except Exception:
        # Failure during peak find, no peaks recorded for this frame.
        locs = np.zeros((0, 2))

    loc_df = pd.DataFrame()
    loc_df["loc_x"] = locs[:, 0]
    loc_df["loc_y"] = locs[:, 1]

    return PeakFindChCyImsResult(loc_df=loc_df)
