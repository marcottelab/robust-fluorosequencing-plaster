"""
Pixel filtering on raw images.

Current implementation is a band-pass filter implemented with an FFT.
"""

import numpy as np

from plaster.tools.image import imops
from plaster.tools.schema import check


def filter_im(im, low_inflection, low_sharpness, high_inflection, high_sharpness):
    """
    Use a band-pass filter to remove background and "bloom" which is
    the light that scatters from foreground to background.

    Note: A low_inflection of -10 effectively removes the low-pass filter
    and a high_inflection of +10 effectively removes the high-pass filter

    Values of sharpness = 50.0 are usually fine.

    Returns:
         Filtered image
    """

    # These number were hand-tuned to Abbe (512x512) and might be wrong for other
    # sizes/instruments and will need to be derived and/or calibrated.
    im = im.astype(np.float64)
    check.array_t(im, ndim=2, is_square=True, dtype=np.float64)
    low_cut = imops.generate_center_weighted_tanh(
        im.shape[0], inflection=low_inflection, sharpness=low_sharpness
    )
    high_cut = 1 - imops.generate_center_weighted_tanh(
        im.shape[0], inflection=high_inflection, sharpness=high_sharpness
    )
    filtered_im = imops.fft_filter_with_mask(im, mask=low_cut * high_cut)

    # The filters do not necessarily create a zero-centered background so
    # not remove the median to pull the background to zero.
    filtered_im -= np.median(filtered_im)

    return filtered_im
