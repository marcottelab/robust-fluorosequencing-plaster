import logging

import numpy as np

from plaster.run.sigproc_v2.psf import approximate_psf
from plaster.tools.image import imops
from plaster.tools.image.coord import WH, XY
from plaster.tools.schema import check
from plaster.tools.zlog.profile import prof

log = logging.getLogger(__name__)


def _descent_quantities(im_A, im_B, sh_x, sh_y):
    """
    Used to copmute the Jacobian during gradient descent of sub-pixel alignment.
    Based on John Haven Davis code.
    """
    mea = im_A.shape[0]
    assert im_A.shape[0] == im_B.shape[0]
    assert im_A.shape[1] == im_B.shape[1]
    assert im_A.shape[0] == im_A.shape[1]

    rng = np.arange(-(mea - 1) // 2, (mea + 1) // 2, 1)
    i, j = np.meshgrid(rng, rng)

    phasor = np.exp(-2.0 * complex(0.0, 1.0) * np.pi * (i * sh_x + j * sh_y) / mea)

    phasor_d_sh_x = np.multiply(phasor, (-2.0 * complex(0.0, 1.0) * np.pi * i / mea))
    phasor_d_sh_y = np.multiply(phasor, (-2.0 * complex(0.0, 1.0) * np.pi * j / mea))

    phasor_d_sh_x_d_sh_x = np.multiply(
        phasor_d_sh_x, (-2.0 * complex(0.0, 1.0) * np.pi * i / mea)
    )
    phasor_d_sh_x_d_sh_y = np.multiply(
        phasor_d_sh_x, (-2.0 * complex(0.0, 1.0) * np.pi * j / mea)
    )
    phasor_d_sh_y_d_sh_y = np.multiply(
        phasor_d_sh_y, (-2.0 * complex(0.0, 1.0) * np.pi * j / mea)
    )

    freq_dom_B = np.fft.fftshift(np.fft.fft2(im_B))
    freq_dom_shifted = np.multiply(freq_dom_B, phasor)
    freq_dom_shifted_d_sh_x = np.multiply(freq_dom_B, phasor_d_sh_x)
    freq_dom_shifted_d_sh_y = np.multiply(freq_dom_B, phasor_d_sh_y)
    freq_dom_shifted_d_sh_x_d_sh_x = np.multiply(freq_dom_B, phasor_d_sh_x_d_sh_x)
    freq_dom_shifted_d_sh_y_d_sh_y = np.multiply(freq_dom_B, phasor_d_sh_y_d_sh_y)
    freq_dom_shifted_d_sh_x_d_sh_y = np.multiply(freq_dom_B, phasor_d_sh_x_d_sh_y)

    shifted_B = np.real(np.fft.ifft2(np.fft.ifftshift(freq_dom_shifted)))
    shifted_B_d_sh_x = np.real(np.fft.ifft2(np.fft.ifftshift(freq_dom_shifted_d_sh_x)))
    shifted_B_d_sh_y = np.real(np.fft.ifft2(np.fft.ifftshift(freq_dom_shifted_d_sh_y)))
    shifted_B_d_sh_x_d_sh_x = np.real(
        np.fft.ifft2(np.fft.ifftshift(freq_dom_shifted_d_sh_x_d_sh_x))
    )
    shifted_B_d_sh_y_d_sh_y = np.real(
        np.fft.ifft2(np.fft.ifftshift(freq_dom_shifted_d_sh_y_d_sh_y))
    )
    shifted_B_d_sh_x_d_sh_y = np.real(
        np.fft.ifft2(np.fft.ifftshift(freq_dom_shifted_d_sh_x_d_sh_y))
    )

    corr = np.multiply(im_A, shifted_B)
    corr_d_sh_x = np.multiply(im_A, shifted_B_d_sh_x)
    corr_d_sh_y = np.multiply(im_A, shifted_B_d_sh_y)

    corr_d_sh_x_d_sh_x = np.multiply(im_A, shifted_B_d_sh_x_d_sh_x)
    corr_d_sh_y_d_sh_y = np.multiply(im_A, shifted_B_d_sh_y_d_sh_y)
    corr_d_sh_x_d_sh_y = np.multiply(im_A, shifted_B_d_sh_x_d_sh_y)

    obj = corr.sum()

    obj_d_sh_x = corr_d_sh_x.sum()
    obj_d_sh_y = corr_d_sh_y.sum()

    obj_d_sh_x_d_sh_x = corr_d_sh_x_d_sh_x.sum()
    obj_d_sh_y_d_sh_y = corr_d_sh_y_d_sh_y.sum()
    obj_d_sh_x_d_sh_y = corr_d_sh_x_d_sh_y.sum()

    gradient = np.array([obj_d_sh_x, obj_d_sh_y])
    hessian = np.array(
        [[obj_d_sh_x_d_sh_x, obj_d_sh_x_d_sh_y], [obj_d_sh_x_d_sh_y, obj_d_sh_y_d_sh_y]]
    )

    return obj, gradient, hessian


class AlignmentError(Exception):
    pass


def _subpixel_align_one_im(fix_im, sft_im, max_pix_shift=1.5):
    """
    Further refine the alignment to sub-pixel accuracy using
    gradient descent in FFT space.

    If this exceeds max_pix_shift it raises AlignmentError.
    The most common reason to exceed is because the image is
    so sparse that there isn't enough peaks to isolate the shift.
    """

    check.array_t(fix_im, ndim=2, is_square=True)
    check.array_t(sft_im, ndim=2, is_square=True)
    assert fix_im.shape == sft_im.shape
    sft_x = 0.0
    sft_y = 0.0
    precision = 1e-2**2
    for step in range(10):
        # _descent_quantities is much more expensive than is the following solver
        objective, gradient, hessian = _descent_quantities(fix_im, sft_im, sft_x, sft_y)
        delta = np.linalg.solve(hessian, gradient)
        sft_x += -delta[0]
        sft_y += -delta[1]

        if sft_x**2 + sft_y**2 > max_pix_shift**2:
            # Out of bounds
            raise AlignmentError

        if delta[0] ** 2 < precision and delta[1] ** 2 < precision:
            break
    else:
        # Failed to converge on sub pixel alignment.
        raise AlignmentError

    # JHD uses opposite shifting notation so the following is negated.
    return -sft_y, -sft_x


def _subsize_sub_pixel_align_cy_ims(pixel_aligned_cy_ims, subsize, n_samples):
    """
    The inner loop of _sub_pixel_align_cy_ims() that executes on a "subsize"
    region of the larger image.

    Is subsize is None then it uses the entire image.
    """
    n_max_failures = n_samples * 2
    sub_pixel_offsets = np.zeros((n_samples, pixel_aligned_cy_ims.shape[0], 2))
    pixel_aligned_cy0_im = pixel_aligned_cy_ims[0]

    im_mea = pixel_aligned_cy_ims.shape[-1]
    assert pixel_aligned_cy_ims.shape[-2] == im_mea

    def _subregion(im, pos):
        if subsize is None:
            return im
        else:
            return imops.crop(im, off=pos, dim=WH(subsize, subsize), center=False)

    sample_i = 0
    n_failures = 0
    while sample_i < n_samples and n_failures < n_max_failures:
        try:
            if subsize is None:
                pos = XY(0, 0)
            else:
                pos = XY(
                    np.random.randint(0, im_mea - subsize - 16),
                    np.random.randint(0, im_mea - subsize - 16),
                )

            subregion_pixel_aligned_cy0_im = _subregion(pixel_aligned_cy0_im, pos)

            for cy_i, pixel_aligned_cy_im in enumerate(pixel_aligned_cy_ims):
                if cy_i == 0:
                    continue

                # Use a small region to improve speed
                subregion_pixel_aligned_cy_im = _subregion(pixel_aligned_cy_im, pos)

                try:
                    _dy, _dx = _subpixel_align_one_im(
                        subregion_pixel_aligned_cy0_im,
                        subregion_pixel_aligned_cy_im,
                    )
                    sub_pixel_offsets[sample_i, cy_i, :] = (_dy, _dx)
                except Exception:
                    # This is a general exception handler because there
                    # are a number of ways that the _subpixel_align_one_im
                    # can fail including linear algebera, etc. All
                    # of which end up with a skip and a retry.
                    n_failures += 1
                    raise AlignmentError

            sample_i += 1

        except AlignmentError:
            # Try again with a new pos
            if n_failures >= n_max_failures:
                raise AlignmentError

    return np.mean(sub_pixel_offsets, axis=0)


def _sub_pixel_align_cy_ims(cy_ims, bounds=None):
    """
    Align image stack with sub-pixel precision.

    As an optimization, this will subdivide into a number of smaller
    random subregions and then average the offset.

    However, this optimization can fail when the image is very sparse.
    And that failure mode is trapped and then the stack is
    re-run without the subregion optimization.

    I tried to go down to 64 pixels sub-regions but could only get to 0.1 of a pixel
    which might be enough but decided to stay at 128 where I can get 0.05
    I found that i needed 8 samples for stability and each is taking about 1 sec
    on 100 cycles.
    """

    # ALIGN within one pixel
    with prof("pixel_align"):
        pixel_offsets, pixel_aligned_cy_ims = imops.align(
            cy_ims, return_shifted_ims=True, bounds=bounds
        )

    # IMPROVE with sub-pixel precision
    with prof("subpixel_align"):
        try:
            sub_pixel_offset = _subsize_sub_pixel_align_cy_ims(
                pixel_aligned_cy_ims, subsize=128, n_samples=8
            )
        except AlignmentError:
            # The subsize parameter is merely an optimization but in very sparse images
            # the aligner can fail to find enough peaks in an subregion to align so in
            # that case we try again but with the subsize optimization disabled.
            try:
                sub_pixel_offset = _subsize_sub_pixel_align_cy_ims(
                    pixel_aligned_cy_ims, subsize=None, n_samples=1
                )
            except AlignmentError:
                # This is a true failure. There was no alignment even using the entire
                # image so we jam in an impossible shift of the entire image.
                im_mea = pixel_aligned_cy_ims.shape[-1]
                far_away = np.full(pixel_offsets.shape, im_mea, dtype=float)
                far_away[0, :] = 0.0
                sub_pixel_offset = far_away

    return pixel_offsets + sub_pixel_offset


def align_ims(ims, bounds=None):
    """
    Align a stack of ims by generating simplified fiducials for each image.

    These ims might be from a single channel over many cycles or
    one cycle over many channels.

    If bounds, do not allow alignment > bounds

    Returns:
        aln_offsets: ndarray(n_cycles, 2); where 2 is (y, x)
    """

    approx_psf = approximate_psf()

    with prof("convolve"):
        fiducial_ims = np.zeros_like(ims)
        for i, im in enumerate(ims):
            im = im.astype(np.float64)
            if not np.all(np.isnan(im)):
                med = float(np.nanmedian(im))
            else:
                med = 0
            im = np.nan_to_num(im, nan=med)
            fid_im = imops.convolve(im, approx_psf - np.mean(approx_psf))
            fid_im -= np.median(fid_im)

            # The edges often end up with artifacts that will cause
            # the convolution grief so we simply zero them out as they
            # are unlikely to contribute much to the alignment
            imops.edge_fill(fid_im, approx_psf.shape[0] * 2)

            fiducial_ims[i, :, :] = fid_im

    return _sub_pixel_align_cy_ims(fiducial_ims, bounds=bounds)


def resample_aligned_ims(chcy_ims, chcy_offsets):
    """
    Given the alignment offsets for each channel and cycle,
    create a new image stack that has the dimensions of the
    intersection ROI (ie the overlapping region that contains
    pixels from all cycles.)

    Returns:
        A newly allocated ndarray(n_channels, n_cycles, dim, dim)
        where the region of interest determined by the pixels that
        all cycles have in common are positioned relative to the
        first cycle.
    """
    check.array_t(chcy_ims, ndim=4)
    n_channels, n_cycles, dim_h, dim_w = chcy_ims.shape
    assert dim_w == dim_h
    dim = dim_w

    # STACK all channel offsets to compute the ROI for the whole stack
    all_channel_aln_offsets = np.vstack(chcy_offsets)
    raw_dim = chcy_ims.shape[-2:]
    roi = imops.intersection_roi_from_aln_offsets(all_channel_aln_offsets, raw_dim)
    roi_h, roi_w = roi[0].stop - roi[0].start, roi[1].stop - roi[1].start
    bot, lft = roi[0].start, roi[1].start

    # RESAMPLE the original images with sub-pixel accuracy into a final buffer
    aligned_chcy_ims = np.zeros((n_channels, n_cycles, dim, dim))

    for ch_i in range(n_channels):
        for cy_i, offset in zip(range(n_cycles), chcy_offsets[ch_i]):
            # Sub-pixel shift the square raw images using phase shifting
            # (This must be done with the original square images)
            im = chcy_ims[ch_i, cy_i]
            shifted_im = imops.fft_sub_pixel_shift(im, -offset)

            # Now that it is shifted we pluck out the ROI into the destination
            aligned_chcy_ims[
                ch_i, cy_i, bot : bot + roi_h, lft : lft + roi_w
            ] = shifted_im[
                roi[0],
                roi[1],
            ]

    return aligned_chcy_ims
