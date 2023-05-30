import itertools

import cv2
import numpy as np
import structlog
from icecream import ic
from numpy import linalg as LA
from numpy.linalg import LinAlgError
from plumbum import local
from scipy import interpolate, optimize, stats

from plaster.tools.image.coord import HW, ROI, WH, XY, YX, clip2d
from plaster.tools.schema import check
from plaster.tools.utils import utils

logger = structlog.get_logger()


def generate_gauss_kernel(
    std=None, offset_x=0.0, offset_y=0.0, mea=None, sf=3, cov=None
):
    """
    DEPRECATED. See gauss2_rho_form()

    Generate a symmetric gaussian kernel with unit Area Under Curve.
    Arguments:
        std: The standard deviation. If None it uses the cov instead.
        offset_x, offset_y: A jitter offset to place the peak not exactly in the center (-0.5 < offset < +0.5)
        mea: The (mea)sure (the length of the side of the square) to embed the kernel in.
             The mea must be odd so that there will be a true center pixel
             If the mea is not specified it makes the mea will be the next odd value where 3 standard deviations
             will fit on either side (ie. 6 sds wide)
        sf:  The scale factor used to super-sample the function. If the std is very small then
             if you try to have a fractional offset you will find that your signal disappears
             because it might not fall into any sample. By using a higher scale factor you
             will pick up that mass that would have fallen between pixels
        cov: Only valid if std is None in which case this arg is the full covariance matrix
    Returns:
        A 2D Gaussian with an Area-Under-Curve forced to 1.0
    """
    # assert -0.5 <= offset_x <= 0.5 and -0.5 <= offset_y <= 0.5

    if mea is None:
        mea = int(std * 8)
        mea += ~(mea & 1)  # Force it to be odd

    assert mea & 1 == 1  # Must be odd

    if std is None:
        assert cov is not None
    else:
        assert cov is None
        cov = std**2 * np.eye(2)

    # Scale if up by the scale factor
    cov *= np.sqrt(sf)
    two_pi = np.pi * 2.0
    offset_x *= sf
    offset_y *= sf
    mea *= sf
    space = np.linspace(-(mea - 1) / 2.0, (mea - 1) / 2.0, mea)
    x, y = np.meshgrid(space, space)

    inv_cov = np.linalg.inv(cov)
    xx = (x - offset_x).flatten()
    yy = (y - offset_y).flatten()
    xxyy = np.stack((xx, yy))
    r = inv_cov @ xxyy
    r = r[0] ** 2 + r[1] ** 2
    exp_term = np.exp(-0.5 * r)
    exp_term = exp_term.reshape((mea, mea))

    hi_res_im = exp_term

    # Down sample it using cv2 good resampling function
    assert hi_res_im.shape[0] == hi_res_im.shape[1]
    low_res_dim = int(hi_res_im.shape[0] / sf)
    low_res_im = cv2.resize(
        hi_res_im, dsize=(low_res_dim, low_res_dim), interpolation=cv2.INTER_AREA
    )
    low_res_im *= hi_res_im.shape[0] ** 2 / low_res_im.shape[0] ** 2

    # assert np.abs(1.0 - low_res_im.sum()) < 0.01
    return low_res_im / np.sum(low_res_im)


def generate_circle_mask(rad, size=None):
    """Generate boolean mask: True inside the circle, False outside. If size is None it is the diameter"""
    diam = (rad + 1) * 2 - 1
    if size is None:
        size = diam
    assert size & 1 == 1  # dimensions must be odd
    assert size >= diam
    if rad == 0:
        return np.zeros((size, size), dtype=bool)
    offset = (diam - size) // 2
    x, y = np.mgrid[:size, :size]
    circle_dist = (x - rad + offset) ** 2 + (y - rad + offset) ** 2
    return circle_dist < (rad + 0.5) ** 2


def generate_donut_mask(outer_rad, inner_rad, size=None):
    """Generate boolean mask: True inside the circle, False outside. If size is None it is the diameter"""
    assert inner_rad < outer_rad
    diam = (outer_rad + 1) * 2 - 1
    if size is None:
        size = diam
    assert size & 1 == 1  # dimensions must be odd
    assert size >= diam
    offset = (diam - size) // 2
    x, y = np.mgrid[:size, :size]
    circle_dist = (x - outer_rad + offset) ** 2 + (y - outer_rad + offset) ** 2
    return (circle_dist <= (outer_rad + 0.5) ** 2) & (
        circle_dist > (inner_rad + 0.5) ** 2
    )


def generate_square_mask(rad, filled=False):
    size = rad * 2 + 1
    if filled:
        im = np.ones((size, size))
    else:
        im = np.zeros((size, size))
        im[0] = 1
        im[-1] = 1
        im[:, 0] = 1
        im[:, -1] = 1
    return im


def generate_center_weighted_tanh(mea, inflection, sharpness):
    """
    Generates a weighting kernel that has most of the weight at the center
    using tanh to create a circular shelf with a gradient edge.

    Arguments:
        mea: The width / height of the image to be generated
        inflection: float(0-1). Where along the radius the transition from low to high
        sharpness: float. >> 1 means a sharp transition from low to high
    """
    space = np.linspace(-1, 1, mea)
    x, y = np.meshgrid(space, space)
    r = np.sqrt(x**2 + y**2)
    return 0.5 * np.tanh((r - inflection) * sharpness) + 0.5


def extract_with_mask(im, mask, loc, center=False):
    """
    Extracts the values from im at loc using the mask.
    Returns zero where the mask was False
    """
    try:
        a = im[ROI(loc, mask.shape, center=center)]
        return np.where(mask, a, 0.0)
    except ValueError:
        ic(
            loc,
            im.shape,
            mask.shape,
            center,
            a.shape,
            ROI(loc, mask.shape, center=center),
        )


def shift(src, loc=XY(0, 0)):
    """Offset"""
    loc = YX(loc)
    extra_dims = src.shape[0:-2]
    n_extra_dims = len(extra_dims)
    src_dim = HW(src.shape[-2:])
    tar_dim = src_dim
    tar_roi, src_roi = clip2d(loc.x, tar_dim.w, src_dim.w, loc.y, tar_dim.h, src_dim.h)
    tar = np.zeros((*extra_dims, *tar_dim))
    if tar_roi is not None and src_roi is not None:
        if n_extra_dims > 0:
            tar[:, tar_roi[0], tar_roi[1]] += src[:, src_roi[0], src_roi[1]]
        else:
            tar[tar_roi[0], tar_roi[1]] += src[src_roi[0], src_roi[1]]
    return tar


def accum_inplace(tar, src, loc=XY(0, 0), center=False):
    """
    Accumulate the src image into the (tar)get
    at loc with optional source centering
    """
    loc = YX(loc)
    tar_dim = HW(tar.shape)
    src_dim = HW(src.shape)
    if center:
        loc -= src_dim // 2
    tar_roi, src_roi = clip2d(loc.x, tar_dim.w, src_dim.w, loc.y, tar_dim.h, src_dim.h)
    if tar_roi is not None and src_roi is not None:
        tar[tar_roi] += src[src_roi]


def set_with_mask_in_place(tar, mask, value, loc=XY(0, 0), center=False):
    loc = YX(loc)
    tar_dim = HW(tar.shape)
    msk_dim = HW(mask.shape)
    if center:
        loc -= msk_dim // 2
    tar_roi, msk_roi = clip2d(loc.x, tar_dim.w, msk_dim.w, loc.y, tar_dim.h, msk_dim.h)
    if tar_roi is not None and msk_roi is not None:
        subset = tar[tar_roi]
        subset[mask[msk_roi]] = value
        tar[tar_roi] = subset


def convolve(src, kernel):
    """
    Wrapper for opencv 2d convolution
    https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#filter2d
    """
    dst = np.zeros(src.shape)
    assert dst.dtype == np.float64
    assert src.dtype == np.float64
    cv2.filter2D(
        dst=dst,
        src=src,
        ddepth=-1,  # Use the same bit-depth as the src
        kernel=kernel,
        borderType=cv2.BORDER_REPLICATE,
    )
    return dst


def crop(src, off=XY(0, 0), dim=WH(-1, -1), center=False):
    if dim.h == -1 and dim.w == -1:
        dim = HW(src.shape)
    return src[ROI(off, dim, center=center)]


def align(im_stack, return_shifted_ims=False, bounds=None):
    """
    Align the image stack (1 pixel accuracy) relative to the first frame in the stack
    Arguments:
        im_stack (3 dimensions)
        return_shifted_ims:
            If True, also return the shifted images truncated to the common
            region of interest
        bounds: If not None limit the search space

    Returns:
        list of YX tuples
        shifted_ims (optional)
    """
    check.array_t(im_stack, ndim=3, dtype=np.float64)
    n_cycles, mea_h, mea_w = im_stack.shape
    assert mea_h == mea_w

    offsets = [YX(0, 0)]
    primary = im_stack[0]
    for im in im_stack[1:]:

        # TODO: This could be optimized by using fft instead of
        #       cv2.filter2D() which would avoid the fft of the
        #       unchanging primary.
        conv = convolve(src=primary, kernel=im)

        # conv is now zero-centered; that is, the peak is
        # an offset relative to the center of the image.

        if bounds is not None:
            edge_fill(conv, (mea_w - 2 * bounds) // 2, val=0)

        peak = YX(np.unravel_index(conv.argmax(), conv.shape))
        center = HW(conv.shape) // 2
        offsets += [center - peak]

    if return_shifted_ims:
        raw_dim = im_stack.shape[-2:]
        roi = intersection_roi_from_aln_offsets(offsets, raw_dim)
        roi_dim = (roi[0].stop - roi[0].start, roi[1].stop - roi[1].start)

        pixel_aligned_cy_ims = np.zeros((n_cycles, mea_h, mea_w))
        for cy_i, offset in zip(range(n_cycles), offsets):
            shifted_im = shift(im_stack[cy_i], offset * -1)
            pixel_aligned_cy_ims[cy_i, 0 : roi_dim[0], 0 : roi_dim[1]] = shifted_im[
                roi[0], roi[1]
            ]
        return np.array(offsets), pixel_aligned_cy_ims

    else:
        return np.array(offsets)


def intersection_roi_from_aln_offsets(aln_offsets, raw_dim):
    """
    Compute the ROI that contains pixels from all frames
    given the aln_offsets (returned from align)
    and the dim of the original images.
    """
    aln_offsets = np.array(aln_offsets)
    check.affirm(
        np.all(aln_offsets[0] == (0, 0)), "intersection roi must start with (0,0)"
    )

    # intersection_roi is the ROI in the coordinate space of
    # the [0] frame that has pixels from every cycle.
    clip_dim = (
        np.min(aln_offsets[:, 0] + raw_dim[0]) - np.max(aln_offsets[:, 0]),
        np.min(aln_offsets[:, 1] + raw_dim[1]) - np.max(aln_offsets[:, 1]),
    )

    b = max(0, -np.min(aln_offsets[:, 0]))
    t = min(raw_dim[0], b + clip_dim[0])
    l = max(0, -np.min(aln_offsets[:, 1]))
    r = min(raw_dim[1], l + clip_dim[1])
    return ROI(loc=YX(b, l), dim=HW(t - b, r - l))


def thresh_filter_inplace(im, thresh=1.0):
    im[im < thresh] = 0.0


def thresh_filter(im, thresh=1.0):
    cp = np.copy(im)
    cp[cp < thresh] = 0.0
    return cp


def composite(ims, offsets, start_accum=None, limit_accum=None):
    """Build up a composite image from the stack with offsets"""

    # FIND the largest offset and add a border around the image of that size
    border_size = np.abs(offsets).max()
    border = HW(border_size, border_size)
    comp_dim = HW(ims[0].shape) + border * 2
    comp = np.zeros(comp_dim)
    comp_count = 0
    for i, (im, offset) in enumerate(zip(ims, offsets)):
        if start_accum is not None and i < start_accum:
            continue
        if limit_accum is not None and comp_count >= limit_accum:
            break
        accum_inplace(comp, src=im, loc=border - YX(offset))
        comp_count += 1
    return comp, border_size


def dot(im_a, im_b):
    """Compute the dot product between two images of the same size"""
    return im_a.ravel().dot(im_b.ravel())


def stack(imstack):
    """Given a list or a singleton image, make a 3-d stack. If already a 3d stack then no-op"""
    if isinstance(imstack, np.ndarray) and imstack.ndim == 2:
        imstack = [imstack]

    if isinstance(imstack, list):
        assert all([im.ndim == 2 for im in imstack])
        imstack = np.stack(imstack)

    assert imstack.ndim == 3

    return imstack


def extract_trace(imstack, loc, dim, center=True):
    """Extract a trace of dim at loc from the stack"""
    imstack = stack(imstack)
    dim = HW(dim)
    loc = YX(loc)
    roi = ROI(loc, dim, center=center)
    return imstack[:, roi[0], roi[1]]


def _circ_gaussian(height, center_y, center_x, var):
    return lambda y, x: height * np.exp(
        -0.5 * (((x - center_x) ** 2 + (y - center_y) ** 2) / var)
    )


def _circ_moments(data):
    """
    Calculate the moments of a 2D distribution.
    Returns (height, x, y, width)
    """
    assert data.ndim == 2
    assert data.shape[0] == data.shape[1]

    total = data.sum()
    y, x = np.indices(data.shape)
    y = (y * data).sum() / total
    x = (x * data).sum() / total
    y = min(data.shape[0] - 1, max(0, y))
    x = min(data.shape[1] - 1, max(0, x))
    col = data[:, int(x)]
    height = data.max()
    width_x = np.sqrt(np.abs((np.arange(col.size) - x) ** 2 * col).sum() / col.sum())
    row = data[int(y), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - y) ** 2 * row).sum() / row.sum())

    return height, x, y, (width_x + width_y) / 2.0


def fit_circ_gaussian(data):
    """
    Fit a 2D Gaussian to the data. Data is a 2D array of values.
    Uses the moments to estimate the starting parameters.
    # Adapted from: http://scipy-cookbook.readthedocs.io/items/FittingData.html
    """
    with np.errstate(all="raise"):
        assert data.ndim == 2
        assert data.shape[0] == data.shape[1]

        def errorfunction(params):
            return np.ravel(_circ_gaussian(*params)(*np.indices(data.shape)) - data)

        p, cov_x, infodict, mesg, ier = optimize.leastsq(
            errorfunction, _circ_moments(data), full_output=True, maxfev=500
        )
        if ier > 4:
            raise Exception("Fit steps exceeded")

        p[1] -= int(data.shape[0] / 2)
        p[2] -= int(data.shape[1] / 2)
        return p


def _imstack_write(dir, name, imstack):
    """Mockable"""
    np.save(local.path(dir) / name, imstack)


def dump_set_root_path(_root_path):
    dump_set_root_path.root_path = _root_path


def dump(name, imstack):
    imstack = stack(imstack)

    if not hasattr(dump_set_root_path, "root_path"):
        dump_set_root_path.root_path = "."

    _imstack_write(dump_set_root_path.root_path, f"{name}.npy", imstack)


def eigen_moments(lump):
    """
    Given a lump of masses compute the moment of inertia on it and use
    eigen analysis to extract the eigen values. Lumps which are concentrated
    in the center will have a small sum of eigenvalues.
    """
    ys, xs = np.indices(lump.shape)
    pos = np.array((ys, xs)).T.reshape(-1, 2)
    mas = lump.T.reshape(-1)
    com_y = (pos[:, 0] * mas).sum() / lump.sum()
    com_x = (pos[:, 1] * mas).sum() / lump.sum()
    com = np.array([com_y, com_x])
    centered = pos - com
    tor_y = centered[:, 0] ** 2 * mas
    tor_x = centered[:, 1] ** 2 * mas
    tor = np.array([tor_y, tor_x]).T
    cov = tor.T @ tor
    eig_vals, _ = LA.eig(cov)
    return eig_vals


def tanh_mask(dim, freq_of_half, smoothness):
    """
    Makes a centered radial-mask with 0 at the center a 1 at the edges.

    Arguments:
        dim: The width and height of the image. (power of 2)
    freq_of_half:
        The point at which the mask is 1/2.
        For example, if freq_of_half is 5 it means that any wave
        that that cycled 4 times in the width of the image
        would be cut whereas a wave that cycled 6 times would survive.
    smoothness:
        A large value make a smooth transition and a small one
        quickly cuts off. A smoother value will result in less ringing
    """
    idim = int(dim)
    assert idim == dim and (idim & (idim - 1)) == 0  # Must be a power of 2
    half_dim = dim / 2.0
    x = np.linspace(-half_dim, +half_dim, dim)
    y = np.linspace(-half_dim, +half_dim, dim)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)
    return np.tanh((d - freq_of_half) / smoothness) / 2 + 0.5


def fft_filter_with_mask(im, mask):
    """
    filter im with the mask from tanh_mask() or similar
    """
    freq_dom = np.fft.fftshift(np.fft.fft2(im)) * mask
    return np.real(np.fft.ifft2(np.fft.ifftshift(freq_dom)))


def fft_measure_power_ratio(im, mask):
    """
    Return the fraction of power in the mask frequencies
    """
    all_freq = np.fft.fftshift(np.fft.fft2(im))
    masked_freq = all_freq * mask
    return np.sum(np.abs(masked_freq)) / np.sum(np.abs(all_freq))


def fill(tar, loc, dim, val=0, center=False):
    """Fill target with value in ROI"""
    loc = YX(loc)
    dim = HW(dim)
    tar_dim = HW(tar.shape)
    if center:
        loc -= dim // 2
    tar_roi, _ = clip2d(loc.x, tar_dim.w, dim.w, loc.y, tar_dim.h, dim.h)
    if tar_roi is not None:
        tar[tar_roi] = val


def edge_fill(tar, thickness, val=0):
    """Fill rect edge target with value in ROI"""
    assert thickness < tar.shape[0] and thickness < tar.shape[1]

    # Bottom
    tar[0:thickness, 0 : tar.shape[1]] = val

    # Top
    tar[tar.shape[0] - thickness : tar.shape[1], 0 : tar.shape[1]] = val

    # Left
    tar[0 : tar.shape[0], 0:thickness] = val

    # Right
    tar[0 : tar.shape[0], tar.shape[1] - thickness : tar.shape[1]] = val

    return tar


def rolling_window(im, window_dim, n_samples, return_coords=False):
    """
    Sample im in windows of shape window_dim n_sample number of times;
    this may require overlapping the sample windows.

    Arguments:
        im: ndarray of ndim==2
        window_dim: 2-tuple, the size of the window (smaller than im.shape)
        n_samples: 2-tuple, the number of sample along each dimension
    """
    check.affirm(im.ndim >= 2)
    check.list_or_tuple_t(window_dim, int, expected_len=2)
    check.list_or_tuple_t(n_samples, int, expected_len=2)
    extra_dims = im.shape[0:-2]
    n_extra_dims = len(extra_dims)

    start = [None, None]
    stop = [None, None]
    slices = [None, None]
    for d in range(2):
        if window_dim[d] * n_samples[d] < im.shape[n_extra_dims + d]:
            raise ValueError(
                f"Dimension of {im.shape[n_extra_dims+d]} can not be spanned by {n_samples[d]} spans of length {window_dim[d]}."
            )

        start[d] = np.linspace(
            0, im.shape[n_extra_dims + d] - window_dim[d], n_samples[d], dtype=int
        )
        stop[d] = start[d] + window_dim[d]

        slices[d] = [slice(i, i + window_dim[d]) for i in start[d]]

    ims = np.zeros(
        (*extra_dims, n_samples[0], n_samples[1], window_dim[0], window_dim[1]),
        dtype=im.dtype,
    )
    coords = np.zeros((n_samples[0], n_samples[1], 2))

    for y, yy in enumerate(slices[0]):
        for x, xx in enumerate(slices[1]):
            coords[y, x] = (yy.start, xx.start)
            if n_extra_dims > 0:
                ims[:, y, x, :, :] = im[:, yy, xx]
            else:
                ims[y, x, :, :] = im[yy, xx]

    if return_coords:
        return ims, coords
    else:
        return ims


def mode(im, nan_policy="omit"):
    _mode, _ = stats.mode(im.flatten(), nan_policy=nan_policy)
    return _mode


def interp(src, dst_shape):
    """
    Interpolate a src image to different dst_shape
    """
    assert src.ndim == 2 and len(dst_shape) == 2
    y_src_space = np.arange(0, src.shape[0])
    x_src_space = np.arange(0, src.shape[1])
    y_dst_space = np.linspace(0, src.shape[0] - 1, dst_shape[0])
    x_dst_space = np.linspace(0, src.shape[1] - 1, dst_shape[1])
    if min(src.shape[0], src.shape[1]) == 1:
        return src[0, 0] * np.ones(dst_shape)
    elif min(src.shape[0], src.shape[1]) < 4:
        f = interpolate.interp2d(x_src_space, y_src_space, src, kind="linear")
    else:
        f = interpolate.interp2d(x_src_space, y_src_space, src, kind="cubic")
    return f(x_dst_space, y_dst_space)


def nan_fg(im, background_threshold_im, radius=3):
    """
    Returns the background with the foreground converted to NaN
    The foreground is considered anything brighter than background_threshold_im with a radius dilation.
    background_threshold_im is typically something like the 1.5*median_im and
    """
    assert im.ndim == 2 and im.shape == background_threshold_im.shape
    fg_mask = np.where(im > background_threshold_im, 1.0, 0.0)
    kernel = generate_circle_mask(radius).astype(np.uint8)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    return np.where(fg_mask > 0, np.nan, im)


def nan_bg(im, background_threshold_im, radius=3):
    """
    Returns the foreground with the background converted to NaN
    The foreground is considered anything brighter than background_threshold_im with a radius dilation.
    background_threshold_im is typically something like the 1.5*median_im and
    """
    assert im.ndim == 2 and im.shape == background_threshold_im.shape
    fg_mask = np.where(im > background_threshold_im, 1.0, 0.0)
    kernel = generate_circle_mask(radius).astype(np.uint8)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    return np.where(fg_mask < 1, np.nan, im)


def locs_to_region(locs, n_divs, im_dim):
    """
    Convert a matrix of locs in (y, x) columns into
    the regional coords of im_dim divided in n_divs.
    """
    check.array_t(locs, shape=(None, 2))
    reg_locs = np.floor(n_divs * locs / im_dim).astype(int)
    assert np.all((0 <= reg_locs) & (reg_locs < n_divs))
    return reg_locs


def region_enumerate(im, n_divs):
    assert im.ndim >= 2
    y_win = (im.shape[-2] // n_divs) + (0 if im.shape[-2] % n_divs == 0 else 1)
    x_win = (im.shape[-1] // n_divs) + (0 if im.shape[-1] % n_divs == 0 else 1)
    wins, coords = rolling_window(
        im, (y_win, x_win), (n_divs, n_divs), return_coords=True
    )
    for y, x in itertools.product(range(n_divs), range(n_divs)):
        if wins.ndim == 4:
            yield wins[y, x], y, x, coords[y, x]
        else:
            yield wins[:, y, x], y, x, coords[y, x]


def region_map(im, func, n_divs=4, include_coords=False, **kwargs):
    """
    Apply the function over window regions of im.
    Regions are divisions of the LAST-TWO dimensions of im.
    """
    assert im.ndim >= 2

    results = []
    for win_im, _, _, coord in region_enumerate(im, n_divs):
        if include_coords:
            kwargs["coords"] = coord
        results += [func(win_im, **kwargs)]

    assert len(results) == n_divs * n_divs
    if isinstance(results[0], tuple):
        # The func returned a tuple of return values
        # These have to be re-assembled into arrays with a rule
        # that all arrays of each component of the tuple
        # have to return the same size.
        n_ret_fields = len(results[0])
        result_fields = []
        for ret_field_i in range(n_ret_fields):
            # Suppose func returns a tuple( array(11, 11), array(n, 8) )
            # For the first argument you want to return a (divs, divs, 11, 11)
            # But for the second arguments you might want (divs, divs

            field = utils.listi(results, ret_field_i)

            # field is expected to be a list of arrays all of same shape
            if isinstance(field[0], np.ndarray):
                field_shape = field[0].shape
                assert all([row.shape == field_shape for row in field])
            elif np.isscalar(field[0]):
                # Convert to an array
                field = np.array(field)
            else:
                raise TypeError(
                    f"Unexpected return type from {func.__name__} in region_map"
                )

            field_array = field.reshape((n_divs, n_divs, *field.shape[1:]))
            result_fields += [field_array]
        results = tuple(result_fields)
    else:
        results = np.array(results)
        results = results.reshape((n_divs, n_divs, *results.shape[1:]))

    return results


def stack_map(ims, func):
    """
    Apply the function over ims
    """
    assert ims.ndim == 3
    return np.array([func(im) for im in ims])


def ims_flat_stack(ims):
    """Flatten all but the last two dimensions of a multi-dimension array of"""
    return ims.reshape((np.product(ims.shape[0:-2]), *ims.shape[-2:]))


def argmin(im):
    return np.unravel_index(im.argmin(), im.shape)


def argmax(im):
    return np.unravel_index(im.argmax(), im.shape)


def low_frequency_power(im, dim_half=3):
    """
    Measure the low_frequency_power (excluding DC) of an image
    by spatial low-pass filter.

    dim_half is the half the width of the region
    """
    a = np.copy(im)
    a -= np.mean(a)
    power = np.abs(np.fft.fftshift(np.fft.fft2(a)))
    power[power == 0] = 1

    cen = YX(power.shape) / 2

    dim = HW(dim_half * 2 + 1, dim_half * 2 + 1)

    # PLUCK out the center (which is the low frequencies)
    roi = ROI(cen, dim, center=True)
    im = power[roi]
    eigen = eigen_moments(im)
    score = power.sum() / np.sqrt(eigen.sum())
    return score


def com(im):
    """
    Compute the center of mass of im.
    Expects that im is leveled (ie zero-centered). Ie, a pure noise image should have zero mean.
    Sometimes this is improved if you square the im first com(im**2)
    Returns:
        y, x in array form.
    """
    im = np.nan_to_num(im)
    mass = np.sum(im)
    ry = (
        np.arange(im.shape[0]) + 0.5
    )  # 0.5 because we want the mass of a pixel at the center of the pixel
    rx = np.arange(im.shape[1]) + 0.5
    y = np.sum(ry * np.sum(im, axis=1))
    x = np.sum(rx * np.sum(im, axis=0))
    return utils.np_safe_divide(np.array([y, x]), mass)


def scale_im(im, scale):
    """Scale an image up or down"""
    check.array_t(im, ndim=2, dtype=float)
    rows, cols = im.shape
    M = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]])
    return cv2.warpAffine(
        im, M, dsize=(int(scale * cols), int(scale * rows)), flags=cv2.INTER_CUBIC
    )


def sub_pixel_shift(im, offset):
    """
    Shift with offset in y, x array form.
    A positive x will shift right. A positive y will shift up.
    """
    check.array_t(im, ndim=2, dtype=float)
    rows, cols = im.shape
    M = np.array([[1.0, 0.0, offset[1]], [0.0, 1.0, offset[0]]])
    # Note the reversal of the dimensions
    return cv2.warpAffine(im, M, dsize=(cols, rows), flags=cv2.INTER_CUBIC)


def fft_sub_pixel_shift(im, offset):
    """
    Like sub_pixel_shift but uses a more accurate FFT phase shifting
    technique -- but it only works when then image is square

    Arguments:
        im: square float ndarray
        offset: float tuple is in (y, x) order
    """
    check.array_t(im, ndim=2, dtype=float, is_square=True)
    mea = im.shape[0]
    rng = np.arange(-(mea - 1) // 2, (mea + 1) // 2, 1)
    i, j = np.meshgrid(rng, rng)

    phasor = np.exp(
        -2.0 * complex(0.0, 1.0) * np.pi * (i * offset[1] + j * offset[0]) / mea
    )
    freq_dom = np.fft.fftshift(np.fft.fft2(im))
    freq_dom = freq_dom * phasor
    return np.real(np.fft.ifft2(np.fft.ifftshift(freq_dom)))


def sub_pixel_center(peak_im):
    com_before = com(peak_im**2)
    center_pixel = np.array(peak_im.shape) / 2
    return sub_pixel_shift(peak_im, center_pixel - com_before)


def distribution_eigen(im):
    ys, xs = np.indices(im.shape)
    pos = np.array((ys, xs)).T.reshape(-1, 2).astype(float)
    mas = im.T.reshape(-1)
    com_y = (pos[:, 0] * mas).sum() / im.sum()
    com_x = (pos[:, 1] * mas).sum() / im.sum()
    com = np.array([com_y, com_x])
    centered = pos - com
    dy = centered[:, 0] * mas
    dx = centered[:, 1] * mas
    cov = np.cov(np.array([dy, dx]))
    eig_vals, eig_vecs = LA.eig(cov)
    return eig_vals, eig_vecs, cov


def distribution_aspect_ratio(im):
    """
    Given a lump of masses compute the moment of inertia on it and use
    eigen analysis to extract the eigen values. Lumps which are concentrated
    in the center will have a small sum of eigenvalues.
    """

    eig_vals, _, _ = distribution_eigen(im)
    return utils.np_safe_divide(np.max(eig_vals), np.min(eig_vals))


def gauss2_rot_form(amp, std_x, std_y, pos_x, pos_y, rot, const, mea):
    """
    2D Gaussian in rotation form.
    Note: std and pos are in (x, y) order not (y, x)
    """
    space = np.linspace(0, mea - 1, mea)
    x, y = np.meshgrid(space, space)

    rot_mat = utils.np_rot_mat(rot)
    scale_mat = np.array([[std_x**2, 0], [0, std_y**2]])
    cov = rot_mat @ scale_mat @ np.linalg.inv(rot_mat)

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    mu = np.array([pos_x - 0.5, pos_y - 0.5])
    radius_vectors = (np.array([x.flatten(), y.flatten()]).T - mu).T

    r = np.sum(np.multiply((radius_vectors.T @ inv_cov), radius_vectors.T), axis=1)

    exp_term = amp * np.exp(-0.5 * r)
    norm_term = np.sqrt(((np.pi * 2.0) ** 2) * det_cov)

    return ((exp_term / norm_term) + const).reshape((mea, mea))


def gauss2_rho_form(
    amp, std_x, std_y, pos_x, pos_y, rho, const, mea, pixel_center=True
):
    """
    2D Gaussian in correlation (rho) form which (hopefully)
    will make fits more robust by eliminating the 2*pi/0 wrap around.
    Note: std and pos are in (x, y) order not (y, x)
    Assumes square set of pixels dim=(mea, mea)

    Args:
        amp (float): Volume of the area under the curve
        std_x (float): stdev of x
        std_y (float): stdev of y
        pos_x (float): center of gaussian
        pos_y (float): center of gaussian
        rho (float): correlation of x and y (will be zero if circular)
        const (float): offset of gaussian relative to zero
        mea (float): (measure) number of pixels along one side of square containing gaussian
    """
    mea = int(mea)
    space = np.linspace(0, mea - 1, mea)
    x, y = np.meshgrid(space, space)

    cov = np.array(
        [
            [std_x**2, rho * std_x * std_y],
            [rho * std_x * std_y, std_y**2],
        ]
    )

    try:
        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)
    except LinAlgError:
        logger.debug("Singular gauss2 covaraince, skipping")
        det_cov = 1.0
        inv_cov = np.eye(cov.shape[0])

    mu = np.array([pos_x, pos_y], dtype=float)
    if pixel_center:
        mu -= np.array([0.5, 0.5])

    radius_vectors = (np.array([x.flatten(), y.flatten()]).T - mu).T

    r = np.sum(np.multiply((radius_vectors.T @ inv_cov), radius_vectors.T), axis=1)

    exp_term = amp * np.exp(-0.5 * r)
    norm_term = np.sqrt(((np.pi * 2.0) ** 2) * det_cov)

    return ((exp_term / norm_term) + const).reshape((mea, mea))


def fit_gauss2(im, guess_params=None):
    """
    Fit im with a 2D gaussian (within limits) using rho form
    Returns the params tuple to pass to gauss2_rho_form and the fir variance

    Example:
        mea = 9
        noise = 0.01
        true_params = (1.0, 1.0, 1.0, 4, 4, 0.0, mea)
        orig_im = imops.gauss2_rho_form(*true_params)
        nois_im = orig_im + noise * np.random.randn(*orig_im.shape)
        fit_params, fit_variance = imops.fit_gauss2(nois_im)
        fit_im = imops.gauss2_rho_form(*fit_params)

        with z(_cols=3, _size=200, _cspan=(0, 0.1)):
            z.im(orig_im, f_title="original")
            z.im(nois_im, f_title="with noise")
            z.im(fit_im, f_title="from fit")

    """
    from scipy.optimize import curve_fit  # Defer slow imports

    dim = im.shape
    mea = dim[0]
    assert mea == dim[1]

    def _gauss2(x, amp, std_x, std_y, pos_x, pos_y, rho, const):
        # Drop x and add mea and convert from 2d to 1d return of gauss2
        return gauss2_rho_form(
            amp, std_x, std_y, pos_x, pos_y, rho, const, mea=mea
        ).ravel()

    im_1d = im.reshape((mea**2,))
    minimum = np.min(im_1d)

    def moments():
        """
        Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its moments
        # https://scipy-cookbook.readthedocs.io/items/FittingData.html
        """
        _im = im.copy()
        _im = _im - minimum
        total = _im.sum()
        pos_y, pos_x = np.indices(_im.shape)
        pos_y = (pos_y * _im).sum() / total
        pos_x = (pos_x * _im).sum() / total
        pos_y = min(_im.shape[0] - 1, max(0, pos_y))
        pos_x = min(_im.shape[1] - 1, max(0, pos_x))
        col = _im[:, int(pos_x)]
        amp = np.sum(_im)
        std_x = np.abs((np.arange(col.size) - pos_x) ** 2 * col).sum() / col.sum()
        std_x = max(0, std_x)
        std_x = np.sqrt(std_x)
        row = _im[int(pos_y), :]

        std_y = np.abs((np.arange(row.size) - pos_y) ** 2 * row).sum() / row.sum()
        std_y = max(0, std_y)
        std_y = np.sqrt(std_y)
        return amp, std_x, std_y, pos_x, pos_y

    if guess_params is None:
        guess_params = (*moments(), 0.0, minimum)
    else:
        if guess_params.shape[0] == 8:
            guess_params = guess_params[0:7]

    try:
        popt, pcov = curve_fit(
            _gauss2,
            np.zeros(dim),
            im_1d,
            p0=guess_params,
            bounds=(
                (0.0, 0.0, 0.0, 0.0, 0.0, -0.8, min(np.min(im_1d), guess_params[6])),
                (
                    np.inf,
                    mea / 2,
                    mea / 2,
                    mea,
                    mea,
                    0.8,
                    max(np.max(im_1d), guess_params[6]),
                ),
            ),
        )
        return (*popt, mea), (*np.diag(pcov), 0)
    except (ValueError, RuntimeError):
        # In the case of failure return a zero-amplitude
        # gaussian so that it will render well but nans for
        # the quality so the caller can tell that it failed
        return (
            (0.0, 1.0, 1.0, mea / 2, mea / 2, 0.0, 0.0, mea),
            (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        )
