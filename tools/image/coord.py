import numpy as np

"""
Coordinate helpers:
    Images in are treated as numpy arrays as they are in cv2.
    numpy uses the convention of a .shape which is in (vertical, horizontal) order.
    To prevent getting confused by this, the following helpers all aliasing
    of .x, .y and .w, .h properties

    Naming convention:
        mea: (Mea)sure refers to a 1-dimensional width or height
        dim: Refers to a 2-dimension width and height
        loc: Refers to a 2-dimensional location
        When passed around as a tuple they are always in (vert, horz) order

    Usage:
        from plaster.tools.image.coord import XY, YX, WH, HW, ROI

        loc = XY(2, 3)
        assert loc.x == 2
        assert loc.y == 3
        vloc == (3, 2)  # Note, reverse order!

        loc = YX(3, 2)
        assert loc.x == 2
        assert loc.y == 3
        assert loc == (3, 2)

        # ... Same for WH and HW

    These coordinate helpers convert themselves to a tuple sub-class called "Coord".
    This helper implements many of the usualy vector operators so that
    you can do basic operations. Examples:

        loc += XY(10, 20)
        assert loc == (12, 23)

        # You can multiply by a scalar by the scalar must be on the right side:
        dim = WH(10, 20) * 2  # Good
        dim = 2 * WH(10, 20)  # Bad, results in two copies of (10, 20), ie: (10, 20, 10, 20)

    The ROI helper allows for easy slicing.
    dim = WH(10, 10)
    image = np.zeros(dim)
    roi = ROI(XY(1,1), dim - WH(2,2))
    cropped_image = image[roi]  # 1 pixel border removed
    assert image[1:9,1:9] == cropped_image[:,:]
"""


def XY(x, y):
    return Coord(y, x)


def YX(y, x=None):
    if isinstance(y, tuple) or isinstance(y, np.ndarray) or isinstance(y, list):
        assert x is None
        return Coord(y[0], y[1])
    return Coord(y, x)


def WH(w, h):
    """
    Wrapper for making (h, w) tuples from reverse order constants:
        WH(w, h) -> (h, w)

    If you get an error saying:
        TypeError: WH() missing 1 required positional argument: 'h'
    Then it is likely that you meant to use HW since numpy shape tuples are in HW order

    EG:
        Bad: WH(i.shape)
        Good: HW(i.shape)
    """
    return Coord(h, w)


def HW(h, w=None):
    if isinstance(h, tuple) or isinstance(h, np.ndarray) or isinstance(h, list):
        assert w is None
        return Coord(h[0], h[1])
    return Coord(h, w)


class Coord(tuple):
    """
    Coordinate helper. Basically it just adds the convenience aliases for x, y, w, h.

    Usage:
        dim = WH(10, 20)  # Converts to tuple (20, 10)
        loc = XY(10, 20)  # Converts to tuple (20, 10)
        dim.w == 10
        loc.x == 10
    """

    def __new__(cls, y, x):
        return super().__new__(cls, (int(y), int(x)))

    def __getnewargs__(self):
        return (self.y, self.x)

    @property
    def x(self):
        return self[1]

    @property
    def y(self):
        return self[0]

    @property
    def w(self):
        return self[1]

    @property
    def h(self):
        return self[0]

    def mag(self):
        return self[0] ** 2 + self[1] ** 2

    def __add__(self, other):
        assert isinstance(other, Coord)
        return Coord(self[0] + other[0], self[1] + other[1])

    def __sub__(self, other):
        assert isinstance(other, Coord)
        return Coord(self[0] - other[0], self[1] - other[1])

    def __floordiv__(self, other):
        return Coord(self[0] // other, self[1] // other)

    def __truediv__(self, other):
        return Coord(self[0] / other, self[1] / other)

    def __mul__(self, other):
        return Coord(self[0] * other, self[1] * other)


def ROI(loc, dim, center=False):
    """
    Slicing operator for extracting a region of interest (ROI) from a 2D array.
    Eg: Accumulate src into dst at (10, 10)
        dst = np.zeors(HW(100, 100))
        src = np.zeors(HW(50, 50))
        roi = ROI(YX(10, 10), HW(src.shape))
        dst[roi] += src
    """
    loc = YX(loc)
    dim = HW(dim)
    if center:
        loc -= dim // 2
    return (slice(loc.y, loc.y + dim.h), slice(loc.x, loc.x + dim.w))


def roi_shift(roi, offset):
    """Returns a new ROI shifted by offset"""
    assert len(offset) == 2
    yx = YX(roi[0].start + offset[0], roi[1].start + offset[1])
    hw = HW(roi[0].stop - roi[0].start, roi[1].stop - roi[1].start)
    return ROI(yx, hw)


# def roi_clip(roi, full_dim):
#     clipped = clip2d(
#         roi[1].start, full_dim[1], roi[1].stop - roi[1].start,
#         roi[0].start, full_dim[0], roi[0].stop - roi[0].start,
#     )
#     if clipped[0] is None:
#         return ROI((0, 0), (0, 0))
#
#     return clipped[0]


def roi_center(dim, percent=0.5):
    """Return loc and dim of the center percent (by single linear measure)"""
    dim = HW(dim)
    center_dim = dim * percent
    offset = (dim - center_dim) // 2
    return ROI(YX(offset), center_dim)


class Rect:
    def __init__(self, b, t, l, r):
        self.b = b
        self.t = t
        self.l = l
        self.r = r

    def w(self):
        return self.r - self.l + 1

    def h(self):
        return self.t - self.b + 1

    def roi(self):
        return (slice(self.b, self.t), slice(self.l, self.r))

    def to_list(self):
        return [self.l, self.r, self.b, self.t]

    @classmethod
    def from_list(cls, list_):
        inst = cls(0, 0, 0, 0)
        inst.l, inst.r, inst.b, inst.t = list_
        return inst


def clip1d(tar_x, tar_w, src_w):
    """
    Placing the src in the coordinates of the target
    Example: tar_x = -1, tar_w = 5, src_w = 4
        ssss
         ttttt
    """

    src_l = 0
    src_r = src_w
    tar_l = tar_x
    tar_r = tar_x + src_w

    # If target is off to the left then bound tar_l to 0 and push src_l over
    if tar_l < 0:
        src_l = -1 * tar_l
        tar_l = 0
        # This may cause src_l to go past its own width

    # If src_l went past its own width (from the above)
    if src_l >= src_w:
        # src l went past its own width
        return None, None, None

    # If the target went over the target right edge
    #   01234
    #   tttt
    #    ssss
    if tar_r >= tar_w:
        src_r -= tar_r - tar_w
        # This may cause src_r to be less than the left

    if src_r - src_l <= 0:
        return None, None, None

    return tar_l, src_l, src_r - src_l


def clip2d(tar_x, tar_w, src_w, tar_y, tar_h, src_h):
    tar_l, src_l, src_w = clip1d(tar_x, tar_w, src_w)
    tar_t, src_t, src_h = clip1d(tar_y, tar_h, src_h)

    tar_roi = None
    src_roi = None
    if tar_l is not None and tar_t is not None:
        tar_roi = ROI(XY(tar_l, tar_t), WH(src_w, src_h))
        src_roi = ROI(XY(src_l, src_t), WH(src_w, src_h))

    return tar_roi, src_roi
