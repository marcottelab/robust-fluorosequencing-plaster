"""
Common data tools such as clustering, sub-sampling, etc.
"""
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from plaster.tools.schema import check
from plaster.tools.utils import utils


def param_sweep(*param_lists):
    """
    Returns an array where all permutations of the parameter lists are enumerated
    Example:
        param_sweep([1, 2, 3], [10, 11], [20, 21])

        array([
            [ 1, 10, 20],
            [ 1, 11, 20],
            [ 2, 10, 20],
            [ 2, 11, 20],
            [ 3, 10, 20],
            [ 3, 11, 20],
            [ 1, 10, 21],
            [ 1, 11, 21],
            [ 2, 10, 21],
            [ 2, 11, 21],
            [ 3, 10, 21],
            [ 3, 11, 21]
        ])
    """
    return np.array(np.meshgrid(*param_lists)).T.reshape(-1, len(param_lists))


def kw_param_sweep(**kws):
    """
    Returns an array where all permutations of the parameter lists are enumerated
    Example:
        kw_param_sweep(a=[1, 2, 3], b=[10, 11], c=[20, 21])
        [
            {'a': 1, 'b': 10, 'c': 20},
            {'a': 1, 'b': 11, 'c': 20},
            {'a': 2, 'b': 10, 'c': 20},
            {'a': 2, 'b': 11, 'c': 20},
            {'a': 3, 'b': 10, 'c': 20},
            {'a': 3, 'b': 11, 'c': 20},
            {'a': 1, 'b': 10, 'c': 21},
            {'a': 1, 'b': 11, 'c': 21},
            {'a': 2, 'b': 10, 'c': 21},
            {'a': 2, 'b': 11, 'c': 21},
            {'a': 3, 'b': 10, 'c': 21},
            {'a': 3, 'b': 11, 'c': 21}
        ]
    """
    params = param_sweep(*list(kws.values()))
    return [{k: v for k, v in zip(kws.keys(), p)} for p in params]


def cluster(data, n_subsample=None, **kwargs):
    return_order = kwargs.pop("return_order", False)
    optimal_ordering = kwargs.pop("optimal_ordering", True)
    method = kwargs.pop("method", "weighted")
    if n_subsample is not None:
        args = arg_subsample(data, n_subsample)
    else:
        args = np.arange(data.shape[0])

    data = data[args]
    assert data.shape[0] <= 800
    if data.shape[0] > 1:
        data_dist = pdist(data)
        data_link = linkage(
            data_dist, optimal_ordering=optimal_ordering, method=method, **kwargs
        )
        order = leaves_list(data_link)
        if return_order:
            return data[order], args[order]
        else:
            return data[order]

    if return_order:
        return data, args
    else:
        return data


def subsample(data, count):
    if isinstance(data, list):
        n = len(data)
    elif isinstance(data, np.ndarray):
        n = data.shape[0]
    else:
        raise TypeError("subsample takes list of ndarray")

    if count is not None and n >= count:
        iz = np.random.choice(n, count, replace=False)
        if isinstance(data, list):
            return [data[i] for i in iz]
        elif isinstance(data, np.ndarray):
            return data[iz]

    return data


def arg_subsample(data, count):
    if isinstance(data, list):
        n = len(data)
    elif isinstance(data, np.ndarray):
        n = data.shape[0]
    elif isinstance(data, int):
        n = data
    else:
        raise TypeError("arg_subsample takes list, ndarray, or int as the data")

    if count is not None and n >= count:
        iz = np.random.choice(n, count, replace=False)
        if isinstance(data, (int, list)):
            return iz.tolist()
        elif isinstance(data, np.ndarray):
            return iz
    else:
        all_iz = np.arange(n)
        if isinstance(data, (int, list)):
            return all_iz.tolist()
        elif isinstance(data, np.ndarray):
            return all_iz


def gaussian(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -np.power((x - mu) / sigma, 2.0) / 2.0
    )


def half_nanstd(arr, mean=0.0, negative_side=True):
    """
    Sometimes one side of a distribution is the metric of interest
    so this returns the std of an array that discards elements
    above the specified mean and is symmetric about that mean.

    Arguments:
        arr: ndarray (1D)
        mean: scalar
        negative_side: Bool (if true, keep the left of the mean, otherwide right)
    """
    check.array_t(arr, ndim=1)
    arr = arr - mean
    if negative_side:
        arr = arr[arr <= 0]
        if len(arr) == 0:
            return 0.0
        return np.nanstd(np.concatenate((arr, -arr)))
    else:
        arr = arr[arr >= 0]
        if len(arr) == 0:
            return 0.0
        return np.nanstd(np.concatenate((arr, -arr)))


class ConfMat(np.ndarray):
    """
    An instance of a confusion matrix.

    This is meant to be a light-weight bag-like object that is meant to be
    passed around, plotted, analyzed but does not have logic
    for filtering or whatnot.

    Rows (axis==0) are predictions.
    Columns (axis==1) are truth.

    Eg:
        test_y = np.array([1, 0, 2, 2, 2])
        pred_y = np.array([1, 0, 1, 1, 2])
        data_tools.conf_mat(test_y, pred_y, 3)

    Result:
        [
            [1, 0, 0],
            [0, 1, 2],  # The 2 here is because there were two times that...
            [0, 0, 1]   #    ...the truth was 2 and it was predicted as a 1.
        ]

    For a nice viz of the confusion matrix, see here:
    https://stackoverflow.com/a/50671617
    (Except that viz is transposed compared to our mats.)
    """

    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    @classmethod
    def from_array(cls, arr):
        return cls(arr)

    @classmethod
    def from_true_pred(cls, true, pred, true_dim, pred_dim):
        assert true.ndim == 1 and pred.ndim == 1
        # This assert can take upwards of 50ms, and given that this fn is called as an inner loop, we can save some time if we omit it
        # assert np.all((0 <= true) & (true < true_dim) & (0 <= pred) & (pred < pred_dim))
        index = (pred * true_dim + true).astype(int)
        return cls(
            np.reshape(
                np.bincount(index, minlength=true_dim * pred_dim), (pred_dim, true_dim)
            )
        )

    def precision(self):
        diag = np.diagonal(self)
        sum_rows = np.sum(self, axis=1)
        return np.divide(
            diag, sum_rows, out=np.zeros(self.shape[0]), where=sum_rows != 0
        )

    def recall(self):
        diag = np.diagonal(self)
        sum_cols = np.sum(self, axis=0)
        return np.divide(
            diag, sum_cols, out=np.zeros(self.shape[0]), where=sum_cols != 0
        )

    def n_calls(self):
        # the 0th row/col are "null" entries and not calls.  Calls get "moved" into
        # the 0-row based on score-thresholding to exclude them from precision calculations
        # but still include them in the column-sum for recall calculations.  So number of calls
        # at the current threshold is the count of entries excluding row&col 0.
        return np.sum(self[1:, 1:])

    def scale_by_abundance(self, abundance):
        """
        DHW 9/28/2020 - I profiled the check.array_t and the assert and in practice the impact appears minimal (<1ms in my test case)
        """
        check.array_t(abundance, shape=(self.shape[1],))
        assert np.all((abundance >= 1.0) | (abundance == 0.0))
        return (self * abundance).astype(int)

    def false_calls(self, elem_i, n_false):
        """
        For a nice viz of the confusion matrix, see here:
        https://stackoverflow.com/a/50671617
        (Except that viz is transposed compared to our mats.)

        There's two kinds of off-diagonal failures wrt to any element "A":
            * FALSE-POSITIVES: Elements that are called A but are not A.
              I use a mnemonic: "im-POS-ters", ie the false "POS-itives"
            * FALSE-NEGATIVES: Elements that are not A but that steal calls
              from true A's. I think of these as "thieves" stealing from
              the truth.

            These are symmetric relationships:
            If B is an imposter of A then A is a thief of B.

            True negatives wrt to "A" are all of the elements outside the row
            and col of A.
        """

        n_dim = self.shape[0]
        assert self.shape[1] == n_dim

        if n_false >= n_dim:
            return None

        check.affirm(0 <= elem_i < n_dim, "elem_i out of range")

        # For now, only square matrices are supported
        assert self.shape[0] == self.shape[1]

        # Grab sums BEFORE removing the diagonal
        row_sum = self[elem_i, :].sum()
        col_sum = self[:, elem_i].sum()

        # FETCH the top falses (imposters and thieves) with the diag removed
        # to avoid self-collision. Make a copy first
        copy = np.copy(self)
        np.fill_diagonal(copy, 0)
        sorted_false_pos_pep_iz = np.argsort(copy[elem_i, :])[::-1]
        sorted_false_neg_pep_iz = np.argsort(copy[:, elem_i])[::-1]

        false_positive_tuples = [
            (
                f"FP{i}",
                sorted_false_pos_pep_iz[i],
                float(
                    utils.np_safe_divide(
                        copy[elem_i, sorted_false_pos_pep_iz[i]], row_sum, default=0.0
                    )
                ),
            )
            for i in range(n_false)
            if sorted_false_pos_pep_iz[i] > 0
        ]

        false_negative_tuples = [
            (
                f"FN{i}",
                sorted_false_neg_pep_iz[i],
                float(
                    utils.np_safe_divide(
                        copy[sorted_false_neg_pep_iz[i], elem_i], col_sum, default=0.0
                    )
                ),
            )
            for i in range(n_false)
            if sorted_false_neg_pep_iz[i] > 0
        ]

        return false_positive_tuples + false_negative_tuples


def interp_missing_in_matrix(mat):
    """
    This is an attempt at 2D interpolation of nans
    But I found it to fail in various corner cases that aren't clear
    so it is probably the start but is not yet useful
    """
    raise NotImplementedError
    from scipy import interpolate

    x = np.arange(0, mat.shape[1])
    y = np.arange(0, mat.shape[0])
    mat = np.ma.masked_invalid(mat)
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~mat.mask]
    y1 = yy[~mat.mask]
    new_mat = mat[~mat.mask]

    return interpolate.griddata((x1, y1), new_mat.ravel(), (xx, yy), method="cubic")


def arrays_subsample(arrays, n_samples):
    """
    arrays is a tuple of parallel arrays all with the same shape[0]
    n_samples is an int.
    Draw n_samples from all the arrays' [0] dimension
    """
    n_rows = arrays[0].shape[0]
    for a in arrays:
        assert (
            a.shape[0] == n_rows
        ), "All arrays in arrs_subsample must have identical shape[0]"

    iz = np.random.choice(n_rows, n_samples)
    return tuple([a[iz] for a in arrays])


def arrays_mask(arrays, mask):
    """
    arrays is a tuple of parallel arrays all with the same shape[0]
    mask is a bool array of shape[0] also.
    Return the masked rows from all the arrays
    """
    n_rows = arrays[0].shape[0]
    for a in arrays:
        assert (
            a.shape[0] == n_rows
        ), "All arrays in arrs_subsample must have identical shape[0]"

    check.array_t(mask, shape=(n_rows,), dtype=bool)

    return tuple([a[mask] for a in arrays])
