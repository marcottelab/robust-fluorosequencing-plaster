import math
import os.path
import pickle

import numpy as np
import pandas as pd
import scipy.stats as scistat
from munch import Munch
from tqdm.auto import tqdm

__doc__ = """
Core code for processing traces to identify step regions.

This module is used to find "step regions" in peak traces. A step region is a region of general homogenaity in a trace the is thought to correspond to a dye in a single state.

The current algorithm uses shifts median absolute deviation to demark step regions.

To use in the context of a run:

  region_report(results, run_name, base_path="")

  with a SigprocV2Results object and a run name. run_name and base_path are used to save the results.

  This generates two data structures:

  regoins_df :  a Pandas data frame one entry per region:
      channel_i,
      peak_i,
      region_i,  # region indexes are relative to the peak. (channel,peak,region) is unique.
      start,
      end,
      l,
      min,
      max,
      rng,
      mean,
      std,
      var,
      med,
      mad

  r_mask : a numpy array of shape (n_peaks, n_cycles) with each entry contianing the relative region id for the peak trace

  regions_df and r_mask are saved as:

    {base_path}/{run_name}.regions.pkl
    {base_path}/{run_name}.regions_mask.npy

To use on a numpy arrary of peaks:

  sig_regions(sig, limit=None):

  where sig is an array of peaks (n_peaks, n_cycles). This returns the data frame and mask but does not save them.
"""

# ------------------------------------------------------------
# Stat Helpers
# ------------------------------------------------------------


def all_stat(x, stat_func=np.mean, upper_only=False, stat_offset=3):
    """
    Generate a matrix that contains the value returned by stat_func for
    all possible sub-windows of x[stat_offset:].

    stat_func is any function that takes a sequence and returns a scalar.

    if upper_only is False, values are added to both the upper and lower
    triangular sections of the matrix. If True, only the upper section
    is populated
    """
    if len(x) < stat_offset:
        return np.zeros([])

    stat = np.zeros((len(x), len(x)))

    for i in range(0, len(x)):
        for j in range(i + stat_offset, len(x)):
            v = stat_func(x[i:j])
            stat[i, j] = v

            if not upper_only:
                stat[j, i] = v

    return stat


def diag_stat(x, diagonal, stat_func=np.mean, stat_offset=3):
    """
    This is really just a windowing function. The size of the window
    corresponds to a diagonal in the all_stat matrix.

    For the input array, generate the diagonal from the all_stat matrix.

    Or, for the input array, compute stat_func for all windows of size
    "diagonal" across the array.

    stat_offset is the minimum valid window size.
    """
    if diagonal > len(x):
        return np.zeros([])

    stat = np.zeros(len(x) - diagonal)

    for i in range(len(x) - diagonal):
        stat[i] = stat_func(x[i : i + diagonal])

    return stat


def len_mat(x):
    """
    Create a symetric matrix with all possible sub-window lengths such that
    mat[i, j] or mat[j, i] is (j-i).

    This is used for generating values lenth values from index arrays when plotting.
    """
    mat = np.zeros((len(x), len(x)))
    for i in range(0, len(x)):
        for j in range(i, len(x)):
            l = j - i
            mat[i, j] = l
            mat[j, i] = l
    return mat


def start_mat(x):
    """
    TODO
    """
    mat = np.zeros((len(x), len(x)))
    for i in range(0, len(x)):
        for j in range(i, len(x)):
            l = j - i
            mat[i, j] = i
            mat[j, i] = len(x) - i
    return mat


# ------------------------------------------------------------
# MAD-based Region Detection
# ------------------------------------------------------------


def mad_regions(x, closeness=0.1, noise_floor=500, stat_offset=3, debug_return=False):
    """
    Use median absolute deviation (MAD) to find regions in a peak trace.

    closeness and noise_floor are parameters used to determine if regions
    are "close enough" to each other to stop looking for new regions.

    TODO: Find a more robust way to set closeness and noise_floor
    """
    if len(x) <= stat_offset:
        return [0, len(x)]

    # Uncomment these (and the diagonal extraction line below) to use the
    # full-matrix versions of the stats. Note that computing the full matrix
    # is expensive - O(n^2). Only use these for experimentation.
    # ---
    # med = all_stat(x, np.median, upper_only=True)
    # mad = all_stat(x, scistat.median_abs_deviation, upper_only=True)
    # ---

    # Pre-allocate to avoid re-allocations when adjusting for offsets
    x_mad = np.zeros(len(x))
    x_mad_ac = np.zeros(len(x))

    # Get the first meaningful diagonal
    # ---
    # x_mad[stat_offset:-stat_offset] = np.diagonal(mad, stat_offset*2)
    # ---
    x_mad[stat_offset:-stat_offset] = diag_stat(
        x, stat_offset * 2, scistat.median_abs_deviation
    )

    # Compute the lag-1 differential/auto-correlation
    x_mad_ac = x_mad[:-1] - x_mad[1:]

    # Find the zero crossings (these correspond to peaks in differential)
    zc = np.where(np.diff(np.sign(x_mad_ac)))[0] + 1

    # Get the intensity/height of the peaks at the zero crossings and
    # sort to create candidate regions
    pi = x_mad[zc]
    pi_sort = np.argsort(pi)
    candidates = zc[pi_sort]

    # Find regions:
    #   - Start with the full trace as the region
    #   - Get the next candidate
    #   - If the median of the regions to its right and left are "different" enough, accept it
    #   - Otherwise, stop looking for regions
    regions = [0, len(x) - 1]
    for c in candidates[::-1]:
        # Ignore cases where MAD doesn't change cycle-to-cycle
        if x_mad_ac[c] == 0 or (c - 1 > 0 and x_mad_ac[c - 1] == 0):
            continue

        # Insert and sort to keep the in order
        # (a b-tree is the right way to do this, this is just a quick way since
        #  python doesn't have a native b-tree and n is always going to be small)
        regions.append(c)
        regions.sort()
        c_i = regions.index(c)
        l = regions[c_i - 1]
        r = regions[c_i + 1]

        # Reject if it's too close to the end of another region, expect at the beginning
        if (c > stat_offset) and ((c - l) < stat_offset or (r - c) < stat_offset):
            regions.remove(c)
            continue

        # See if the change is "significant" enough. noise_floor helps with adjacent
        # regions that are near zero. 0.
        l_med = np.median(x[l:c])
        r_med = np.median(x[c:r])

        # print(f'Testing {regions[c_i-1]}:{c}:{regions[c_i+1]} {l_med:0.2f} {r_med:0.2f}')
        if math.isclose(l_med, r_med, rel_tol=closeness, abs_tol=noise_floor):
            regions.remove(c)
            break

    if debug_return:
        return regions, x_mad, x_mad_ac
    else:
        return regions


def region_stats(x, r_start, r_end):
    """
    Generate basic stats on each region. Return a dict for easy insertion into a DataFrame.
    """
    stats = Munch()
    stats["start"] = r_start
    stats["end"] = r_end
    stats["l"] = r_end - r_start
    stats["min"] = np.min(x[r_start:r_end])
    stats["max"] = np.max(x[r_start:r_end])
    stats["rng"] = stats["max"] - stats["min"]
    stats["mean"] = np.mean(x[r_start:r_end])
    stats["std"] = np.std(x[r_start:r_end])
    stats["var"] = np.var(x[r_start:r_end])
    stats["med"] = np.median(x[r_start:r_end])
    stats["mad"] = scistat.median_abs_deviation(x[r_start:r_end])

    return stats


# ------------------------------------------------------------
# Radmat/Sigproc Processing
# ------------------------------------------------------------


def sig_regions(sig, limit=None):
    """
    Compute step regions and region stats for the peak traces in sig[:limit].

    Return:
      - A Pandas DataFrame with one row per region (see below for the columns).
      - A numpy array with the same shape as sig with region indexes for each trace, e.g.:
         regions = [0, 2, 5, 8[ (len=8)
         r_mask  = [0, 0, 1, 1, 1, 2, 2, 2]
    """

    df = pd.DataFrame(
        columns=(
            "channel_i",
            "peak_i",
            "region_i",
            "start",
            "end",
            "l",
            "min",
            "max",
            "rng",
            "mean",
            "std",
            "var",
            "med",
            "mad",
        )
    )

    r_mask = np.zeros(sig.shape)
    if limit is None or len(sig) < limit:
        limit = len(sig)

    n_ch = sig.shape[1]
    ch_iz = range(n_ch)

    status = tqdm(total=n_ch * limit)

    for ch_i in ch_iz:
        status.set_description(f"Processing Channel {ch_i}")
        for peak_i in range(limit):
            # TODO: Switch this to Zack's output method
            status.update(1)

            peak = sig[peak_i][ch_i]
            regions = mad_regions(peak)

            # Get region stats and add them to the data frame
            r_start = regions[0]
            for r_i, r_end in enumerate(regions[1:]):
                rd = region_stats(peak, r_start, r_end)
                rd["channel_i"] = ch_i
                rd["peak_i"] = peak_i
                rd["region_i"] = r_i
                df = df.append(rd, ignore_index=True)

                r_mask[peak_i][ch_i][r_start:r_end] = r_i
                r_start = r_end
            # /r_i
            r_mask[peak_i][ch_i][r_end:] = r_i
        # /peak_i
    # /ch_i
    status.close()

    return df, r_mask


def region_paths(run_name, base_path):
    regions_path = os.path.join(base_path, "step_regions")

    regions_file_name = f"{run_name}.regions.pkl"
    mask_file_name = f"{run_name}.regions_mask.npy"
    df_path = os.path.join(regions_path, regions_file_name)
    mask_path = os.path.join(regions_path, mask_file_name)

    return df_path, mask_path


def region_report(results, run_name, base_path=""):
    """
    Create a region report for results where results are SigprocV2Results.

    Save the data frame and any future notebook based reports in:
      {base_path}/step_regions/{run_name}.regions.pkl
      {base_path}/step_regions/{run_name}.region_mask.npy
      {base_path}/step_regions/{future_notebook}.ipynb
    """

    regions_path = os.path.join(base_path, "step_regions")
    try:
        os.mkdir(regions_path)
    except FileExistsError:
        print(
            f"Warning: {regions_path} already exists. Possibly overwriting existing results."
        )

    df_path, mask_path = region_paths(run_name, base_path)

    # Get the peak signals and generate the regions
    sig = results.sig()
    df, r_mask = sig_regions(sig)

    df.to_pickle(df_path)
    np.save(mask_path, r_mask)

    return df, r_mask, (df_path, mask_path)


def load_region_report(run_name, base_path):
    """
    Load a region report. Return None for all files if one of the files does not exist.
    """
    df_path, mask_path = region_paths(run_name, base_path)

    try:
        with open(df_path, "rb") as df_file:
            df = pickle.load(df_file)
            r_mask = np.load(mask_path)
    except FileNotFoundError:
        return None, None, (df_path, mask_path)

    return df, r_mask, (df_path, mask_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result

    # Paths
    JOBS_FOLDER = "/erisyon/internal/jobs_folder"
    SIGPROC_PATH = "sigproc_v2/plaster_output/sigproc_v2"

    args = sys.argv[1:]

    if len(args) == 0:
        print(f"Usage: python {sys.argv[0]} run_names...")
        sys.exit(0)

    for run_name in args:

        base_run_path = os.path.join(JOBS_FOLDER, run_name)
        sig_path = os.path.join(base_run_path, SIGPROC_PATH)

        print("Processing ", run_name)
        results = SigprocV2Result(sig_path)
        df, r_mask, paths = region_report(results, run_name, base_run_path)
