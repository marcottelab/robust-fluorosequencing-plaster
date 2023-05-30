import numpy as np
import pandas as pd
import structlog
from scipy.signal import savgol_filter

from plaster.tools.schema import check
from plaster.tools.utils import stats, utils

logger = structlog.get_logger()


def radiometry_histogram_analyzer(sig):
    """
    This is a bespoke histogram analyzer for radiometry to extract certain guesses.
    Assumptions:
      * There will be a dominante peak with a mean near zero and easily separated
        from a one peak which is above 3 stdevs of the zero peak
      * The one peak will be most dominant at the last cycle
      * The zero peak will have a negative side that is essentially uncontaminated
        by signal.
    """

    # REMOVE 0s -- these are the result of failures
    sig = sig.flatten()
    sig = sig[sig != 0.0]

    lft, rgt = np.percentile(sig, (0.1, 97.0))

    # Use the negative values to estimate the sigma of the zero peak
    zero_sigma = stats.half_nanstd(sig)
    if zero_sigma == 0.0:
        z.hist(sig)
        print("ERROR: Unable to determine beta on channel")
        return 0.0, 1.0, 0.0, [0.0], [0.0], 0.0, 0.0, 0.0

    # Go n-stds on the right side of the zero peak
    zero_hist_right_side_thresh = 3.0 * zero_sigma
    zero_bins = np.linspace(lft, zero_hist_right_side_thresh, 200)
    zero_hist, zero_edges = np.histogram(
        sig[sig < zero_hist_right_side_thresh], bins=zero_bins
    )
    zero_edges = zero_edges[1:]
    top = np.max(zero_hist)
    zero_mu = zero_edges[np.argmax(zero_hist)]
    rgt = np.percentile(sig, 97)
    one_bins = np.linspace(zero_hist_right_side_thresh, rgt, 200)
    one_hist, one_edges = np.histogram(
        sig[sig > zero_hist_right_side_thresh], bins=one_bins
    )
    one_edges = one_edges[1:]

    # Smooth this with a savgol filter
    one_filt = savgol_filter((one_edges, one_hist), window_length=27, polyorder=3)
    top = np.max(one_hist)
    beta = one_edges[np.argmax(one_filt[1])]

    return zero_mu, zero_sigma, beta, one_edges, one_filt[1], lft, rgt, top


# def beta_per_channel(sig):
#     check.array_t(sig, ndim=3)
#     if report_params.get("beta_per_channel"):
#         # This is a hack to allow manually setting for situations
#         # where radiometry_histogram_analyzer() isn't doing well
#         print("USING USER SPECIFIED BETA PER CHANNEL")
#         beta = np.array(report_params.get("beta_per_channel"))
#     else:
#         beta = np.zeros((n_channels))
#         for ch_i in range(n_channels):
#             _, _, _beta, _, _, _, _, _ = radiometry_histogram_analyzer(sig[:, ch_i])
#             beta[ch_i] = _beta
#     return np.nan_to_num(beta)


def field_quality(ims_import, sigproc_v2, field_quality_thresh):
    """
    Builds up a (field, channel) DataFrame with quality and alignment

    Arguments:
        ims_import: ImsImportResult
        sigproc_v2: SigprocV2Result
        field_quality_thresh: Need a way to auto-tune this

    Returns:
        field_df: (field_i, channel_i, alignment, mean_quality, good_field_alignment, good_field_quality)
        field_align_thresh: float (derived from the image size)
    """

    field_df = sigproc_v2.fields().copy()
    n_fields = sigproc_v2.n_fields
    assert n_fields == field_df.field_i.nunique()

    index = ["field_i", "channel_i"]

    # ALIGNMENT: Max field alignment is 10% of the width or height of the import image
    # It might be possible to increase this but as the alignment gets worse it begins
    # to break the assumption that the aln coordinates can be used to look up
    # spatial calibration information such as the regional PSF.
    field_align_thresh = int(0.15 * ims_import.dim)
    field_df["alignment"] = np.sqrt(field_df.aln_x**2 + field_df.aln_y**2)
    field_df = field_df.set_index(index)
    field_df = field_df.groupby(index).alignment.max().reset_index()

    # MEAN QUALITY (each channel, all cycles)
    qual_df = ims_import.qualities()
    if len(qual_df) == 0:
        # If there is no quality data from ims_import then create one with all NaNs
        qual_df = field_df.copy()[index]
        qual_df["quality"] = np.nan
    qual_df = qual_df.groupby(index).mean()[["quality"]]

    field_df = field_df.set_index(index).join(qual_df)
    field_df["good_field_alignment"] = field_df.alignment < field_align_thresh
    field_df["good_field_quality"] = field_df.quality > field_quality_thresh

    if np.all(np.isnan(field_df.quality)):
        field_df.good_field_quality = True

    return field_df.reset_index(), field_align_thresh


def dark_thresh_per_channel(sig, dark_thresh_in_stds=4.0):
    """
    Find the dark threshold (the intensity of transition from dark to light)
    by computing a one-sided std on each channel

    Arguments:
        sig: ndarray(n_peaks, n_channels, n_cycles)

    Returns:
        dark_thresh_per_ch: ndarray(n_channels)
            The estimated intensity threshold between off and on
    """
    check.array_t(sig, ndim=3)

    n_channels = sig.shape[1]
    dark_thresh_per_ch = np.zeros((n_channels,))
    for ch_i in range(n_channels):
        zero_sigma_est = stats.half_nanstd(sig[:, ch_i].flatten())
        dark_thresh_per_ch[ch_i] = dark_thresh_in_stds * zero_sigma_est
    return dark_thresh_per_ch


def features(ims_import, sigproc_v2, dark_thresh_in_stds, n_samples=None):
    """
    Extract a variety of features for every peak

    Arguments:
        ims_import: ImsImportResult
        sigproc_v2: SigprocV2Result

    Returns:
        per_peak_df: Dataframe peak traits independent of channel (one row per peak)
        ch_peak_df: Dataframe peak traits by channel (one row per peak per channel)

    TO DO:
        Consider parallelization
    """

    raise DeprecationWarning
    # Killed off by the per-channel below

    from scipy.spatial.distance import cdist

    from plaster.run.prep.prep_worker import triangle_dytmat

    per_peak_df = sigproc_v2.peaks()

    if n_samples is not None:
        per_peak_df = per_peak_df.sample(n_samples)

    # Convenience aliases
    n_channels = sigproc_v2.n_channels
    n_cycles = sigproc_v2.n_cycles
    # n_peaks = per_peak_df.peak_i.max() + 1
    # assert len(per_peak_df) == n_peaks
    im_mea = ims_import.dim

    # Merge in stage metadata
    if ims_import.has_metadata():
        meta_df = ims_import.metadata()
        column_names = ["field_i", "stage_x", "stage_y"]
        if all([col in meta_df for col in column_names]):
            stage_df = meta_df[column_names].groupby("field_i").mean()
            per_peak_df = pd.merge(
                left=per_peak_df, right=stage_df, left_on="field_i", right_on="field_i"
            )
            per_peak_df["flowcell_x"] = per_peak_df.stage_x + per_peak_df.aln_x
            per_peak_df["flowcell_y"] = per_peak_df.stage_y + per_peak_df.aln_y

    per_peak_df["radius"] = np.sqrt(
        (per_peak_df.aln_x - im_mea // 2) ** 2 + (per_peak_df.aln_y - im_mea // 2) ** 2
    )

    sampled_sig = sigproc_v2.sig()[per_peak_df.peak_i]
    sampled_noi = sigproc_v2.noi()[per_peak_df.peak_i]
    dark_thresh_per_ch = dark_thresh_per_channel(sampled_sig, dark_thresh_in_stds)

    per_ch_dfs = []
    for ch_i in range(n_channels):
        dark_thresh = dark_thresh_per_ch[ch_i]
        ch_sig = sampled_sig[:, ch_i, :]
        ch_noi = sampled_noi[:, ch_i, :]

        # has_neighbor_stats = run.sigproc_v2.has_neighbor_stats()
        # if has_neighbor_stats:
        #     try:
        #         nei = run.sigproc_v2.neighborhood_stats()
        #         assert nei.shape[0] == n_peaks
        #         # There's an issue here on some fields that have no neighbor info
        #     except TypeError:
        #         has_neighbor_stats = False

        # "Lifespan" is the cycles over which a peak is "on". Abbreviated "lif"
        # Use the cosine distance to determine lif_len. This is based on trying
        # practically every distance metric in the cdist family and seeing that
        # cosine tends to do the best job. There is likely a better theoretical
        # understanding to be had for this. The main goal is to approximately
        # assign row lengths noisy noisy rows like:
        #   [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # Ie, is that length 1 or length 7?  7 Seems more reasonable and that
        # would be the result of the cosine distance.
        dyts1 = triangle_dytmat(n_cycles, n_dyes=1, include_nul_row=False)
        dyt1_dists = cdist(ch_sig > dark_thresh, dyts1, "cosine")

        # life length is the measured by the arg minimum cdist along each row
        # But we need to add one because the triangle_dytmat does not include
        # the nul row (all zeros) thus the dyts1[0] has length 1, not 0
        lif_len = np.argmin(dyt1_dists, axis=1) + 1

        # Extract signal during and after the lifetime ("afl" = "afterlife")
        row_iz, col_iz = np.indices(ch_sig.shape)
        sig_lif = np.where(col_iz < lif_len[:, None], ch_sig, np.nan)
        sig_afl = np.where(col_iz >= lif_len[:, None], ch_sig, np.nan)

        def stats(mat, prefix):
            with utils.np_no_warn():
                return pd.DataFrame(
                    {
                        f"{prefix}_med": np.nanmedian(mat, axis=1),
                        f"{prefix}_men": np.nanmean(mat, axis=1),
                        f"{prefix}_std": np.nanstd(mat, axis=1),
                        f"{prefix}_iqr": np.subtract(
                            *np.nanpercentile(mat, [75, 25], axis=1)
                        ),
                        f"{prefix}_max": np.nanmax(mat, axis=1),
                        f"{prefix}_min": np.nanmin(mat, axis=1),
                    }
                )

        ch_peak_df = pd.DataFrame(
            dict(
                peak_i=per_peak_df.peak_i,
                field_i=per_peak_df.field_i,
                channel_i=ch_i,
                lif_len=lif_len,
                noi_cy0=ch_noi[:, 0],
                dark_cy0=ch_sig[:, 0] <= dark_thresh,
            )
        )
        ch_peak_df = pd.concat(
            (
                ch_peak_df,
                stats(sig_lif, "lif"),
                stats(sig_afl, "afl"),
            ),
            axis=1,
        )

        # Multi-channel can have zero-length lives in SOME channels but not others.
        # This is because the peak finder only requires that ONE channel have
        # non-zero signal. But the above calculations for lif_len
        # will not handle this situation as it doesn't (and must not) include
        # the "all zeros (aka nul) row".
        # Thus, the lif_len will report length 1 when the true length is 0
        # for those channels with no signal at cycle 0.
        # Thus, we have to detect this situation by looking
        # for rows with lif_len == 1 where cy[0] value is very low.
        true_lif_0 = ch_peak_df[
            (ch_peak_df.lif_len == 1) & (ch_peak_df.lif_men < dark_thresh)
        ]
        ch_peak_df.loc[:, "lif_len"] = 0.0

        ch_peak_df.loc[:][
            ["lif_med", "lif_men", "lif_std", "lif_iqr", "lif_max", "lif_min"]
        ] = 0.0

        per_ch_dfs += [ch_peak_df]

    return per_peak_df, pd.concat(per_ch_dfs)


def features_one_ch(ims_import, sigproc_v2, ch_i, n_samples=None):
    """
    Extract a variety of features for every peak for a single channel

    Arguments:
        ims_import: ImsImportResult
        sigproc_v2: SigprocV2Result

    Returns:
        per_peak_df: Dataframe peak traits, one row per peak, possible sampled
    """
    from scipy.spatial.distance import cdist

    from plaster.run.prep.prep_worker import triangle_dytmat

    per_peak_df = sigproc_v2.peaks()

    if n_samples is not None:
        per_peak_df = per_peak_df.sample(n_samples, replace=True)

    # Convenience aliases
    n_channels = sigproc_v2.n_channels
    n_cycles = sigproc_v2.n_cycles
    im_mea = ims_import.dim

    # Merge in stage metadata
    if ims_import.has_metadata():
        meta_df = ims_import.metadata()
        column_names = ["field_i", "stage_x", "stage_y"]
        if all([col in meta_df for col in column_names]):
            # This is a little hack to cause field_i to match the sigproc
            # data in the case that sigproc was done on a subset of fields,
            # in which case the sigproc field_i will still start at 0,
            # but the meta_df field_i will start at the actual field number.
            # If this is the case, the merge operation will fail and leave
            # the peaks_df empty.
            if np.min(per_peak_df.field_i) != np.min(meta_df.field_i):
                logger.warning(
                    "Normalizing meta_df field_i to start at 0, because (probably) this sigproc job was run on a subset of fields."
                )
                meta_df.field_i = meta_df.field_i - np.min(meta_df.field_i)

            stage_df = meta_df[column_names].groupby("field_i").mean()
            per_peak_df = pd.merge(
                left=per_peak_df, right=stage_df, left_on="field_i", right_on="field_i"
            )
            per_peak_df["flowcell_x"] = per_peak_df.stage_x + per_peak_df.aln_x
            per_peak_df["flowcell_y"] = per_peak_df.stage_y + per_peak_df.aln_y

    per_peak_df["radius"] = np.sqrt(
        (per_peak_df.aln_x - im_mea // 2) ** 2 + (per_peak_df.aln_y - im_mea // 2) ** 2
    )

    sampled_sig = sigproc_v2.sig()[per_peak_df.peak_i, ch_i]
    sampled_noi = sigproc_v2.noi()[per_peak_df.peak_i, ch_i]

    # "Lifespan" is the cycles over which a peak is "on". Abbreviated "lif"
    # Use the cosine distance to determine lif_len. This is based on trying
    # practically every distance metric in the cdist family and seeing that
    # cosine tends to do the best job. There is likely a better theoretical
    # understanding to be had for this. The main goal is to approximately
    # assign row lengths noisy noisy rows like:
    #   [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # Ie, is that length 1 or length 7?  7 Seems more reasonable and that
    # would be the result of the cosine distance.
    dyts1 = triangle_dytmat(n_cycles, n_dyes=1, include_nul_row=False)
    dyt1_dists = cdist(sampled_sig, dyts1, "cosine")

    # life length is the measured by the arg minimum cdist along each row
    # But we need to add one because the triangle_dytmat does not include
    # the nul row (all zeros) thus the dyts1[0] has length 1, not 0
    lif_len = np.argmin(dyt1_dists, axis=1) + 1

    # Extract signal during and after the lifetime ("afl" = "afterlife")
    row_iz, col_iz = np.indices(sampled_sig.shape)
    sig_lif = np.where(col_iz < lif_len[:, None], sampled_sig, np.nan)
    sig_afl = np.where(col_iz >= lif_len[:, None], sampled_sig, np.nan)

    def stats(mat, prefix, peak_iz):
        with utils.np_no_warn():
            df = pd.DataFrame(
                {
                    "peak_i": peak_iz,
                    f"{prefix}_med": np.nanmedian(mat, axis=1),
                    f"{prefix}_men": np.nanmean(mat, axis=1),
                    f"{prefix}_std": np.nanstd(mat, axis=1),
                    f"{prefix}_iqr": np.subtract(
                        *np.nanpercentile(mat, [75, 25], axis=1)
                    ),
                    f"{prefix}_max": np.nanmax(mat, axis=1),
                    f"{prefix}_min": np.nanmin(mat, axis=1),
                }
            )
            return df

    per_peak_df["lif_len"] = lif_len
    per_peak_df["noi_cy0"] = sampled_noi[:, 0]

    df0 = per_peak_df
    df1 = stats(sig_lif, "lif", per_peak_df.peak_i)
    df2 = stats(sig_afl, "afl", per_peak_df.peak_i)

    df3 = utils.easy_join(df0, df1, "peak_i")
    df4 = utils.easy_join(df3, df2, "peak_i")

    return df4


def _noise_thresh_one_channel(noi_cy0, noi_thresh_in_stds=2.5):
    """
    Use a filter to smooth the histogram of a noise distribution
    and return a threshold based on noi_thresh_in_stds to the right
    of the peak.
    """
    check.array_t(noi_cy0, ndim=1)

    bins = np.linspace(0, np.percentile(noi_cy0, 99), 200)

    # Smooth this with a savgol filter and use the main peak
    # to one-sideds estimate the std
    _hist, _edges = np.histogram(noi_cy0, bins=bins)
    _edges = _edges[1:]
    _filt = savgol_filter((_edges, _hist), window_length=11, polyorder=3)
    _filt = _filt[1]
    center = _edges[np.argmax(_filt)]
    std = stats.half_nanstd(noi_cy0, mean=center)
    return center + noi_thresh_in_stds * std


def noise(ch_peak_df, noi_thresh_in_stds=2.5):
    """
    Get the noise and thresholds for each channel.

    Returns:
        noi_cy0_per_ch: list(ndarray)
        noi_thresh_per_ch: ndarray(n_channels)
    """
    n_channels = ch_peak_df.channel_i.nunique()
    noi_cy0_per_ch = [None] * n_channels
    noi_thresh_per_ch = np.zeros((n_channels,))
    for ch_i in range(n_channels):
        noi_cy0_per_ch[ch_i] = ch_peak_df[
            (ch_peak_df.channel_i == ch_i) & (ch_peak_df.dark_cy0 == 0)
        ].noi_cy0
        noi_thresh_per_ch[ch_i] = _noise_thresh_one_channel(
            noi_cy0_per_ch[ch_i].values, noi_thresh_in_stds=noi_thresh_in_stds
        )
    return noi_cy0_per_ch, noi_thresh_per_ch


def monotonic(bal_sig, beta, lif_len, monotonic_threshold=1.0):
    """
    Examine a cycle-balanced radat (one channel) for the
    maximum increase in intensity per row and normalize
    by beta. This puts it roughly into units of dye-count.

    Arguments:
        bal_sig: ndarray(n_peaks, n_cycle). Cycle balanced
        beta: float. approximate intensity per dye
        lif_len: ndarray(n_peaks). lifespan of each row in cycles
        monotonic_threshold: float. In dye count units, max increase alloed

    Returns:
        monotonic_metric: ndarray((n_peaks)). Max increase in any cycle for each peak in dye counts
        good_mask: ndarray((n_peaks), dtype=bool).
            Where monotonic_metric > monotonic_threshold and life_len > 1 and first cycle is not dark
    """
    check.array_t(bal_sig, ndim=2)
    check.t(beta, float)
    check.array_t(lif_len, ndim=1)
    check.t(monotonic_threshold, float)
    assert len(lif_len) == bal_sig.shape[0]

    _, col_iz = np.indices(bal_sig.shape)
    sig_lif = np.where(col_iz < lif_len[:, None], bal_sig, np.nan)

    with utils.np_no_warn():
        d = np.diff(sig_lif, append=0.0, axis=1)
        maxs_diff = np.nanmax(d, axis=1)
        monotonic_metric = maxs_diff / beta
        monotonic_metric_exceeds_thresh_mask = monotonic_metric > monotonic_threshold
        lif_gt_1_mask = lif_len > 1
        starts_high_mask = bal_sig[:, 0] > 0.8 * beta
        good_mask = ~(
            monotonic_metric_exceeds_thresh_mask & lif_gt_1_mask & starts_high_mask
        )

    return monotonic_metric, good_mask


def _sig_in_range(sigproc_v2):
    """
    Returns a mask indicating which signals are in range.  "In range" means
    able to be represented as np.float32 for the moment.  The radmat is float64,
    but some downstream ops (e.g. classify_rf) want to represent as float32.

    Or we could just truncate these signals?  But these signals really are
    probably bad.
    """
    finfo = np.finfo(np.float32)
    max_allowed = finfo.max
    min_allowed = finfo.min

    radmat = utils.mat_flatter(sigproc_v2.sig())
    peak_max = np.max(radmat, axis=1)
    peak_min = np.min(radmat, axis=1)
    in_range_mask = (peak_min > min_allowed) & (peak_max < max_allowed)
    return in_range_mask


def build_filter_df(sigproc_v2, field_df, per_peak_df, ch_peak_df, noi_thresh):
    _field_df = (
        field_df.groupby("field_i")
        .agg(dict(good_field_alignment=np.nanmin, good_field_quality=np.nanmin))
        .reset_index()
    )
    df = per_peak_df.merge(
        right=_field_df[["field_i", "good_field_alignment", "good_field_quality"]],
        how="inner",
        on="field_i",
    )

    # nanmax across the channels on each peak to get the highest noise at cycle 0
    # Note, this is assuming that all channels have relatively similar
    # noise profiles. If this is not true then we need a more complicated
    # calculation where we look at the distance of the noise compared
    # to something like a z-score. For now, I'm using a single threshold
    # in common for all channels
    max_noi0_over_all_channels = (
        ch_peak_df.groupby("peak_i").agg(dict(noi_cy0=np.nanmax)).reset_index()
    )
    max_noi0_over_all_channels.rename(columns=dict(noi_cy0="max_noi0"), inplace=True)
    df = df.set_index("peak_i").join(max_noi0_over_all_channels.set_index("peak_i"))
    df["good_noi"] = df.max_noi0 < noi_thresh

    # TODO: Monotonic?
    # for ch_i in range(sigproc_v2.n_channels):
    #     sig = sigproc_v2.sig()[:, ch_i, :]
    #     _df = ch_peak_df[ch_peak_df.channel_i == ch_i]
    #     monotonic_metric, monotonic_good_mask = monotonic(
    #         sig, beta_per_channel[ch_i], _df.lif_len.values, monotonic_threshold=monotonic_threshold
    #     )
    # df["good_monotonic_any_ch"] = ch_peak_df.groupby("peak_i").agg({"good_monotonic": [np.nanmax]}).values.flatten()
    # df["good_monotonic_all_ch"] = ch_peak_df.groupby("peak_i").agg({"good_monotonic": [np.nanmin]}).values.flatten()

    # TODO: SNR? (Consider using structure of sig vs noi)
    # _snr = run.sigproc_v2.snr()[all_fea_df.peak_i, :, 0]
    # all_fea_df["good_snr"] = np.all(_snr[:, :] > ch_valley, axis=1)
    # all_fea_df.pass_all = all_fea_df.pass_all & all_fea_df.good_snr

    # TODO: how best to handle out-of-range radmat values?  For now
    # just reject signal that is outside the boundaries of float32
    df["sig_in_range"] = _sig_in_range(sigproc_v2)

    df["pass_quality"] = (
        df.good_field_alignment & df.good_field_quality & df.good_noi & df.sig_in_range
    )

    return df.reset_index()


def cycle_balance():
    raise NotImplementedError


def radmat_filter_mask(rad_filter_result):
    """
    Return a mask indicating which radmat rows pass the most recent application of
    filtering as represented by the "pass_quality" column of RadFilterResult.filter_df.
    """
    return (rad_filter_result.filter_df.pass_quality).astype(bool).values
