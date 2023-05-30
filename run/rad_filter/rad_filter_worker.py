from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.rad_filter.rad_filter_params import RadFilterParams
from plaster.run.rad_filter.rad_filter_result import RadFilterResult
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2Result


def rad_filter(
    params: RadFilterParams,
    ims_import: ImsImportResult,
    sigproc_v2: SigprocV2Result,
):
    """
    Creates a variety of metrics that are used to pre-filter data
    before it heads into the classifier.

    field_df (Row per field):
        Field Quality
        Field Alignment
    field_aligned_thresh:
        The threshold choice made to accept field alignments
    per_peak_df (Row per peak):
        field_peak_i (the peak i in the frame's peak address space)
        aln_y, aln_x, radius
    ch_peak_df (Row per peak and channel)
        lif_len
        lif_*: Stats about intensities during lifespan
        afl_*: Stats about intensities after lifespan
    noi_thresh_per_ch:
        Fit noise threshold per channel
    filter_df (Row per peak):
        good_field_alignment
        good_field_quality
        good_noi
    """

    # Deprecated until further notice
    # returning all nones for now

    return RadFilterResult(
        params=params,
        field_df=None,
        field_align_thresh=None,
        per_peak_df=None,
        ch_peak_df=None,
        noi_thresh_per_ch=None,
        filter_df=None,
    )

    # field_df, field_align_thresh = rfilt.field_quality(
    #     ims_import, sigproc_v2, field_quality_thresh=params.field_quality_thresh
    # )
    #
    # per_peak_df, ch_peak_df = None, None
    # per_peak_df, ch_peak_df = rfilt.features(
    #     ims_import, sigproc_v2, params.dark_thresh_in_stds
    # )
    #
    # _, noi_thresh_per_ch = rfilt.noise(ch_peak_df, params.noi_thresh_in_stds)
    # noi_thresh = np.mean(noi_thresh_per_ch)
    #
    # filter_df = rfilt.build_filter_df(
    #     sigproc_v2, field_df, per_peak_df, ch_peak_df, noi_thresh
    # )
    #
    # return RadFilterResult(
    #     params=params,
    #     field_df=field_df,
    #     field_align_thresh=field_align_thresh,
    #     per_peak_df=per_peak_df,
    #     ch_peak_df=ch_peak_df,
    #     noi_thresh_per_ch=noi_thresh_per_ch,
    #     filter_df=filter_df,
    # )
