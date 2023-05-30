import numpy as np
import pandas as pd

from plaster.run.base_result import BaseResult
from plaster.run.rad_filter.rad_filter_params import RadFilterParams


class RadFilterResult(BaseResult):
    name = "rad_filter"
    filename = "rad_filter.pkl"

    required_props = dict(
        params=RadFilterParams,
        field_df=(type(None), pd.DataFrame),
        field_align_thresh=(type(None), int),
        per_peak_df=(type(None), pd.DataFrame),
        ch_peak_df=(type(None), pd.DataFrame),
        noi_thresh_per_ch=(type(None), np.ndarray),
        filter_df=(type(None), pd.DataFrame),
    )

    def __repr__(self):
        return "RadFilterResult"
