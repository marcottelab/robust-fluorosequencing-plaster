from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin

from plaster.reports.helpers.report_params import FilterParams


@dataclass
class WhatprotV1Params(DataClassJsonMixin):
    # See docs for these in WhatprotConfig
    wp_numgenerate: int = (10000,)
    wp_neighbors: int = (10000,)
    wp_sigma: float = (0.5,)
    wp_passthrough: int = (1000,)
    wp_hmmprune: Optional[int] = 5
    wp_stoppingthreshold: float = 0.00001
    wp_maxruntime_minutes: int = 180
    wp_numbootstrap: int = 200
    wp_confidenceinterval: float = 0.9

    # See docs for this in FilterParams and ReportParams
    filter_params: Optional[FilterParams] = None

    filter_reject_thresh_all_cycles: Optional[bool] = False
