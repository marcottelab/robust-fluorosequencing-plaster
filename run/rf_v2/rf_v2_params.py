from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin

from plaster.reports.helpers.report_params import FilterParams


@dataclass
class RFV2Params(DataClassJsonMixin):
    # See docs for this in FilterParams and ReportParams
    filter_params: Optional[FilterParams] = None
    filter_reject_thresh_all_cycles: Optional[bool] = False
