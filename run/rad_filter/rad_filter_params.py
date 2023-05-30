from munch import Munch

from plaster.tools.schema.schema import Params
from plaster.tools.schema.schema import Schema as s


class RadFilterParams(Params):
    defaults = Munch(
        field_quality_thresh=450.0,
        dark_thresh_in_stds=4.0,
        noi_thresh_in_stds=2.5,
    )

    schema = s(
        s.is_kws_r(
            field_quality_thresh=s.is_float(),
            dark_thresh_in_stds=s.is_float(),
            noi_thresh_in_stds=s.is_float(),
        )
    )
