from dataclasses import dataclass, field
from typing import List

from dataclasses_json import DataClassJsonMixin

from plaster.tools.utils import utils

NO_FIELDS_LIMIT = -1
NO_CYCLES_LIMIT = -1


@dataclass
class ImsImportParams(utils.DataclassUnpickleFromMunchMixin, DataClassJsonMixin):
    """
    Warning: Flyte doesn't like Optional/Noneable ints currently (Apr 2022),
    so all int fields in this class with defaults must not be set to None.
    """

    # Note that in movie mode what is called "field" is really the "frame" since the
    # stage does not move between shots.
    # The single .nd2 file in movie mode then treats the "fields" as if they are "cycles"
    # of a single field.
    is_movie: bool = False
    start_field: int = 0
    n_fields_limit: int = NO_FIELDS_LIMIT
    start_cycle: int = 0
    n_cycles_limit: int = NO_CYCLES_LIMIT
    dst_ch_i_to_src_ch_i: List[int] = field(default_factory=list)
    is_z_stack_single_file: bool = False
    z_stack_n_slices_per_field: int = -1
