"""
These are files that are written by Angela's scope script.
That script is odd in several ways:
    * It is in utf-16
    * It incorrectly encodes tabs (\t) as backslash-t. ie "\\t" as a 2-character string!
    * Same with newlines: "\\n"
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import dataclasses_json
import plumbum
from dataclasses_json import DataClassJsonMixin
from plumbum import local

from plaster.tools.utils import utils

log = logging.getLogger(__name__)


@dataclass
class TSVAutoFocusData(DataClassJsonMixin):
    focus_area_start_pfs: float
    focus_area_x: float
    focus_area_y: float
    focus_area_z: float


@dataclass
class TSVChannelData(DataClassJsonMixin):
    channel_name: str = field(
        metadata=dataclasses_json.config(
            encoder=lambda v: v,
            # Sometimes channel name gets passed in as a float from the parser
            decoder=lambda v: str(v),
        )
    )
    setting_name: Optional[str] = None
    pfs_adjust: float = -1


@dataclass
class TSVData(DataClassJsonMixin):
    auto_focus: Optional[TSVAutoFocusData] = None
    bleach_exposure: float = -1
    bleach_focus_intv: float = -1
    bleach_wait: float = -1
    bleach_wash_intv: float = -1
    channel: List[TSVChannelData] = field(default_factory=list)
    chem_name: Optional[str] = None
    date: float = -1
    epoch_stamp: float = -1
    exp_notes: Optional[str] = None
    experiment_id: Optional[str] = None
    file_index_delta: float = -1
    first_file_index: float = -1
    focus_type: Optional[str] = None
    n_channels: float = -1
    n_col: float = -1
    n_cycles: float = -1
    n_edmans: float = -1
    n_fields: float = -1
    # One of the example tsv data I looked had this field, which looks like a mispelling.
    # I didn't see any instance of it or the correctly spelled field being used, so leaving this commented out for now.
    # n_foucs_areas: float = -1
    n_mocks: float = -1
    n_pres: float = -1
    n_rows: float = -1
    n_samples_in_experiment: float = -1
    n_space: float = -1
    quality_channel: float = -1
    quality_offset: float = -1
    sample_id: Optional[str] = None
    sample_index: float = -1
    scope_name: Optional[str] = None
    scripts: dict = field(default_factory=dict)
    start_x: float = -1
    start_y: float = -1
    start_z: float = -1


def parse_tsv(tsv):
    kvs = {}
    for line in tsv.split("\\n"):
        parts = line.split("\\t")
        if len(parts) == 2:
            # These are in the form:
            # ("part1/part2/part3", "some value")
            # The "channel" is special because it is a LIST so that
            # has to be treated separately.

            k, v = parts
            k = k.replace("/", ".")

            # If v can be converted into a float, great, otherwise leave it alone
            try:
                v = float(v)
            except ValueError:
                pass

            try:
                # Remember if "channel" was present before the exception
                had_channel = "channel" in kvs
                utils.block_update(kvs, k, v)
            except ValueError:
                # This is probably because the "channel" key has
                # not been yet been constructed; add "channel" and try again
                n_channels = int(kvs.get("n_channels", 0))
                if not had_channel and n_channels > 0:
                    kvs["channel"] = [dict() for c in range(n_channels)]

                # RETRY
                utils.block_update(kvs, k, v)
    return TSVData.from_dict(kvs)


def load_tsv(tsv_path):
    with open(tsv_path, "rb") as f:
        return parse_tsv(f.read().decode("utf-16"))


def load_tsv_for_folder(folder: plumbum.Path) -> TSVData:
    tsv_data = TSVData()
    tsv = sorted(list(local.path(folder) // "*.tsv"))
    if tsv is not None:
        if len(tsv) > 1:
            raise ValueError(f"Too many .tsv files were found in {folder}")
        if len(tsv) == 1:
            try:
                tsv_data = load_tsv(tsv[0])
            except FileNotFoundError:
                pass
            except Exception:
                log.error(f"File {tsv[0]} was not readable")
    return tsv_data
