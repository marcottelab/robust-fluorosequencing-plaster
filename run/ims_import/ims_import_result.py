from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import flytekit
import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin
from munch import Munch

from plaster.run.base_result import ArrayResult
from plaster.run.base_result_dc import (
    BaseResultDC,
    LazyField,
    generate_flyte_result_class,
    lazy_field,
)
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.tools.tsv import tsv
from plaster.tools.utils import utils
from plaster.tools.utils.fancy_indexer import FancyIndexer


@dataclass
class ImsImportResult(BaseResultDC, DataClassJsonMixin):
    """
    The image importer results in a large number of image files, one per field.

    The normal result manifest contains a list of those filenames
    and a helper function lazy loads them.

    Warning: Flyte doesn't like Optional/Noneable ints currently (Apr 2022),
    so all int fields in this class with defaults must not be set to None.
    """

    n_fields: int = 0
    n_channels: int = 0
    n_cycles: int = 0
    dim: int = 0
    params: Optional[ImsImportParams] = None
    dtype: Optional[str] = None
    src_dir: Optional[str] = None
    tsv_data: Optional[tsv.TSVData] = None
    is_loaded_result: bool = False
    _metadata_df: LazyField = lazy_field(pd.DataFrame)
    _qualities_df: LazyField = lazy_field(pd.DataFrame)

    _metadata_columns = (
        "x",
        "y",
        "z",
        "pfs_status",
        "pfs_offset",
        "exposure_time",
        "camera_temp",
        "field_i",
        "cycle_i",
    )

    def _field_ims_filename(self, field_i: int):
        return str(self._folder / f"field_{field_i:03d}.npy")

    def _field_metadata_filename(self, field_i: int):
        return str(self._folder / f"field_{field_i:03d}_metadata.pkl")

    def _field_qualities_filename(self, field_i: int):
        return str(self._folder / f"field_{field_i:03d}_qualities.pkl")

    def allocate_field(self, field_i: int, shape: tuple, dtype: np.dtype):
        filename = self._field_ims_filename(field_i)
        return ArrayResult(filename, dtype, shape, mode="w+")

    def save_field(
        self,
        field_i: int,
        field_chcy_ims: ArrayResult,
        metadata_by_cycle: List[Any] = None,
        chcy_qualities: np.ndarray = None,
    ):
        """
        When using parallel field maps we can not save into the result
        because that will not be serialized back to the main thread.
        Rather, all field oriented results are written to a
        temporary pickle file and are reduced to a single value
        in the main thread's result instance.
        """
        field_chcy_ims.flush()

        if metadata_by_cycle is not None:
            utils.pickle_save(self._field_metadata_filename(field_i), metadata_by_cycle)

        if chcy_qualities is not None:
            utils.pickle_save(self._field_qualities_filename(field_i), chcy_qualities)

    def save(self):
        """
        Gather metadata that was written into temp files
        into one metadata_df and remove those files.
        """
        files = sorted(self.folder.glob("field_*_metadata.pkl"))
        rows = [
            cycle_metadata
            for file in files
            for cycle_metadata in utils.pickle_load(file)
        ]

        for file in files:
            file.unlink()

        self._metadata_df.set(
            pd.DataFrame(rows).rename(
                columns=dict(x="stage_x", y="stage_y", z="stage_z")
            )
        )

        files = sorted(self.folder.glob("field_*_qualities.pkl"))
        rows = []
        for field_i, file in enumerate(files):
            chcy_qualities = utils.pickle_load(file)
            for ch in range(self.n_channels):
                for cy in range(self.n_cycles):
                    rows += [(field_i, ch, cy, chcy_qualities[ch, cy])]

        for file in files:
            file.unlink()

        self._qualities_df.set(
            pd.DataFrame(rows, columns=("field_i", "channel_i", "cycle_i", "quality"))
        )

    def field_chcy_ims(self, field_i: int):
        if field_i not in self._cache_field_chcy_ims:
            self._cache_field_chcy_ims[field_i] = ArrayResult(
                self._field_ims_filename(field_i),
                dtype=np.dtype(self.dtype),
                shape=(self.n_channels, self.n_cycles, self.dim, self.dim),
            )
        return self._cache_field_chcy_ims[field_i].arr()

    def n_fields_channel_cycles(self):
        assert self.params.is_movie is False
        return self.n_fields, self.n_channels, self.n_cycles

    def n_fields_channel_frames(self):
        # assert self.params.is_movie is True
        return self.n_fields, self.n_channels, self.n_cycles

    @property
    def ims(self):
        """Return a fancy-indexer that can return slices from [fields, channels, cycles]"""
        return FancyIndexer(
            (self.n_fields, self.n_channels, self.n_cycles),
            lookup_fn=lambda fl, ch, cy: self.field_chcy_ims(fl)[ch, cy],
        )

    def metadata(self):
        return self._metadata_df.get()

    def has_metadata(self):
        return (
            hasattr(self, "_metadata_df")
            and len(self.metadata()) > 0
            and "field_i" in self.metadata()
        )

    def qualities(self):
        return self._qualities_df.get()

    def __post_init__(self):
        super().__post_init__()
        self._cache_field_chcy_ims = {}

        if self._folder is None:
            self.set_folder(
                Path(flytekit.current_context().working_directory) / "ims_import"
            )
            self.folder.mkdir(exist_ok=True, parents=True)

    def __repr__(self):
        try:
            return f"ImsImportResult with {self.n_fields} fields."
        except:
            return "ImsImportResult"


ImsImportFlyteResult = generate_flyte_result_class(ImsImportResult)
