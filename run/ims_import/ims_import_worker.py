"""
Import image files

Options:

    * Nikon ND2 files.

        Each ND2 file is a collection of channels/field images per cycle.
        This organization needs to be transposed to effectively parallelize the
        sigproc stage (which acts on all cycles/channels of 1 field in parallel).

        Done in two stages:
            1. scatter the .nd2 files into individual .npy files
            2. gather those .nd2 files back into field stacks.

    * TIF files


Work needed Jul 2020

This module is responsible for accepting image input from
a variety of input formats and converting it into numpy
arrays in the shape and organization that are convenient for
downstream processing.

Definitions:
    * Frame / image
        An image taken by the microscope. This is at one wavelength,
        at one (X,Y,Z) position of the microscope's stage.
        Pixel's bit-depth varies depending on the camera used.
        Image dimensions are being fixed to the nearest power-of-two (upside or downside).
    * Channel
        A set of frames that correspond to a certain wavelength (ie
        filter setting) on the scope.
    * Cycle
        A set of frames (comprising all channels, all fields) that
        are taken after a certain "chemical cycle"
    * Field
        A field is all frames (all channels, cycles) that coorespond
        to a given (x, y, z) position of the microscope's stage.
    * Metadata
        Various metadata about the camera. Example: focus, brightess, etc.
        Not consistent on all input formats and scopes
    * "mea" or "measure" is a 1-dimensional measure
    * "dim" is a 2-d measure. If something is square then dim == (mea, mea)

Input formats:
    *.nd2
        This is a Nikon format the some of our older scopes use.
        It is not a well supported nor documented format but
        after some significant reverse-engineering I was able to
        get most of what I wanted out of the files. See nd2.py.

        ND2 files are usually, BUT NOT ALWAYS, written in the
        order that 1 nd2 file contains 1 cycle (all channels).

        The later steps of processing want the data in 1-file-per-field
        order so that each field can be processed in parallel.

    *.tif
        This is an even older use-case where some of the earliest
        experiments dumped large numbers of 16-but tif files with magic
        semantically significant filenames.  At least TIF is relatively
        well-supported and can use the skimage library.
        The tif files are sometimes spread out over a directory
        tree and require recursive traversal to find them all.

    *.npy
        This is the simplest format and will be what our production
        scopes will emit (to hopefully avoid a lot of the work
        in this module!)

    Input considerations:
        * The order of the input data / files is not typically in
          the most convenient order for processing.
        * The order of the input data is not always consistent and
          various modes have to accommodate this.
        * The input frames are not always in a power-of-two and
          have to be converted
        * A quality metric is useful to have and we might as well
          calculate it while we have the frames in memory

Output format:
    The output is in .npy format, all frames correctly scale to a power of two
    and organized by field.


Current approach:
    If you are in "movie mode" (which is an unhelpful name and needs to be
    revisited) then the .nd2 files are already in field-major order and
    therefore the task is simpler.

    If NOT in movie mode then a scatter/gather approach is taken:
        1. Scan file system and use some hints provided by the ims_import_params
           to decide which file-names/file-paths will be imported
        2. Scatter files by deconstructing ND2 or TIF files into individual
           frames (1 file per frame) and converting them to the nearest power-of-two
           if needed.
        3. Gather files by copying individual scattered files into the
           correct output order (field, channel, cycle).

Other notes
    * TODO Explain clamping and note that the input start/stop is not nec same as output
    * TODO Explain zap
    * As currently implemented even the scanning of the files is a bit
      slow as it opens and checks vitals on every file when it could
      do that progressively.
    * It might be faster to avoid the scatter stage and instead have
      each gather thread/process open the source files and scan them
      to find the desired frame.
    * The dimension conversions are painful
    * I use memory mapping to keep memory requiremetns down
    * Things are generally hard-coded to expect 16-bit files in places
      and that's okay as we do not expect other formats but it
      would be nice to be cleaner about it.
    * Some of the metadata is accompaniyed by TSV (tab separated value) files
      of an unusual format.


"""
import gc
import re
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import Dict, List, Tuple, Union

import numpy as np
from munch import Munch
from plumbum import Path, local
from skimage.io import imread

from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.run.ims_import.nd2 import ND2
from plaster.tools.image import imops
from plaster.tools.pipeline.pipeline import Progress
from plaster.tools.schema import check
from plaster.tools.tsv import tsv
from plaster.tools.utils import utils
from plaster.tools.zap import zap

OUTPUT_NP_TYPE = np.float32


log = getLogger(__name__)


def _scan_nd2_files(src_dir: Path) -> List[Path]:
    """Mock-point

    Returns a list of files in src_dir that have the suffix ".nd2"
    Note that this function is non-recursive, so .nd2 files in subfolders are not returned
    """
    return list(src_dir // "*.nd2")


def _scan_tif_files(src_dir: Path) -> List[Path]:
    """Mock-point

    Returns a list of files in src_dir and its subfolders that have the suffix ".tif"
    Note that this function is recursive, so .tif files in subfolders will be returned
    """
    return list(src_dir.walk(filter=lambda f: f.suffix == ".tif"))


def _scan_npy_files(src_dir: Path) -> List[Path]:
    """Mock-point

    Returns a list of files in src_dir and its subfolders that have the suffix ".npy"
    Note that this function is recursive, so .npy files in subfolders will be returned
    """
    return list(src_dir.walk(filter=lambda f: f.suffix == ".npy"))


def _nd2(src_path):
    """Mock-point"""
    return ND2(src_path)


def _load_npy(npy_path):
    """Mock-point"""
    return np.load(str(npy_path))


class ScanFileMode(Enum):
    npy = "npy"
    tif = "tif"
    nd2 = "nd2"


@dataclass
class ScanFilesResult:
    mode: ScanFileMode
    nd2_paths: List[Path]
    tif_paths_by_field_channel_cycle: Dict[Tuple[int, int, int], Path]
    npy_paths_by_field_channel_cycle: Dict[Tuple[int, int, int], Path]
    n_fields: int
    n_channels: int
    n_cycles: int
    dim: Tuple[int, int]


def _sort_nd2_files(files):
    """
    The script used on the Nikon scopes is not handling > 100 file names
    correctly and is generating a pattern like:
        ESN_2021_01_08_00_jsp116_00_P_009.nd2
        ESN_2021_01_08_00_jsp116_00_P_010.nd2
        ESN_2021_01_08_00_jsp116_00_P_0100.nd2
        ESN_2021_01_08_00_jsp116_00_P_011.nd2
    So this function parses the last number and treats it as an int for sorting
    """

    pat = re.compile(r"(.*_)(\d+)(\.nd2)$")
    file_splits = []
    did_split = None
    for file in files:
        g = pat.match(file)
        if g is not None:
            file_splits += [(g.group(1), g.group(2), g.group(3))]
            assert did_split is True or did_split is None
            did_split = True
        else:
            assert did_split is False or did_split is None
            did_split = False
    if did_split:
        numerically_sorted = sorted(file_splits, key=lambda x: int(x[1]))
        return ["".join(i) for i in numerically_sorted]
    else:
        return sorted(files)


def _sort_tif_files(files):
    return sorted(files)


def _sort_npy_files(files):
    return sorted(files)


def _scan_files(src_dir: Path) -> ScanFilesResult:
    """
    Search for .nd2 (non-recursive) or .tif files (recursively) or .npy (non-recursive)

    For .npy the he naming convention is:
        area, field, channel, cycle
        examples:
        area_000_cell_000_555nm_001.npy
        area_000_cell_000_647nm_001.npy
    """
    nd2_paths = _sort_nd2_files(_scan_nd2_files(src_dir))
    tif_paths = _sort_tif_files(_scan_tif_files(src_dir))
    npy_paths = _sort_npy_files(_scan_npy_files(src_dir))

    tif_paths_by_field_channel_cycle = {}
    npy_paths_by_field_channel_cycle = {}
    n_fields = 0
    n_channels = 0
    n_cycles = 0
    min_field = 10000
    min_channel = 10000
    min_cycle = 10000

    if len(nd2_paths) > 0:
        mode = ScanFileMode.nd2

        # OPEN a single image to get the vitals
        with _nd2(nd2_paths[0]) as nd2:
            n_fields = nd2.n_fields
            n_channels = nd2.n_channels
            dim = nd2.dim

    elif len(npy_paths) > 0:
        mode = ScanFileMode.npy

        # area_000_cell_000_555nm_001.npy
        npy_pat_v0 = re.compile(
            r"area_(?P<area>\d+)_cell_(?P<cell>\d+)_(?P<channel>\d+)nm_(?P<cycle>\d+)\.npy"
        )

        # {folder_name}_fl{self._field:04d}_ch{frame.channel:01d}_cy{cycle:03d}.npy
        npy_pat_v1 = re.compile(
            r"(?P<name>\w+)_fl(?P<field>\d+)_ch(?P<channel>\d+)_cy(?P<cycle>\d+)\.npy"
        )

        # All images must be of same name format. Check the first image to discover the format.
        # If we find that one of the files is not of that format we'll raise

        first_path = str(npy_paths[0])
        if npy_pat_v0.search(first_path):
            area_cells = set()
            channels = set()
            cycles = set()

            # PARSE the path names to determine channel, field, cycle
            for p in npy_paths:
                m = npy_pat_v0.search(str(p))
                if m:
                    found = Munch(m.groupdict())
                    area_cells.add((int(found.area), int(found.cell)))
                    channels.add(int(found.channel))
                    cycles.add(int(found.cycle))
                else:
                    raise ValueError(
                        f"npy file found ('{str(p)}') that did not match expected pattern."
                    )

            cycle_by_cycle_i = {
                cycle_i: cycle_name for cycle_i, cycle_name in enumerate(sorted(cycles))
            }
            n_cycles = len(cycle_by_cycle_i)

            channel_by_channel_i = {
                channel_i: channel_name
                for channel_i, channel_name in enumerate(sorted(channels))
            }
            n_channels = len(channel_by_channel_i)

            area_cell_by_field_i = {
                field_i: area_cell
                for field_i, area_cell in enumerate(sorted(area_cells))
            }
            n_fields = len(area_cell_by_field_i)

            for field_i in range(n_fields):
                area, cell = area_cell_by_field_i[field_i]
                for channel_i in range(n_channels):
                    channel = channel_by_channel_i[channel_i]
                    for cycle_i in range(n_cycles):
                        cycle = cycle_by_cycle_i[cycle_i]
                        npy_paths_by_field_channel_cycle[
                            (field_i, channel_i, cycle_i)
                        ] = (
                            local.path(src_dir)
                            / f"area_{area:03d}_cell_{cell:03d}_{channel:03d}nm_{cycle:03d}.npy"
                        )
        elif npy_pat_v1.search(first_path):
            fields = set()
            channels = set()
            cycles = set()
            for p in npy_paths:
                m = npy_pat_v1.search(str(p))
                if m:
                    found = Munch(m.groupdict())
                    field_i = int(found.field)
                    channel_i = int(found.channel)
                    cycle_i = int(found.cycle)

                    fields.add(field_i)
                    channels.add(channel_i)
                    cycles.add(cycle_i)

                    # TODO: why are we regenerating the filename for the npy_pat_v0?
                    npy_paths_by_field_channel_cycle[(field_i, channel_i, cycle_i)] = p
                else:
                    raise ValueError(
                        f"npy file found ('{str(p)}') that did not match expected pattern."
                    )

            n_fields = len(fields)
            n_channels = len(channels)
            n_cycles = len(cycles)

        # OPEN a single image to get the vitals
        im = _load_npy(str(npy_paths[0]))
        assert im.ndim == 2
        dim = im.shape

    elif len(tif_paths) > 0:
        mode = ScanFileMode.tif

        tif_pat = re.compile(
            r"_c(\d+)/img_channel(\d+)_position(\d+)_time\d+_z\d+\.tif"
        )

        # PARSE the path names to determine channel, field,
        for p in tif_paths:
            m = tif_pat.search(str(p))
            if m:
                cycle_i = int(m[1])
                channel_i = int(m[2])
                field_i = int(m[3])
                n_channels = max(channel_i, n_channels)
                n_cycles = max(cycle_i, n_cycles)
                n_fields = max(field_i, n_fields)
                min_field = min(field_i, min_field)
                min_channel = min(channel_i, min_channel)
                min_cycle = min(cycle_i, min_cycle)
                tif_paths_by_field_channel_cycle[(field_i, channel_i, cycle_i)] = p
            else:
                raise ValueError(
                    f"tif file found ('{str(p)}') that did not match expected pattern."
                )

        assert min_channel == 0
        n_channels += 1

        assert min_field == 0
        n_fields += 1

        if min_cycle == 0:
            n_cycles += 1
        elif min_cycle == 1:
            _tifs = {}
            for field_i in range(n_fields):
                for channel_i in range(n_channels):
                    for target_cycle_i in range(n_cycles):
                        _tifs[
                            (field_i, channel_i, target_cycle_i)
                        ] = tif_paths_by_field_channel_cycle[
                            (field_i, channel_i, target_cycle_i + 1)
                        ]
            tif_paths_by_field_channel_cycle = _tifs
        else:
            raise ValueError("tif cycle needs to start at 0 or 1")

        # OPEN a single image to get the vitals
        im = imread(str(tif_paths[0]))
        dim = im.shape
    else:
        raise ValueError(f"No image files (.nd2, .tif) were found in '{src_dir}'")

    return ScanFilesResult(
        mode=mode,
        nd2_paths=nd2_paths,
        tif_paths_by_field_channel_cycle=tif_paths_by_field_channel_cycle,
        npy_paths_by_field_channel_cycle=npy_paths_by_field_channel_cycle,
        n_fields=n_fields,
        n_channels=n_channels,
        n_cycles=n_cycles,
        dim=dim,
    )


def _npy_filename_by_field_channel_cycle(field, channel, cycle):
    return f"__{field:03d}-{channel:02d}-{cycle:02d}.npy"


def _metadata_filename_by_field_cycle(field, cycle):
    return f"__{field:03d}-{cycle:02d}.json"


def _adjust_im_dim(
    working_im: np.ndarray,
    orig_im: np.ndarray,
    actual_dim: Tuple[int, int],
    target_mea: int,
):
    """Copy data from original image to working image in the correct dimensions."""
    if actual_dim != (target_mea, target_mea):
        assert actual_dim[0] == actual_dim[1]
        if actual_dim[0] > target_mea:
            working_im[:, :] = orig_im[:target_mea, :target_mea]
        else:
            # CONVERT into a zero pad
            working_im[0 : actual_dim[0], 0 : actual_dim[1]] = orig_im
        return working_im

    # otherwise return the original image
    return orig_im


def _do_nd2_scatter(src_path, start_field, n_fields, cycle_i, n_channels, target_mea):
    """
    Scatter a cycle .nd2 into individual numpy files.

    target_mea is a scalar. The target will be put into this square form.
    """

    working_im = np.zeros((target_mea, target_mea), np.uint16)

    with _nd2(src_path) as nd2:
        dst_files = []
        for field_i in range(start_field, start_field + n_fields):
            # this is due to the new (12bit) version of Nikon microscope
            if hasattr(nd2, "camera_temp"):
                info = Munch(
                    x=nd2.x[field_i],
                    y=nd2.y[field_i],
                    z=nd2.z[field_i],
                    pfs_status=nd2.pfs_status[field_i],
                    pfs_offset=nd2.pfs_offset[field_i],
                    exposure_time=nd2.exposure_time[field_i],
                    camera_temp=nd2.camera_temp[field_i],
                    cycle_i=cycle_i,
                    field_i=field_i,
                )
            else:
                info = Munch(
                    x=nd2.x[field_i],
                    y=nd2.y[field_i],
                    z=nd2.z[field_i],
                    pfs_status=nd2.pfs_status[field_i],
                    pfs_offset=nd2.pfs_offset[field_i],
                    exposure_time=nd2.exposure_time[field_i],
                    # camera_temp=nd2.camera_temp[field_i],
                    cycle_i=cycle_i,
                    field_i=field_i,
                )
            info_dst_file = _metadata_filename_by_field_cycle(field_i, cycle_i)
            utils.json_save(info_dst_file, info)

            for channel_i in range(n_channels):
                im = nd2.get_field(field_i, channel_i)

                if im.shape[0] != target_mea or im.shape[1] != target_mea:
                    working_im[0 : im.shape[0], 0 : im.shape[1]] = im[
                        :target_mea, :target_mea
                    ]
                    im = working_im

                dst_file = _npy_filename_by_field_channel_cycle(
                    field_i, channel_i, cycle_i
                )
                dst_files += [dst_file]
                np.save(dst_file, im)
                # Need to make sure im gets garbage collected before we close the nd2 memmap
                im = None

        # Clean up numpy frombuffer before closing the memmap
        gc.collect()

    return dst_files


def _do_tif_scatter(field_i, channel_i, cycle_i, path):
    im = imread(str(path))
    dst_file = _npy_filename_by_field_channel_cycle(field_i, channel_i, cycle_i)
    np.save(dst_file, im)
    return dst_file


def _quality(im):
    """
    Quality of an image by spatial low-pass filter.
    High quality images are one where there is very little
    low-frequency (but above DC) bands.
    """
    return imops.low_frequency_power(im, dim_half=3)


def _do_gather(
    input_field_i: int,
    output_field_i: int,
    start_cycle: int,
    n_cycles: int,
    dim: int,
    import_result: ImsImportResult,
    mode: ScanFileMode,
    npy_paths_by_field_channel_cycle: dict,
    dst_ch_i_to_src_ch_i: List[int],
):
    """Gather a field"""
    n_dst_channels = len(dst_ch_i_to_src_ch_i)

    field_chcy_arr = import_result.allocate_field(
        output_field_i, (n_dst_channels, n_cycles, dim, dim), OUTPUT_NP_TYPE
    )
    field_chcy_ims = field_chcy_arr.arr()

    chcy_i_to_quality = np.zeros((n_dst_channels, n_cycles))
    cy_i_to_metadata = [None] * n_cycles

    output_cycle_i = 0
    for input_cycle_i in range(start_cycle, start_cycle + n_cycles):
        # GATHER channels

        for dst_ch_i in range(n_dst_channels):
            src_ch_i = dst_ch_i_to_src_ch_i[dst_ch_i]

            if mode == ScanFileMode.npy:
                # These are being imported by npy originally with a different naming
                # convention than the scattered files.
                scatter_fp = npy_paths_by_field_channel_cycle[
                    (input_field_i, src_ch_i, input_cycle_i)
                ]
            else:
                scatter_fp = _npy_filename_by_field_channel_cycle(
                    input_field_i, src_ch_i, input_cycle_i
                )

            im = _load_npy(scatter_fp)
            if im.dtype != OUTPUT_NP_TYPE:
                im = im.astype(OUTPUT_NP_TYPE)
            field_chcy_ims[dst_ch_i, output_cycle_i, :, :] = im
            chcy_i_to_quality[dst_ch_i, output_cycle_i] = _quality(im)

        # GATHER metadata files if any
        cy_i_to_metadata[output_cycle_i] = None
        try:
            cy_i_to_metadata[output_cycle_i] = utils.json_load_munch(
                _metadata_filename_by_field_cycle(input_field_i, input_cycle_i)
            )
        except FileNotFoundError:
            pass

        output_cycle_i += 1

    import_result.save_field(
        field_i=output_field_i,
        field_chcy_ims=field_chcy_arr,
        metadata_by_cycle=cy_i_to_metadata,
        chcy_qualities=chcy_i_to_quality,
    )

    return output_field_i


def _do_movie_import_nd2(
    scan_result,
    input_field_i,
    output_field_i,
    start_cycle,
    n_cycles,
    target_mea,
    import_result,
    dst_ch_i_to_src_ch_i,
):
    """
    Import Nikon ND2 "movie" files.

    In this mode, each .nd2 file is a collection of images taken sequentially for a single field.
    This is in contrast to the typical mode where each .nd2 file is a chemical cycle spanning
    all fields/channels.

    Since all data for a given field is already in a single file, the parallel
    scatter/gather employed by the "normal" ND2 import task is not necessary.

    The "fields" from the .nd2 file become "cycles" as if the instrument had
    taken 1 field with a lot of cycles.
    """
    working_im = np.zeros((target_mea, target_mea), OUTPUT_NP_TYPE)

    nd2_path = scan_result.nd2_paths[input_field_i]
    with _nd2(nd2_path) as nd2:
        n_actual_cycles = nd2.n_fields
        n_dst_channels = len(dst_ch_i_to_src_ch_i)
        actual_dim = nd2.dim

        chcy_arr = import_result.allocate_field(
            output_field_i,
            (n_dst_channels, n_cycles, target_mea, target_mea),
            OUTPUT_NP_TYPE,
        )
        chcy_ims = chcy_arr.arr()

        assert start_cycle + n_cycles <= n_actual_cycles
        check.affirm(
            actual_dim[0] <= target_mea and actual_dim[1] <= target_mea,
            f"nd2 scatter requested {target_mea} which is smaller than {actual_dim}",
        )

        for dst_ch_i in range(n_dst_channels):
            src_ch_i = dst_ch_i_to_src_ch_i[dst_ch_i]
            for cy_in_i in range(start_cycle, start_cycle + n_cycles):
                cy_out_i = cy_in_i - start_cycle

                im = nd2.get_field(cy_in_i, src_ch_i).astype(OUTPUT_NP_TYPE)
                im = _adjust_im_dim(working_im, im, actual_dim, target_mea)

                chcy_ims[dst_ch_i, cy_out_i, :, :] = im

        # Task: Add quality
        import_result.save_field(output_field_i, chcy_arr)

    return output_field_i, n_actual_cycles


def _do_movie_import_npy(
    scan_result,
    input_field_i,
    output_field_i,
    start_cycle,
    n_cycles,
    target_mea,
    import_result,
    dst_ch_i_to_src_ch_i,
):
    """
    In this mode, each field is a collection of images taken sequentially without moving stage.
    """
    n_dst_channels = len(dst_ch_i_to_src_ch_i)
    actual_dim = scan_result.dim

    working_im = np.zeros((target_mea, target_mea), OUTPUT_NP_TYPE)

    chcy_arr = import_result.allocate_field(
        output_field_i,
        (n_dst_channels, n_cycles, target_mea, target_mea),
        OUTPUT_NP_TYPE,
    )
    chcy_ims = chcy_arr.arr()

    assert start_cycle + n_cycles <= scan_result.n_cycles
    check.affirm(
        actual_dim[0] <= target_mea and actual_dim[1] <= target_mea,
        f"npy requested {target_mea} which is smaller than {actual_dim}",
    )

    for dst_ch_i in range(n_dst_channels):
        src_ch_i = dst_ch_i_to_src_ch_i[dst_ch_i]
        for cy_in_i in range(start_cycle, start_cycle + n_cycles):
            cy_out_i = cy_in_i - start_cycle

            im_path = scan_result.npy_paths_by_field_channel_cycle[
                input_field_i, src_ch_i, cy_in_i
            ]
            im = np.load(str(im_path)).astype(OUTPUT_NP_TYPE)
            assert im.shape == actual_dim
            im = _adjust_im_dim(working_im, im, actual_dim, target_mea)

            chcy_ims[dst_ch_i, cy_out_i, :, :] = im

        # Task: Add quality
        import_result.save_field(output_field_i, chcy_arr)

    return output_field_i, n_cycles


def _z_stack_import(
    nd2_path: Path,
    target_mea: int,
    import_result: ImsImportResult,
    dst_ch_i_to_src_ch_i: List[int],
    movie_n_slices_per_field,
):
    """
    A single ND2 file with multiple fields
    """
    working_im = np.zeros((target_mea, target_mea), OUTPUT_NP_TYPE)

    with _nd2(nd2_path) as nd2:
        n_actual_cycles = nd2.n_fields
        n_dst_channels = len(dst_ch_i_to_src_ch_i)
        actual_dim = nd2.dim

        assert n_actual_cycles % movie_n_slices_per_field == 0
        n_fields = n_actual_cycles // movie_n_slices_per_field

        for field_i in range(n_fields):
            chcy_arr = import_result.allocate_field(
                field_i,
                (n_dst_channels, movie_n_slices_per_field, target_mea, target_mea),
                OUTPUT_NP_TYPE,
            )
            chcy_ims = chcy_arr.arr()

            check.affirm(
                actual_dim[0] <= target_mea and actual_dim[1] <= target_mea,
                f"nd2 scatter requested {target_mea} which is smaller than {actual_dim}",
            )

            for dst_ch_i in range(n_dst_channels):
                src_ch_i = dst_ch_i_to_src_ch_i[dst_ch_i]
                for cy_out_i, cy_in_i in enumerate(
                    range(
                        field_i * movie_n_slices_per_field,
                        (field_i + 1) * movie_n_slices_per_field,
                    )
                ):
                    im = nd2.get_field(cy_in_i, src_ch_i).astype(OUTPUT_NP_TYPE)
                    im = _adjust_im_dim(working_im, im, actual_dim, target_mea)

                    chcy_ims[dst_ch_i, cy_out_i, :, :] = im

            # Task: Add quality
            import_result.save_field(field_i, chcy_arr)

    return list(range(n_fields)), movie_n_slices_per_field


def ims_import(
    src_dir: Path,
    ims_import_params: ImsImportParams,
    result_class=ImsImportResult,
    folder: str = None,
):
    reference_nd2_file_for_metadata = None

    with Progress("scan files") as progress:
        scan_result = _scan_files(src_dir)

    if len(scan_result.nd2_paths) > 0:
        reference_nd2_file_for_metadata = scan_result.nd2_paths[0]

    max_dim = max(scan_result.dim)
    target_mea = utils.normalize_to_square(max_dim)
    if target_mea != max_dim:
        log.warning(
            f"Normalizing image dimensions: original={scan_result.dim[0]}x{scan_result.dim[1]}px, "
            f"target={target_mea}x{target_mea}px"
        )

    def clamp_fields(n_fields_true: int) -> Tuple[int, int]:
        n_fields = n_fields_true
        n_fields_limit = ims_import_params.n_fields_limit
        if n_fields_limit > 0:
            n_fields = n_fields_limit

        start_field = ims_import_params.start_field
        if start_field + n_fields > n_fields_true:
            n_fields = n_fields_true - start_field

        return start_field, n_fields

    def clamp_cycles(n_cycles_true: int) -> Tuple[int, int]:
        n_cycles = n_cycles_true
        n_cycles_limit = ims_import_params.n_cycles_limit
        if n_cycles_limit > 0:
            n_cycles = n_cycles_limit

        start_cycle = ims_import_params.start_cycle
        if start_cycle + n_cycles > n_cycles_true:
            n_cycles = n_cycles_true - start_cycle

        return start_cycle, n_cycles

    tsv_data = tsv.load_tsv_for_folder(src_dir)

    # ALLOCATE the ImsImportResult
    ims_import_result = result_class(params=ims_import_params, tsv_data=tsv_data)

    if folder is not None:
        ims_import_result.set_folder(folder)

    dst_ch_i_to_src_ch_i = ims_import_params.dst_ch_i_to_src_ch_i
    if not dst_ch_i_to_src_ch_i:
        dst_ch_i_to_src_ch_i = [i for i in range(scan_result.n_channels)]

    n_out_channels = len(dst_ch_i_to_src_ch_i)

    # Sanity check that we didn't end up with any src_channels outside of the channel range
    if not all(
        [0 <= src_ch_i < scan_result.n_channels for src_ch_i in dst_ch_i_to_src_ch_i]
    ):
        raise ValueError(
            f"channel out of bounds. dst_ch_i_to_src_ch_i={dst_ch_i_to_src_ch_i} found_n_channels={scan_result.n_channels}"
        )

    if ims_import_params.is_z_stack_single_file:
        with Progress("import z stack") as progress:
            field_iz, n_cycles_found = _z_stack_import(
                scan_result.nd2_paths[0],
                target_mea,
                ims_import_result,
                dst_ch_i_to_src_ch_i,
                ims_import_params.z_stack_n_slices_per_field,
            )
            n_cycles = ims_import_params.z_stack_n_slices_per_field

    elif ims_import_params.is_movie:
        with Progress("import movie") as progress:
            if scan_result.mode == ScanFileMode.nd2:
                # "Movie mode" means that there aren't any chemical cycles, but rather we are using "cycles" to represent different images in a zstack
                start_field, n_fields = clamp_fields(len(scan_result.nd2_paths))

                # In movie mode, the n_fields from the .nd2 file is becoming n_cycles
                scan_result.n_cycles = scan_result.n_fields
                start_cycle, n_cycles = clamp_cycles(scan_result.n_cycles)

                with zap.Context(progress=progress):
                    field_iz, n_cycles_found = zap.arrays(
                        _do_movie_import_nd2,
                        dict(
                            input_field_i=list(
                                range(start_field, start_field + n_fields)
                            ),
                            output_field_i=list(range(n_fields)),
                        ),
                        _stack=True,
                        scan_result=scan_result,
                        start_cycle=start_cycle,
                        n_cycles=n_cycles,
                        target_mea=target_mea,
                        import_result=ims_import_result,
                        dst_ch_i_to_src_ch_i=dst_ch_i_to_src_ch_i,
                    )
            elif scan_result.mode == ScanFileMode.npy:
                start_field, n_fields = clamp_fields(scan_result.n_fields)
                start_cycle, n_cycles = clamp_cycles(scan_result.n_cycles)

                with zap.Context(progress=progress):
                    field_iz, n_cycles_found = zap.arrays(
                        _do_movie_import_npy,
                        dict(
                            input_field_i=list(
                                range(start_field, start_field + n_fields)
                            ),
                            output_field_i=list(range(n_fields)),
                        ),
                        _stack=True,
                        scan_result=scan_result,
                        start_cycle=start_cycle,
                        n_cycles=n_cycles,
                        target_mea=target_mea,
                        import_result=ims_import_result,
                        dst_ch_i_to_src_ch_i=dst_ch_i_to_src_ch_i,
                    )
            else:
                raise NotImplementedError()

    else:
        start_field, n_fields = clamp_fields(scan_result.n_fields)

        with Progress("scatter files") as progress:

            if scan_result.mode == ScanFileMode.nd2:
                scan_result.n_cycles = len(scan_result.nd2_paths)

                # SCATTER
                with zap.Context(mode="thread", progress=progress):
                    zap.arrays(
                        _do_nd2_scatter,
                        dict(
                            cycle_i=list(range(len(scan_result.nd2_paths))),
                            src_path=scan_result.nd2_paths,
                        ),
                        _stack=True,
                        start_field=start_field,
                        n_fields=n_fields,
                        n_channels=scan_result.n_channels,
                        target_mea=target_mea,
                    )

            elif scan_result.mode == ScanFileMode.tif:
                # SCATTER
                work_orders = [
                    Munch(field_i=k[0], channel_i=k[1], cycle_i=k[2], path=path)
                    for k, path in scan_result.tif_paths_by_field_channel_cycle.items()
                ]
                with zap.Context(trap_exceptions=False):
                    results = zap.work_orders(_do_tif_scatter, work_orders)

                # CHECK that every file exists
                for f in range(n_fields):
                    for ch in range(scan_result.n_channels):
                        for cy in range(scan_result.n_cycles):
                            expected = f"__{f:03d}-{ch:02d}-{cy:02d}.npy"
                            if expected not in results:
                                raise FileNotFoundError(
                                    f"File is missing in tif pattern: {expected}"
                                )

            elif scan_result.mode == ScanFileMode.npy:
                # In npy mode there's no scatter as the files are already fully scattered
                pass

            else:
                raise ValueError(f"Unknown im import mode {scan_result.mode}")

        with Progress("gather files") as progress:

            # GATHER
            start_cycle, n_cycles = clamp_cycles(scan_result.n_cycles)

            with zap.Context(progress=progress):
                field_iz = zap.arrays(
                    _do_gather,
                    dict(
                        input_field_i=list(range(start_field, start_field + n_fields)),
                        output_field_i=list(range(0, n_fields)),
                    ),
                    _stack=True,
                    start_cycle=start_cycle,
                    n_cycles=n_cycles,
                    dim=target_mea,
                    import_result=ims_import_result,
                    mode=scan_result.mode,
                    npy_paths_by_field_channel_cycle=scan_result.npy_paths_by_field_channel_cycle,
                    dst_ch_i_to_src_ch_i=dst_ch_i_to_src_ch_i,
                )

    if reference_nd2_file_for_metadata:
        with _nd2(reference_nd2_file_for_metadata) as nd2:
            if hasattr(nd2, "metadata"):
                full = Munch(
                    metadata=nd2.metadata,
                    metadata_seq=nd2.metadata_seq,
                )
                ims_import_result._nd2_metadata_full = full

                def me(block_name, default=None):
                    return utils.block_search(
                        full.metadata.SLxExperiment, block_name, default
                    )

                def mp(block_name, default=None):
                    return utils.block_search(
                        full.metadata_seq.SLxPictureMetadata, block_name, default
                    )

                n_channels = mp("sPicturePlanes.uiSampleCount", 1)

                ims_import_result._nd2_metadata = Munch(
                    calibrated_pixel_size=mp("dCalibration"),
                    experiment_type="movie" if me("eType") == 1 else "edman",
                    n_cycles=me("uLoopPars.uiCount"),
                    cmd_before=me("wsCommandBeforeCapture"),
                    cmd_after=me("wsCommandAfterCapture"),
                    n_channels=n_channels,
                )

                per_channel = []
                for ch_i in range(n_channels):
                    laser_wavelength = None
                    laser_power = None
                    n_lasers = mp(
                        f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_uiMultiLaserLines0",
                        0,
                    )
                    for i in range(n_lasers):
                        is_used = mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_bMultiLaserLineUsed0-{i:02d}",
                            0,
                        )
                        if is_used == 1:
                            laser_wavelength = mp(
                                f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_uiMultiLaserLineWavelength0-{i:02d}",
                                0,
                            )
                            laser_power = mp(
                                f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_dMultiLaserLinePower0-{i:02d}",
                                0,
                            )

                    ch_munch = Munch(
                        laser_wavelength=laser_wavelength,
                        laser_power=laser_power,
                        camera_name=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.CameraUniqueName"
                        ),
                        sensor_pixels_x=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.fmtDesc.sizeSensorPixels.cx"
                        ),
                        sensor_pixels_y=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.fmtDesc.sizeSensorPixels.cy"
                        ),
                        sensor_microns_x=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.fmtDesc.sizeSensorMicrons.cx"
                        ),
                        sensor_microns_y=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.fmtDesc.sizeSensorMicrons.cy"
                        ),
                        bin_x=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.fmtDesc.dBinningX"
                        ),
                        bin_y=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.fmtDesc.dBinningY"
                        ),
                        format=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.fmtDesc.wszFormatDesc"
                        ),
                        roi_l=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.rectSensorUser.left"
                        ),
                        roi_r=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.rectSensorUser.right"
                        ),
                        roi_t=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.rectSensorUser.top"
                        ),
                        roi_b=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.FormatQuality.rectSensorUser.bottom"
                        ),
                        averaging=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.PropertiesQuality.Average"
                        ),
                        integration=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.PropertiesQuality.Integrate"
                        ),
                        name=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pCameraSetting.Metadata.Channels.Channel_0.Name"
                        ),
                        dichroic_filter=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_sFilterName0"
                        ),
                        emission_filter=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_sFilterName1"
                        ),
                        optivar=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_dZoomPosition"
                        ),
                        tirf_focus=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_dTIRFPositionFocus"
                        ),
                        tirf_align_x=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_dTIRFPositionX"
                        ),
                        tirf_align_y=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pDeviceSetting.m_dTIRFPositionY"
                        ),
                        objective_mag=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pObjectiveSetting.dObjectiveMag"
                        ),
                        objective_na=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pObjectiveSetting.dObjectiveNA"
                        ),
                        objective_refractive_index=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.pObjectiveSetting.dRefractIndex"
                        ),
                        settings_name=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.sOpticalConfigs.\x02.sOpticalConfigName"
                        ),
                        readout_mode=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.sSpecSettings.Readout Mode"
                        ),
                        readout_rate=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.sSpecSettings.Readout Rate"
                        ),
                        noise_filter=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.sSpecSettings.Noise Filter"
                        ),
                        temperature=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.sSpecSettings.Temperature"
                        ),
                        exposure=mp(
                            f"sPicturePlanes.sSampleSetting.a{ch_i}.dExposureTime"
                        ),
                    )
                    per_channel += [ch_munch]

                ims_import_result._nd2_metadata.update(**Munch(per_channel=per_channel))

                if me("eType") == 1:
                    # Movie mode
                    ims_import_result._nd2_metadata.update(
                        **Munch(
                            movie_start=me("dStart"),
                            movie_period=me("dPeriod"),
                            movie_duration=me("dDuration"),
                            movie_duration_pref=me("bDurationPref"),
                            movie_max_period_diff=me("dMaxPeriodDiff"),
                            movie_min_period_diff=me("dMinPeriodDiff"),
                            movie_avg_period_diff=me("dAvgPeriodDiff"),
                        )
                    )

    ims_import_result.n_fields = len(field_iz)
    ims_import_result.n_channels = n_out_channels
    ims_import_result.n_cycles = n_cycles
    ims_import_result.dim = target_mea
    ims_import_result.dtype = np.dtype(OUTPUT_NP_TYPE).name
    ims_import_result.src_dir = src_dir

    # CLEAN
    for file in local.cwd // "__*":
        file.delete()

    return ims_import_result


# Experiments in reverse engineering ND2 metadata
# if __name__ == "__main__":
# r = ims_import(local.path("./jobs_folder/metadata/"), ImsImportParams())
# r.save()

# r = ImsImportResult.load_from_folder(".")

# from plaster.run.ims_import.nd2 import _ND2
# with open("./jobs_folder/metadata/ESN_2020_09_17_00_jsp092_60minpitc_00_P_001.nd2",  "rb") as f:
#     nd2 = _ND2(f.read())
#     print(json.dumps(nd2.metadata, indent=4))
#     print()
#     print(json.dumps(nd2.metadata_seq, indent=4))

# Interesting:
#  nd2.metadata.SLxExperiment.wsCameraName
#  nd2.metadata_seq.sPicturePlanes.sPlaneNew.a0.sizeObjFullChip.cx
#  nd2.metadata_seq.sPicturePlanes.sPlaneNew.a0.sizeObjFullChip.cy
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.CameraUniqueName
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.sizeSensorPixels.cx: 2048
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.sizeSensorPixels.cy: 2044
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.sizeSensorMicrons.cx: 13312
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.sizeSensorMicrons.cy: 13286
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.sizeSensorMin.cx: 4
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.sizeSensorMin.cy: 4
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.sizeSensorStep.cx: 2
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.sizeSensorStep.cy: 2
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.dBinningX: 1.0
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.dBinningY: 1.0
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.FormatFast.fmtDesc.wszFormatDesc: "16-bit - No Binning (100.0 FPS)"
#  nd2.metadata_seq.sPicturePlanes.sSampleSetting.a0.pCameraSetting.PropertiesQuality.Exposure: 1000.0
