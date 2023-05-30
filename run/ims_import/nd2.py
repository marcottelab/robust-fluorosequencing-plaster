"""
A hand-crafted ND2 reader that is much faster and more stable than others
that I found online but doesn't support all the crazy modes that ND2 files
support. Only supports 16-bit forms.

Helpful reference:
https://github.com/openmicroscopy/bioformats/blob/develop/components/formats-gpl/src/loci/formats/in/NativeND2Reader.java

* Each ND2 file contains multiple fields over channels in ONE cycle.
"""

import mmap
import struct
from contextlib import contextmanager
from typing import Callable

import numpy as np
from munch import Munch


class _ND2:
    data_types = {
        1: "unsigned_char",
        2: "unsigned_int",
        3: "unsigned_int_2",
        5: "unsigned_long",
        6: "double",
        8: "string",
        9: "char_array",
        11: "dict",
    }

    # this number describes a "large enough" number to catch the signature at the end of the file
    LAST_SECTION_IN_FILE_LEN = 10_000

    BLOCK_NAMES = Munch(
        IM_ATTR_LV="ImageAttributesLV",
        IM_DATA_SEQ="ImageDataSeq|",
        CUST_DATA_X="CustomData|X",
        CUST_DATA_Y="CustomData|Y",
        CUST_DATA_Z="CustomData|Z",
        CUST_DATA_PFS_STAT="CustomData|PFS_STATUS",
        CUST_DATA_PFS_OFF="CustomData|PFS_OFFSET",
        CUST_DATA_CAM_EXP_TIME="CustomData|Camera_ExposureTime1",
        CUST_DATA_CAM_TEMP="CustomData|CameraTemp1",
        IM_METADATA_SEQ_LV="ImageMetadataSeqLV|0",
        IM_METADATA_LV="ImageMetadataLV",
    )

    def __init__(self, data: mmap.mmap):
        # initailize data with mmap object
        self.data = data
        # The CHUNK MAP SIGNATURE is generally near the end of the file
        end_block_len = len(data) - self.LAST_SECTION_IN_FILE_LEN
        tail_data = data[end_block_len:]
        sig = bytes("ND2 CHUNK MAP SIGNATURE 0000001!", encoding="utf-8")
        sig_offset_rel_to_end = tail_data.find(sig)
        if sig_offset_rel_to_end < 0:
            raise ValueError("nd2 file did not find a signature")

        # signature's absolute offset (from the beginning of the file)
        sig_offset_abs = end_block_len + sig_offset_rel_to_end
        chunk_map_pos = self._u64(sig_offset_abs + len(sig))

        tmp_len_two = self._u32(chunk_map_pos + 4)
        chunk_map_length = self._u64(chunk_map_pos + 8)
        chunk_map_pos += 16 + tmp_len_two
        chunk_map_end = chunk_map_pos + chunk_map_length

        name_terminator = bytes("!", encoding="utf-8")
        off = chunk_map_pos

        for block in self._parse_nd2(off, chunk_map_end, name_terminator, sig):
            if block.name.startswith(self.BLOCK_NAMES.IM_ATTR_LV):
                d = self._metadata_section(block)
                self.n_channels = d.SLxImageAttributes.uiVirtualComponents
                # assert d.SLxImageAttributes.uiSequenceCount == len(self.images.keys())
                self.n_fields = d.SLxImageAttributes.uiSequenceCount
                self.dim = (d.SLxImageAttributes.uiHeight, d.SLxImageAttributes.uiWidth)
                # assert d.SLxImageAttributes.uiBpcSignificant == 16
                assert d.SLxImageAttributes.uiBpcInMemory == 16
            elif block.name == self.BLOCK_NAMES.CUST_DATA_X:
                self.x = self._data_block(block, self._f64, 8)
            elif block.name == self.BLOCK_NAMES.CUST_DATA_Y:
                self.y = self._data_block(block, self._f64, 8)
            elif block.name == self.BLOCK_NAMES.CUST_DATA_Z:
                self.z = self._data_block(block, self._f64, 8)
            elif block.name == self.BLOCK_NAMES.CUST_DATA_PFS_STAT:
                self.pfs_status = self._data_block(block, self._u32, 4)
            elif block.name == self.BLOCK_NAMES.CUST_DATA_PFS_OFF:
                self.pfs_offset = self._data_block(block, self._u32, 4)
            elif block.name == self.BLOCK_NAMES.CUST_DATA_CAM_EXP_TIME:
                self.exposure_time = self._data_block(block, self._f64, 8)
            elif block.name == self.BLOCK_NAMES.CUST_DATA_CAM_TEMP:
                self.camera_temp = self._data_block(block, self._f64, 8)
            elif block.name == self.BLOCK_NAMES.IM_METADATA_SEQ_LV:
                self.metadata_seq = self._metadata_section(block)
                # There's some extra weird info encoded in strings in:
                # SLxPictureMetadata.sPicturePlanes.sSampleSetting.a0.sSpecSettings
                # Where "a0" refers to channel 0.

                n_channels = (
                    self.metadata_seq.SLxPictureMetadata.sPicturePlanes.uiSampleCount
                )

                for ch_i in range(n_channels):
                    extra = self.metadata_seq.SLxPictureMetadata.sPicturePlanes.sSampleSetting[
                        f"a{ch_i:01d}"
                    ].sSpecSettings
                    extra = extra.split("\r\n")
                    extra_kvs = Munch()
                    for e in extra:
                        parts = e.split(":")
                        if len(parts) == 2:
                            k, v = parts
                            v = v.strip()
                            extra_kvs[k.strip()] = v.strip()

                    self.metadata_seq.SLxPictureMetadata.sPicturePlanes.sSampleSetting[
                        f"a{ch_i:01d}"
                    ].sSpecSettings = extra_kvs
            elif block.name == self.BLOCK_NAMES.IM_METADATA_LV:
                self.metadata = self._metadata_section(block)

    def _parse_nd2(
        self, off: int, chunk_map_end: int, name_terminator: bytes, sig: bytes
    ):
        """Blocks generator for nd2 file."""
        self.images = {}

        while off < chunk_map_end:
            term = self.data[off:].find(name_terminator)
            name = self.data[off : off + term + 1]
            # going to next block in nd2 file
            off += term + 1
            pos = self._u64(off)
            off += 8
            cnt = self._u64(off)
            off += 8
            # end of parsing if we encounter sig
            if name == sig:
                break

            name = str(name[:-1], encoding="utf-8")
            block = Munch(pos=pos, cnt=cnt, name=name)
            yield block
            if name.startswith(self.BLOCK_NAMES.IM_DATA_SEQ):
                field = int(name[len(self.BLOCK_NAMES.IM_DATA_SEQ) :])
                # map from field number to block Munch(pos, count, name)
                self.images[field] = block

    def _hex(self, start, count=0x100):
        col = 16
        printable = [chr(i) if 32 <= i < 128 else "." for i in range(256)]
        rows = count // col
        for row in range(rows):
            row_data = self.data[start + row * col : start + (row + 1) * col]
            x = f"{start + row * 16:08X}"
            h = " ".join([f"{i:02X}" for i in row_data])
            a = "".join([printable[i] for i in row_data])
            print(f"{x}  {h}  {a}")
        print()

    def _u8(self, off):
        return struct.unpack("B", self.data[off : off + 1])[0]

    def _u16(self, off):
        return struct.unpack("H", self.data[off : off + 2])[0]

    def _u32(self, off):
        return struct.unpack("I", self.data[off : off + 4])[0]

    def _u64(self, off):
        return struct.unpack("L", self.data[off : off + 8])[0]

    def _f64(self, off):
        return struct.unpack("d", self.data[off : off + 8])[0]

    def _str16(self, off):
        start = off
        while True:
            if self._u16(off) == 0:
                break
            off += 2
        stop = off
        try:
            return str(self.data[start:stop], encoding="utf-16"), (stop - start) + 2
        except Exception as e:
            print(f"FAIL {start:08X} {stop:08X} len={stop-start+2}")
            self._hex(start, stop - start + 2)
            raise

    def _read_metadata_field(self, pos, indent=0):
        """Parse metadata and return results."""
        type_ = self._u8(pos)
        pos += 1

        strlen = self._u8(pos) * 2
        pos += 1

        key, _ = self._str16(pos)
        pos += strlen

        if type_ == 1:
            val = self._u8(pos)
            pos += 1
        elif type_ == 2 or type_ == 3:
            val = self._u32(pos)
            pos += 4
        elif type_ == 5:
            val = self._u64(pos)
            pos += 8
        elif type_ == 6:
            val = self._f64(pos)
            pos += 8
        elif type_ == 8:
            val, val_len = self._str16(pos)
            pos += val_len
        elif type_ == 9:
            val = "?"
            val_len = self._u64(pos)
            pos += 8 + val_len
        elif type_ == 11:
            dict_len, val = self._read_dict(pos, indent)
            pos += dict_len
        else:
            raise Exception(f"Unhandled type {type_}")

        return pos, key, val

    def _read_dict(self, pos, indent):
        """
        A dict is an int process_count of keys followed a (mysterious) length
        4-bytes process_count-of-keys
        8-bytes mysterious length
        followed by the keys values:
            1 byte-type
            1 byte key len
            n-bytes the key
            Value
                The value depends on the type which are all simple except dicts
                When the value is itself a dict then the mysterious length includes the offset
                of this dict in the outer dict. Why?
        """
        d = Munch()

        start = pos
        count_of_keys = self._u32(pos)
        pos += 4

        _ = self._u64(
            pos
        )  # This is weird, it seems to includes a running off plus the 12 bytes we just read
        pos += 8

        for ki in range(count_of_keys):
            pos, key, val = self._read_metadata_field(pos, indent + 2)
            d[key] = val

        # Weird padding
        pos += 8 * count_of_keys

        return pos - start, d

    def _metadata_section(self, block):
        """Get metadata on given image."""
        pos = block.pos
        pos += 4  # Skip magic header

        name_len = self._u32(pos)
        pos += 4

        data_len = self._u64(pos)
        pos += 8

        pos += name_len

        end = pos + data_len

        d = Munch()
        while pos < end:
            pos, key, val = self._read_metadata_field(pos)
            d[key] = val

        return d

    def _data_block(self, block: Munch, func: Callable, elem_size: int) -> list:
        """Extract and return data from given block.

        ## pos -->  #### MAGIC HEADER   [4 bytes]        ####
                    #### NAME_LEN       [4 bytes]        ####
                    #### DATA_LEN       [8 bytes]        ####
                    #### NAME           [NAME_LEN bytes] ####
                    #### DATA           [DATA_LEN bytes] ####
        """
        pos = block.pos

        pos += 4  # Skip magic header

        name_len = self._u32(pos)
        pos += 4

        data_len = self._u64(pos)
        pos += 8

        count = data_len // elem_size
        assert data_len % elem_size == 0
        data = [func(pos + name_len + i * elem_size) for i in range(count)]
        return data

    def _dumpd(self, d, indent=0):
        for k, v in d.items():
            if not isinstance(v, dict):
                print(f"{'  ' * indent}{k} = {v}")
            else:
                print(f"{'  ' * indent}{k} = DICT")
                self._dumpd(v, indent + 1)

    # def get_fields(self, n_fields=None):
    #     """
    #     Returns numpy array of shape (n_fields, n_channels, dim, dim)
    #     """
    #     if n_fields is None:
    #         n_fields = self.n_fields
    #
    #     ims = np.zeros((n_fields, self.n_channels, *self.dim))
    #
    #     for field in range(n_fields):
    #         block = self.images[field]
    #
    #         pos = block.pos
    #         pos += 4  # Skip magic header
    #
    #         name_len = self._u32(pos)
    #         pos += 4
    #
    #         data_len = self._u64(pos)
    #         pos += 8
    #
    #         pos += name_len
    #
    #         timestamp = self._f64(pos)
    #         pos += 8
    #
    #         n_pixels = self.dim[0] * self.dim[1] * self.n_channels
    #         im = np.ndarray(
    #             (n_pixels,), buffer=self.data[pos : pos + n_pixels * 2], dtype="uint16"
    #         )
    #         for channel in range(self.n_channels):
    #             ims[field, channel, :, :] = np.reshape(
    #                 im[channel :: self.n_channels], self.dim
    #             )
    #
    #     return ims

    def get_field(self, field: int, channel: int) -> np.ndarray:
        """Return asked field at the relevant channel from nd2 file.

        1) First stage is to advance the pointer to the image itself.

        2) Second stage is to convert the data to numpy array and to extract the
           the relevant channel (channels are multiplexed).

            # CH_1_1|CH_2_1|...|CH_N_1 #
            # CH_1_2|CH_2_2|...|CH_N_2 #
            # CH_1_3|CH_2_3|...|CH_N_3 #
            #            ...           #
            # CH_1_M|CH_2_M|...|CH_N_M #

            * the above stored concatenated on one axis in the numpy array
              it's shown as a table for simplicity

        """
        # First stage
        off = self._get_field_offset_in_file(field)
        n_pixels = self.dim[0] * self.dim[1] * self.n_channels

        # Seconds stage
        im = np.frombuffer(buffer=self.data, offset=off, dtype="uint16", count=n_pixels)
        return np.reshape(im[channel :: self.n_channels], self.dim)

    def _get_field_offset_in_file(self, field: int) -> int:
        """Calculate and return offset of field in file.

        ## pos -->  ####  MAGIC_HEADER [4 bytes]     ####
                    ####  NAME_LEN  [4 bytes]        ####
                    ####  DATA_LEN  [8 bytes]        ####
                    ####  NAME      [NAME_LEN bytes] ####
                    ####  TIMESTAMP [8 bytes]        ####
                    ####  DATA                       ####
        """
        block = self.images[field]
        pos = block.pos

        pos += 4  # Skip magic header
        name_len = self._u32(pos)
        pos += 4
        data_len = self._u64(pos)
        pos += 8
        pos += name_len
        timestamp = self._f64(pos)
        pos += 8

        return pos


@contextmanager
def ND2(path):
    with open(path, "rb") as f:
        map_obj = mmap.mmap(f.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ)
        try:
            nd2 = _ND2(map_obj)
            yield nd2
        finally:
            map_obj.close()
