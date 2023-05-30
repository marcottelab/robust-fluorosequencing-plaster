#ifndef PEAK_FIND_H
#define PEAK_FIND_H

#include "stdint.h"
#include "c_common.h"

/* WARNING
 * All structures below was generated using
 *    plaster.tools.c_common.c_common_tools.struct_emit_header
 * Editing these structures require careful updating of the
 * corresponding *.c file to maintain consistency in the
 * C-Python interface.
 */

typedef struct {
    U16Arr ch_ims;
    Size n_channels;
    Size max_n_locs;
    Size im_mea;
    Tab sub_locs_tab;
    U64Arr out_n_locs;
    F64Arr out_locs;
    U64Arr out_warning_n_locs_overflow;
    U64Arr out_debug;
} PeakFindContext;

typedef struct {
    ChannelLocId loc_ids;
    Size ambiguous;
} SubLoc;

#endif
