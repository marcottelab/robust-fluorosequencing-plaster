#ifndef RADIOMETRY_H
#define RADIOMETRY_H

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
    F64Arr cy_ims;
    F64Arr locs;
    F64Arr reg_psf_samples;
    Index ch_i;
    Size n_cycles;
    Size n_peaks;
    Float64 n_divs;
    Size peak_mea;
    Float64 height;
    Float64 width;
    Float64 raw_height;
    Float64 raw_width;
    F64Arr out_radiometry;
} RadiometryContext;

#endif
