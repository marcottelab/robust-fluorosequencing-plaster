#ifndef RADSIM_H
#define RADSIM_H

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
    Size n_cycles;
    Size n_channels;
    U8Arr dytmat;
    U32Arr dytpeps;
    U32Arr dytpep_i_to_out_i;
    Bool use_lognormal_model;
    PriorParameterType row_k_sigma;
    Tab ch_illum_priors;
    F32Arr out_radmat;
    F32Arr out_row_ks;
    U32Arr out_dyt_iz;
    U32Arr out_pep_iz;
} RadSimContext;

typedef struct {
    U32Arr dytpeps;
    Index n_samples_per_pep;
    U32Arr out_dytpeps;
    U32Arr pep_i_to_dytpep_i;
    U32Arr pep_i_to_n_dytpeps;
    Size n_peps;
} SamplePepsContext;

typedef struct {
    Float64 gain_mu;
    Float64 gain_sigma;
    Float64 bg_mu;
    Float64 bg_sigma;
    Float64 row_k_sigma;
} ChIllumPriors;

#endif
