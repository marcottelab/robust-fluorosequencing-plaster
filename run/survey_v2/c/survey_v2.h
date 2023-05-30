#ifndef SIM_V2_H
#define SIM_V2_H

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
    Tab dytmat;
    Tab dytpeps;
    Tab pep_i_to_dytpep_row_i;
    Tab dyt_i_to_n_reads;
    Tab dyt_i_to_mlpep_i;
    Tab output_pep_i_to_isolation_metric;
    Tab output_pep_i_to_mic_pep_i;
    Index next_pep_i;
    Size n_threads;
    Size n_flann_cores;
    Size n_peps;
    Size n_neighbors;
    Size n_dyts;
    Size n_dyt_cols;
    Float32 distance_to_assign_an_isolated_pep;
    pthread_mutex_t *work_order_lock;
    struct FLANNParameters *flann_params;
    void *flann_index_id;
} SurveyV2Context;

#endif
