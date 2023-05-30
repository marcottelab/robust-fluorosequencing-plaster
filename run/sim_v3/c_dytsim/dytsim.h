#ifndef DYTSIM_H
#define DYTSIM_H

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
    Size n_peps;
    Size n_cycles;
    Size n_samples;
    Size n_channels;
    PIType pi_bleach;
    PIType pi_cyclic_block;
    PIType pi_initial_block;
    PIType pi_detach;
    PIType pi_edman_success;
    PIType pi_label_fail;
    Uint64 allow_edman_cterm;
    Tab cycles;
    Tab pcbs;
    Tab cbbs;
    Tab pep_recalls;
    Size count_only;
    Size n_max_dyts;
    Size n_max_dytpeps;
    Size n_dyt_row_bytes;
    Size n_max_dyt_hash_recs;
    Size n_max_dytpep_hash_recs;
    Tab pep_i_to_pcb_i;
    Tab out_counts;
    pthread_mutex_t *_work_order_lock;
    pthread_mutex_t *_tab_lock;
    Tab _dytrecs;
    Tab _dytpeps;
    Hash _dyt_hash;
    Hash _dytpep_hash;
} DytSimContext;

typedef struct {
    PCBType pep_i;
    PCBType ch_i;
    PCBType p_bright;
    PCBType p_bleach;
} PCB;

typedef struct {
    CBBType ch_i;
    CBBType p_bright;
    CBBType p_bleach;
} CBB;

typedef struct {
    Size n_new_dyts;
    Size n_new_dytpeps;
} Counts;

#endif
