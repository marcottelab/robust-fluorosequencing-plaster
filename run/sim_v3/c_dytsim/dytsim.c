#include "alloca.h"
#include "inttypes.h"
#include "math.h"
#include "memory.h"
#include "pthread.h"
#include "signal.h"
#include "stdarg.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"
#include "c_common.h"
#include "dytsim.h"

/*
This is the "sim" phase of plaster implemented in C.

Inputs (see typedef DytSimContext in sim.h):
    * A list of "PCB"s which encode a "flu" or "fluorosequence"
      wherein each position in a peptide is a block in a table like:

      position, channel, brightness_probability, bleach_probability
             0,  np.nan,  np.nan,  np.nan
             1,       0,     0.9,     0.1
             2,  np.nan,  np.nan,  np.nan
             3,       1,     0.9,     0.2

    * Various parameters of the simulation such as
      probability_of_bleaching, etc.
    * Working buffers

Algorithm:
    For each peptide "flu" (PCB):
        Allocate a buffer for a working copy of the flu
        For n_samples Monte-Carlo simulations:
            * Copy the master peptide flu into the working copy
            * For each chemical cycle:
                  * Possibly chemically damage the peptide, preventing Edmans
                  * Remove the leading edge aa (or Edman fail)
                  * Allow flu to possibly "detach" (all fluors go dark)
                  * "Image" by summing the remaining dyts
                  * Allow each fluorophore on the working copy an
                    opportunity to bleach
            * We now have a dyetrack or "dyt"
            * Make a 64 bit hash key from that dyetrack
            * Look up the dyetrack in the Dye Tracks (Dyts) Hash;
              if it has never been seen before, add it; increment count.
            * Make another 64 bit hash key by combining the dyetrack hash key
              with the pep_i.
            * Lookup using this "DytPep" hash key into the Dyt Pep Hash;
              if it has never been seen before, add it; increment count.

Definitions:
    DytRec = Dyt-Track - a monotonically decreasing array of dyecounts
        ex: 3, 3, 2, 2, 2, 1, 0
    DytPepRec = A record that associates (dye_i, pep_i, count)
    Tab = A generic object that tracks how many rows have been added
        into a growing array. The pre-allocated tab buffer must large
        enough to accommodate the row or an assertion will be thrown.
    Hash = A simple 64-bit hashkey tab that maintains a pointer value.
    DytSimContext = All of the context (parameters, buffers, inputs, etc)
        that are needed in order to run the simulation
    Flu = Fluoro label sequence -
        For peptide: ABCDEF
        flu:         .10.01
        Labels: ch0 labels CE, c1 labels BF
        the encoding of "." is max_channels-1
    p_* = floating point probability of * (0-1)
    pi_* = integer probability where (0-1) is mapped to (0-MAX_UINT)
    bright_prob = the inverse of all the ways a dye can fail to be visible.
        In other words, the probability that a dye is active, ie bright


There are two Tabs maintained in the context:
    dyts: (count, dyt_i, array(n_channels, n_cycles))
    dytpeps: (count, dyt_i, pep_i)

There are two hash tabs:
    dyt_hash: key=dyetrack (note: not dyt_i), val=(count, dyt_i)
    dytpep_hash: key=(dyetrack, pep_i) , val=(count, dyt_i, pep_i)
*/

// Helpers
//=========================================================================================

PIType prob_to_p_i(double p) {
    // Convert p (double 0-1) into a 64 bit integer
    ensure(0.0 <= p && p <= 1.0, "probability out of range");
    long double w = floorl((long double)p * (long double)(UINT64_MAX));
    Uint64 ret = (Uint64)w;
    // printf("ret=%" PRIu64 "\n", ret);
    return ret;
}

PIType p_i_inv(PIType p_i) {
    // return the (1-p) probability in Uint64 space
    return (Uint64)UINT64_MAX - p_i;
}

PriorParameterType p_i_to_prob(PIType p_i) {
    return (PriorParameterType)((long double)p_i / (long double)(UINT64_MAX));
}

// See setup() and *_get_haskey()
static Uint64 hashkey_factors[256];

int setup_and_sanity_check(RNG *rng, Size n_channels, Size n_cycles) {
    // Setup the hashkey_factors with random numbers and
    // Check that the compiler sizes are what is expected.
    // return 0 == success

    if(rng_p_i(rng, UINT64_MAX) != 1) {
        printf("Failed sanity check: rng_p_i\n");
        return 6;
    }

    if(sizeof(DytRec) != 16) {
        printf("Failed sanity check: DytRec size\n");
        return 7;
    }

    Size n_hashkey_factors =
        sizeof(hashkey_factors) / sizeof(hashkey_factors[0]);
    for(Index i = 0; i < n_hashkey_factors; i++) {
        hashkey_factors[i] =
            (rand() * rand() * rand() * rand() * rand() * rand() * rand()) %
            UINT64_MAX;
    }

    if(n_channels * n_cycles >= n_hashkey_factors) {
        printf("Failed sanity check: n_channels * n_cycles >= "
               "n_hashkey_factors\n");
        return 9;
    }

    if(prob_to_p_i(0.0) != (Uint64)0) {
        printf("Failed sanity check: prob_to_p_i(0.0)\n");
        return 10;
    }

    if(prob_to_p_i(1.0) != (Uint64)UINT64_MAX) {
        printf(
            "Failed sanity check: prob_to_p_i(1.0) %ld %ld\n", prob_to_p_i(1.0),
            (Uint64)UINT64_MAX);
        return 11;
    }

    PIType p_i = prob_to_p_i(.3);
    PIType p_i_inv_inv = p_i_inv(p_i_inv(p_i));
    if(p_i_inv_inv != p_i) {
        printf(
            "Failed sanity check: p_i_inv(p_i_inv(%ld)) = %ld\n", p_i,
            p_i_inv_inv);
        return 12;
    }

    return 0;
}

// Dyts = Dye tracks
//=========================================================================================

HashKey dyt_get_hashkey(DytRec *dyt, Size n_channels, Size n_cycles) {
    // Get a hashkey for the DytRec by a dot product with a set of random 64-bit
    // values initialized in the hashkey_factors
    HashKey key = 0;
    Uint64 *p = hashkey_factors;
    DytType *d = dyt->chcy_dyt_counts;
    for(Index i = 0; i < n_channels * n_cycles; i++) {
        key += (*p++) * (Uint64)(*d++);
    }
    return key + 1; // +1 to reserve 0
}

Size dyt_n_bytes(Size n_channels, Size n_cycles) {
    // Return aligned DytRec size
    Size size = sizeof(DytRec) + sizeof(DytType) * n_cycles * n_channels;
    int over = size % 8;
    int padding = over == 0 ? 0 : 8 - over;
    return size + padding;
}

void dyt_set_chcy(
    DytRec *dst, DytType src_val, Size n_channels, Size n_cycles, Index ch_i,
    Index cy_i) {
    // DytRec chcy_dyt_counts is a 2D array (n_channels, n_cycles)
    ensure_only_in_debug(
        0 <= ch_i && ch_i < n_channels && 0 <= cy_i && cy_i < n_cycles,
        "dyt set out of bounds");
    Uint64 index = (n_cycles * ch_i) + cy_i;
    ensure_only_in_debug(
        0 <= index && index < n_channels * n_cycles,
        "dyt set out of bounds index");
    dst->chcy_dyt_counts[index] = src_val;
}

void dyt_clear(DytRec *dst, Size n_channels, Size n_cycles) {
    // Clear a single DytRec
    memset(dst->chcy_dyt_counts, 0, sizeof(DytType) * n_channels * n_cycles);
}

Size dyt_sum(DytRec *dyt, Size n_chcy) {
    // return the sum of all channel, all cycles (for debugging)
    Size sum = 0;
    for(Index i = 0; i < n_chcy; i++) {
        sum += dyt->chcy_dyt_counts[i];
    }
    return sum;
}

void dyt_dump_one(DytRec *dyt, Size n_channels, Size n_cycles) {
    // debugging
    for(Index ch_i = 0; ch_i < n_channels; ch_i++) {
        for(Index cy_i = 0; cy_i < n_cycles; cy_i++) {
            printf("%d ", dyt->chcy_dyt_counts[ch_i * n_cycles + cy_i]);
        }
        printf("  ");
    }
    printf(": count=%4ld\n", dyt->count);
}

void dyt_dump_all(Tab *dyts, Size n_channels, Size n_cycles) {
    // debugging
    for(Index i = 0; i < dyts->n_rows; i++) {
        tab_var(DytRec, dyt, dyts, i);
        dyt_dump_one(dyt, n_channels, n_cycles);
    }
}

void dyt_dump_one_hex(
    DytRec *dyts, Size n_dyts, Size n_channels, Size n_cycles) {
    // debugging
    DytRec *rec = dyts;
    Uint8 *ptr = (Uint8 *)dyts;
    for(Index i = 0; i < n_dyts; i++) {
        HashKey key = dyt_get_hashkey(rec, n_channels, n_cycles);
        printf("%016lX ", key);
        for(Index i = 0; i < 8; i++) {
            printf("%02x", *ptr++);
        }
        printf("  ");
        for(Index ch_i = 0; ch_i < n_channels; ch_i++) {
            for(Index cy_i = 0; cy_i < n_cycles; cy_i++) {
                printf("%02x ", *ptr++);
            }
            printf("  ");
        }
        printf("\n");
        rec++;
    }
}

// DytPep
//=========================================================================================

HashKey dytpep_get_hashkey(HashKey dyt_hashkey, Index pep_i) {
    // Note, 0 is an illegal return but is very unlikely except
    // under very weird circumstances. The check is therefore only
    // performec under DEBUG
    HashKey key = dyt_hashkey * hashkey_factors[0] +
                  pep_i * hashkey_factors[1] + 1; // + 1 to reserve 0
    ensure_only_in_debug(key != 0, "dytpep hash == 0");
    return key;
}

void dytpep_dump_one(DytPepRec *dytpep) {
    // Debugging
    printf("%4d %4d %4d\n", dytpep->dyt_i, dytpep->pep_i, dytpep->n_reads);
}

void dytpep_dump_all(Tab *dytpeps) {
    // Debugging
    for(Index i = 0; i < dytpeps->n_rows; i++) {
        dytpep_dump_one(tab_ptr(DytPepRec, dytpeps, i));
    }
}

// sim
//=========================================================================================

void model_sequential_labeling(
    RNG *rng, Tab *pcb_block, Size n_aas, DytType *flu, PIType *pi_bright,
    PIType *pi_bleach, PIType *pi_bright_by_ch, PIType *pi_bleach_by_ch,
    Size n_channels, Uint64 *pi_label_accum) {

    // It may be for performance that this should be inlined into dytsim_one_pep
    // but for development clarity I'll start with it as a separate fn which is
    // called once per sample for a given peptide.
    //
    // Previously the flu and other arrays for the peptide were built once with
    // the information in pcb_block and used for every sample.  But we wish to
    // model "sequential" labeling in which each channel is labeled one after
    // the other using the same 'click-clack' chemsitry with efficiency X, such
    // that subsequent channels have a chance of mis-labeling an aa that failed
    // to label in the previous step(s).
    //
    // If p_label_failure is 0.2, then the following shows the probabilities
    // of labeling/mis-labeling an aa that begins with label in correct channel:
    //
    // Correct Channel      P(ch0)    P(ch1)   P(ch2)      P(ch3)       P(nan)
    //      0                 0.8     .2(.8)   (.2)^2(.8) (.2)^3(.8)   (.2)^4
    //      1                 0       0.8      .2(.8)     (.2)^2(.8)   (.2)^3
    //
    //  and so on for other "Correct Channels"
    //
    // This information is captured in pi_label_accum which is n_channels long
    // and stores the first row of cumulative probabilities such that for each
    // aa label, we can use rng once to get a value that we then compare to the
    // values in the accum to find what the resulting channel label is.
    // E.g. using the floating point values above, if the rng yields .97 for an
    // aa that began with correct channel 0, it means we land in the column
    // for ch2, (for each col check if .97 < cumulative prob).  Note that in
    // practice pi_label_accum only contains cumulative probabilities up to
    // the last possible channel, and there is a probability slice at the end
    // which is the probability of the aa not getting labeled at all - so if
    // the prob lands beyond all cumulative probability for existing chans,
    // it means the aa will be unlabeled.
    //
    // Since the label for an aa may change to another channel, we
    // also need the p_bright and p_bleach values for all channels to
    // correctly construct the pi_bright and pi_bleach arrays.  These are
    // in the pi_bleach_by_ch and pi_bright_by_ch arrays.

    for(Index aa_i = 0; aa_i < n_aas; aa_i++) {
        tab_var(PCB, pcb_row, pcb_block, aa_i);

        PCBType f_ch_i = isnan(pcb_row->ch_i) ? NO_LABEL : (pcb_row->ch_i);
        ensure_only_in_debug(
            0 <= f_ch_i && f_ch_i < N_MAX_CHANNELS, "f_ch_i out of bounds");

        // If this aa is labeled, allow for sequential-label mislabel event
        if(f_ch_i < n_channels) {
            // an aa can only ever be labeled with its correct ch or ones that
            // come after, so n_possible_ch says how many possibilities there
            // are and how far to read into the pi_label_accum of probabilities.
            // rng_p_iz will return the index into the pi_label_accum allowing
            // us to select a correct label, a mislabel, or a no-label event
            // based on the set of probabilities.
            //
            Size n_possible_ch = n_channels - (Size)f_ch_i;
            Index mislabel_offset =
                rng_p_iz(rng, pi_label_accum, n_possible_ch);
            f_ch_i += mislabel_offset;
            if(f_ch_i ==
               n_channels) // beyond existing chans, this is "no label"
                f_ch_i = NO_LABEL;

            ensure_only_in_debug(
                f_ch_i == NO_LABEL || f_ch_i < n_channels,
                "f_ch_i bad channel");

            Index i_ch_i = (Index)f_ch_i;
            pi_bright[aa_i] = pi_bright_by_ch[i_ch_i];
            pi_bleach[aa_i] = pi_bleach_by_ch[i_ch_i];
        } else {
            // This aa not labeled
            pi_bright[aa_i] = 0;
            pi_bleach[aa_i] = 0;
        }
        flu[aa_i] = (DytType)f_ch_i;
    }
}

Counts dytsim_one_pep(
    DytSimContext *ctx, RNG *rng, Index pep_i, Tab *pcb_block, Size n_aas) {
    // Runs the Monte-Carlo simulation of one peptide flu over n_samples
    // See algorithm described at top of file.
    // Returns the number of NEW dyts

    // Make local copies of inner-loop variables
    DytType ch_sums[N_MAX_CHANNELS];
    Size n_cycles = ctx->n_cycles;
    Size n_samples = ctx->n_samples;
    Size n_channels = ctx->n_channels;
    CycleKindType *cycles = tab_ptr(CycleKindType, &ctx->cycles, 0);
    Uint64 pi_detach = ctx->pi_detach;
    Uint64 pi_edman_success = ctx->pi_edman_success;
    Uint64 pi_label_fail = ctx->pi_label_fail;
    Uint64 pi_cyclic_block = ctx->pi_cyclic_block;
    Uint64 pi_initial_block = ctx->pi_initial_block;
    Uint64 allow_edman_cterm = ctx->allow_edman_cterm;
    Tab *dyts = &ctx->_dytrecs;
    Tab *dytpeps = &ctx->_dytpeps;
    Tab *cbbs = &ctx->cbbs;
    Hash dyt_hash = ctx->_dyt_hash;
    Hash dytpep_hash = ctx->_dytpep_hash;
    Size n_flu_bytes = sizeof(DytType) * n_aas;
    Size n_new_dyts = 0;
    Size n_new_dytpeps = 0;

    if(ctx->count_only) {
        // Add one record to both dyt and dytpeps
        if(dyts->n_rows == 0) {
            tab_add(dyts, 0, TAB_NO_LOCK);
        }
        if(dytpeps->n_rows == 0) {
            tab_add(dytpeps, 0, TAB_NO_LOCK);
        }
    }

    // save the original pi_edman_success
    Uint64 save_pi_edman_success = pi_edman_success;

    DytType *flu = (DytType *)alloca(n_flu_bytes);
    DytType *working_flu = (DytType *)alloca(n_flu_bytes);
    PIType *pi_bright = (PIType *)alloca(sizeof(PIType) * n_aas);
    PIType *pi_bleach = (PIType *)alloca(sizeof(PIType) * n_aas);

    // This loop sets up the flu, pi_bright, and pi_bleach arrays for
    // the current peptide based on information in the pcb_block (the
    // pcbs for the current peptide).  If we want to model "sequential-labeling"
    // in which mis-labeling can occur across channels, it means we need to
    // essentially do this per-sample -- because we need to potentially
    // alter the contents of the pcb_block (which represents perfect labeling)
    // per sample -- and note that if an AA that should be labeled in ch0
    // now gets labeled in ch1, we really also need new ch1-appropriate
    // values for p_bright and p_bleach.  Which means we should probably
    // just pass in p_bright and p_bleach for each channel separately.
    //
    // Always do perfect labeling once to check for unlabeled peptide:
    // TODO: this could be removed and a call made to
    // model_sequential_labeling with p_label_sucess = 1.0.  For now I want to
    // use it as a sanity check that the fn produces the same result as this one
    // in that case.
    for(Index aa_i = 0; aa_i < n_aas; aa_i++) {
        tab_var(PCB, pcb_row, pcb_block, aa_i);

        ensure_only_in_debug(
            (Index)pcb_row->pep_i == pep_i,
            "Mismatching pep_i in pcb_row pep_i=%ld row_pep_i=%ld aa_i=%ld",
            pep_i, (Index)pcb_row->pep_i, aa_i);

        PCBType f_ch_i = isnan(pcb_row->ch_i) ? NO_LABEL : (pcb_row->ch_i);
        ensure_only_in_debug(
            0 <= f_ch_i && f_ch_i < N_MAX_CHANNELS, "f_ch_i out of bounds");
        flu[aa_i] = (DytType)f_ch_i;

        PCBType p_bright = pcb_row->p_bright;
        p_bright = isnan(p_bright) ? 0.0 : p_bright;
        ensure_only_in_debug(
            0.0 <= p_bright && p_bright <= 1.0,
            "p_bright out of range pep_i=%ld aa_i=%ld %f", pep_i, aa_i,
            p_bright);
        pi_bright[aa_i] = prob_to_p_i(p_bright);

        PCBType p_bleach = pcb_row->p_bleach;
        p_bleach = isnan(p_bleach) ? 0.0 : p_bleach;
        ensure_only_in_debug(
            0.0 <= p_bleach && p_bleach <= 1.0,
            "p_bleach out of range pep_i=%ld aa_i=%ld %f", pep_i, aa_i,
            p_bleach);
        pi_bleach[aa_i] = prob_to_p_i(p_bleach);
    }

    // If we're doing sequential labeling, setup local arrays with
    // pi_bright and pi_bleach per channel for easy lookup.
    // pi_label_accum will store cumulative probabilities for labeling
    // the "correct" channel followed by subsequent possible incorrect
    // channel label.  See docs in model_sequential_labeling()
    PIType *pi_bright_by_ch = (PIType *)0;
    PIType *pi_bleach_by_ch = (PIType *)0;
    PIType *pi_label_accum = (PIType *)0;
    if(pi_label_fail != 0) {
        pi_bright_by_ch = (PIType *)alloca(sizeof(PIType) * n_channels);
        pi_bleach_by_ch = (PIType *)alloca(sizeof(PIType) * n_channels);
        pi_label_accum = (PIType *)alloca(sizeof(PIType) * n_channels);
        PIType pi_label_success = p_i_inv(pi_label_fail);
        PriorParameterType accum_increment;
        PriorParameterType p_label_fail = p_i_to_prob(pi_label_fail);
        PriorParameterType p_label_success = 1.0 - p_label_fail;
        for(Index ch_i = 0; ch_i < n_channels; ch_i++) {
            tab_var(CBB, cbb_row, cbbs, ch_i);
            ensure_only_in_debug(
                (Index)cbb_row->ch_i == ch_i,
                "Mismatching ch_i in cbb_row ch_i=%ld row_ch_i=%ld", ch_i,
                (Index)cbb_row->ch_i);
            pi_bright_by_ch[ch_i] = prob_to_p_i(cbb_row->p_bright);
            pi_bleach_by_ch[ch_i] = prob_to_p_i(cbb_row->p_bleach);

            // accum label probabilities - see model_sequential_labeling()
            if(ch_i == 0) {
                accum_increment = p_label_success;
                pi_label_accum[ch_i] = pi_label_success;
            } else {
                accum_increment *= p_label_fail;
                pi_label_accum[ch_i] =
                    pi_label_accum[ch_i - 1] + prob_to_p_i(accum_increment);
            }
        }
    }

    // working_dyt is volatile stack copy of the out-going DytRec
    Size n_dyttrack_bytes = dyt_n_bytes(ctx->n_channels, ctx->n_cycles);
    DytRec *working_dyt = (DytRec *)alloca(n_dyttrack_bytes);
    memset(working_dyt, 0, n_dyttrack_bytes);

    DytRec *nul_dyt = (DytRec *)alloca(n_dyttrack_bytes);
    memset(nul_dyt, 0, n_dyttrack_bytes);

    // CHECK for unlabelled peptide
    // This check must be done for the "perfect" labeling as indicated by the
    // initial pcb_block -- not because of a mislabel event in the sequential
    // label model.
    int has_any_dyt = 0;
    for(Index i = 0; i < n_aas; i++) {
        if(flu[i] != N_MAX_CHANNELS - 1) {
            has_any_dyt = 1;
            break;
        }
    }

    Size n_dark_samples = 0;
    Size n_non_dark_samples = 0;
    while(has_any_dyt && n_non_dark_samples < n_samples) {
        if(n_dark_samples > 10 * n_samples) {
            // Emergency exit. The recall is so low that we need to
            // just give up and declare that it can't be measured.
            n_dark_samples = 0;
            n_non_dark_samples = 0;
            break;
        }

        if(pi_label_fail != 0) {
            // model sequential labeling in which aas may be mislabeled.  This
            // will overwrite the working_flu, pi_bright, and pi_bleach arrays
            model_sequential_labeling(
                rng, pcb_block, n_aas, working_flu, pi_bright, pi_bleach,
                pi_bright_by_ch, pi_bleach_by_ch, n_channels, pi_label_accum);
        } else {
            // otherwise just copy the 'perfect label' flu into working_flu
            memcpy(working_flu, flu, n_flu_bytes);
        }

        // GENERATE the working_dyetrack sample (Monte Carlo)
        //-------------------------------------------------------
        dyt_clear(working_dyt, n_channels, n_cycles);
        pi_edman_success = save_pi_edman_success;

        // MODEL dark-dyes (dyes dark before the first image)
        // These darks are the product of various dye factors which
        // are passed into this module already converted into PI form
        // (probability in 0 - max_unit64) by the pi_bright arrays
        for(Index aa_i = 0; aa_i < n_aas; aa_i++) {
            if(!rng_p_i(rng, pi_bright[aa_i])) {
                working_flu[aa_i] = NO_LABEL;
            }
        }

        // Initial DAMAGE.  See comment below for DAMAGE.  This is a one-time
        // initial probability of the peptide being damaged such that no
        // Edman cycles will work.
        if(rng_p_i(rng, pi_initial_block)) {
            pi_edman_success = prob_to_p_i(0.);
        }

        Index head_i = 0;
        for(Index cy_i = 0; cy_i < n_cycles; cy_i++) {
            // DAMAGE...(cyclic)
            // Damage is when an Edman reagent damages the peptide so that
            // no further Edman degradation is possible. This is a
            // hypothesized source of remainders. If damage happens then
            // pi_edman_success will be set to 0. and no further Edman
            // degradation steps will succeed.
            // EDMAN...
            // Edman degradation chews off the N-terminal amino-acid.
            // With some peptide-attachment schemes, edman of the C-terminal AA
            // isn't possible. If successful this advances the "head_i" which is
            // where we're summing from.
            if(cycles[cy_i] == CYCLE_TYPE_EDMAN) {
                if(rng_p_i(rng, pi_cyclic_block)) {
                    pi_edman_success = prob_to_p_i(0.);
                }

                if(rng_p_i(rng, pi_edman_success)) {
                    // always do rand64 to preserve RNG order independent of
                    // following condition
                    if(allow_edman_cterm || head_i < n_aas - 1) {
                        head_i++;
                    }
                }
            }

            // DETACH...
            // Detachment is when a peptide comes loose from the surface.
            // This means that all subsequent measurements go dark.
            if(rng_p_i(rng, pi_detach)) {
                for(Index aa_i = head_i; aa_i < n_aas; aa_i++) {
                    working_flu[aa_i] = NO_LABEL;
                }
                break;
            }

            // IMAGE (sum up all active dyes in each channel)...
            // To make this avoid any branching logic, the ch_sums[]
            // is allocated to with N_MAX_CHANNELS which includes the
            // "NO_LABEL" which is defined to be N_MAX_CHANNELS-1. Thus the
            // sums will also count the number of unlabelled positions, but
            // we can just ignore that extra "NO LABEL" channel.
            memset(ch_sums, 0, sizeof(ch_sums));
            for(Index aa_i = head_i; aa_i < n_aas; aa_i++) {
                ch_sums[working_flu[aa_i]]++;
            }

            for(Index ch_i = 0; ch_i < n_channels; ch_i++) {
                dyt_set_chcy(
                    working_dyt, ch_sums[ch_i], n_channels, n_cycles, ch_i,
                    cy_i);
            }

            // TODO: Try treating BLEACH pre image because destruction is
            // chemically mediated as well as photo mediated
            // Will have to fix tests
            // BLEACH / DYE DESTRUCTION...
            for(Index aa_i = head_i; aa_i < n_aas; aa_i++) {
                // For all REMAINING dyes (head_i:...) give
                // each dye a chance to be destroyed
                // TODO: Profile which is better, the branch here or just
                // letting it over-write
                if(working_flu[aa_i] < NO_LABEL &&
                   rng_p_i(rng, pi_bleach[aa_i])) {
                    working_flu[aa_i] = NO_LABEL;
                }
            }
        }

        // At this point we have the flu sampled into working_dyt
        // Now we look it up in the hash tabs.
        //-------------------------------------------------------

        if(memcmp(working_dyt, nul_dyt, n_dyttrack_bytes) == 0) {
            // The row was empty, note this and continue to try another sample
            n_dark_samples++;
            continue;
        }

        n_non_dark_samples++;

        HashKey dyt_hashkey =
            dyt_get_hashkey(working_dyt, n_channels, n_cycles);
        HashRec *dyt_hash_rec = hash_get(dyt_hash, dyt_hashkey);
        DytRec *dyt;
        ensure(dyt_hash_rec != (HashRec *)0, "dyt_hash full");
        if(dyt_hash_rec->key == 0) {
            // New record
            n_new_dyts++;
            Index dyt_i = 0;
            if(!ctx->count_only) {
                dyt_i = tab_add(dyts, working_dyt, ctx->_tab_lock);
            }
            dyt = tab_ptr(DytRec, dyts, dyt_i);
            dyt_hash_rec->key = dyt_hashkey;
            dyt->count++;
            dyt->dyt_i = dyt_i;
            dyt_hash_rec->val = dyt;
        } else {
            // Existing record
            // Because this is a MonteCarlo sampling it really doesn't
            // matter if we occasionally mis-count due to thread
            // contention therefore there is no lock here.
            dyt = (DytRec *)(dyt_hash_rec->val);
            tab_validate_only_in_debug(dyts, dyt);
            dyt->count++;
        }
        tab_validate_only_in_debug(dyts, dyt);

        // SAVE the (dyt_i, pep_i) into dytpeps
        // (or inc count if it already exists)
        //-------------------------------------------------------
        HashKey dytpep_hashkey = dytpep_get_hashkey(dyt_hashkey, pep_i);
        HashRec *dytpep_hash_rec = hash_get(dytpep_hash, dytpep_hashkey);
        ensure(dytpep_hash_rec != (HashRec *)0, "dytpep_hash full");
        if(dytpep_hash_rec->key == 0) {
            // New record
            // If this were used multi-threaded, this would be a race condition
            n_new_dytpeps++;
            Index dytpep_i = 0;
            if(!ctx->count_only) {
                dytpep_i = tab_add(dytpeps, NULL, ctx->_tab_lock);
            }
            tab_var(DytPepRec, dytpep, dytpeps, dytpep_i);
            dytpep_hash_rec->key = dytpep_hashkey;
            dytpep->dyt_i = dyt->dyt_i;
            dytpep->pep_i = pep_i;
            dytpep->n_reads++;
            dytpep_hash_rec->val = dytpep;
        } else {
            // Existing record
            // Same argument as above
            DytPepRec *dpr = (DytPepRec *)dytpep_hash_rec->val;
            tab_validate_only_in_debug(dytpeps, dpr);
            dpr->n_reads++;
        }
    }

    RecallType recall = (RecallType)0.0;
    if(n_dark_samples + n_non_dark_samples > 0) {
        recall = (RecallType)n_non_dark_samples /
                 (RecallType)(n_dark_samples + n_non_dark_samples);
        tab_set(&ctx->pep_recalls, pep_i, &recall);
    } else {
        tab_set(&ctx->pep_recalls, pep_i, &recall);
    }

    Counts counts;
    counts.n_new_dyts = n_new_dyts;
    counts.n_new_dytpeps = n_new_dytpeps;
    return counts;
}

char *context_init(DytSimContext *ctx) {
    // Sanity check
    RNG rng; // A placeholder rng during sanity test
    ensure(
        setup_and_sanity_check(&rng, ctx->n_channels, ctx->n_cycles) == 0,
        "Sanity checks failed");

    // Allocate memory
    uint8_t *dyts_buf = calloc(ctx->n_max_dyts, ctx->n_dyt_row_bytes);
    check_and_return(dyts_buf != NULL, "dytsim.c OOM dyts_buf");
    ctx->_dytrecs = tab_by_n_rows(
        dyts_buf, ctx->n_max_dyts, ctx->n_dyt_row_bytes, TAB_GROWABLE);

    uint8_t *dytpeps_buf = calloc(ctx->n_max_dytpeps, sizeof(DytPepRec));
    check_and_return(dytpeps_buf != NULL, "dytsim.c OOM dytpeps_buf");
    ctx->_dytpeps = tab_by_size(
        dytpeps_buf, ctx->n_max_dytpeps * sizeof(DytPepRec), sizeof(DytPepRec),
        TAB_GROWABLE);

    HashRec *dyt_hash_buf = calloc(ctx->n_max_dyt_hash_recs, sizeof(HashRec));
    check_and_return(dyt_hash_buf != NULL, "dytsim.c OOM dyt_hash_buf");
    ctx->_dyt_hash = hash_init(dyt_hash_buf, ctx->n_max_dyt_hash_recs);

    HashRec *dytpep_hash_buf =
        calloc(ctx->n_max_dytpep_hash_recs, sizeof(HashRec));
    check_and_return(dytpep_hash_buf != NULL, "dytsim.c OOM dytpep_hash_buf");
    ctx->_dytpep_hash = hash_init(dytpep_hash_buf, ctx->n_max_dytpep_hash_recs);

    ctx->_work_order_lock = malloc(sizeof(pthread_mutex_t));
    check_and_return(
        ctx->_work_order_lock != NULL, "dytsim.c OOM ctx->_work_order_lock");

    ctx->_tab_lock = malloc(sizeof(pthread_mutex_t));
    check_and_return(ctx->_tab_lock != NULL, "dytsim.c OOM ctx->_tab_lock");

    // POPULATE the lookup from pep_i to the index into pcb_i
    Index last_pep_i = 0;
    Size n_pcb_rows = ctx->pcbs.n_rows;
    tab_set(&ctx->pep_i_to_pcb_i, 0, &last_pep_i);
    for(Index row_i = 0; row_i < n_pcb_rows; row_i++) {
        PCB *pcb = tab_ptr(PCB, &ctx->pcbs, row_i);
        Index pep_i = (Index)pcb->pep_i;
        if(pep_i != last_pep_i) {
            tab_set(&ctx->pep_i_to_pcb_i, pep_i, &row_i);
            last_pep_i = pep_i;
        }
    }

    // Add end of table lookup
    tab_set(&ctx->pep_i_to_pcb_i, ctx->n_peps, &n_pcb_rows + 1);

    // Allocate thread locks
    int ret = pthread_mutex_init(ctx->_work_order_lock, NULL);
    check_and_return(ret == 0, "pthread work_order_lock create failed");

    ret = pthread_mutex_init(ctx->_tab_lock, NULL);
    check_and_return(ret == 0, "pthread tab_lock create failed");

    return NULL;
}

char *context_free(DytSimContext *ctx) {
    free(ctx->_work_order_lock);
    free(ctx->_tab_lock);
    free(ctx->_dytrecs.base);
    free(ctx->_dytpeps.base);
    free(ctx->_dyt_hash.recs);
    free(ctx->_dytpep_hash.recs);

    return NULL;
}

char *dytsim_batch(
    DytSimContext *ctx, RNG *rng, Index start_pep_i, Index stop_pep_i) {
    Size n_dyts = 0;
    Size n_dytpeps = 0;
    for(Index pep_i = start_pep_i; pep_i < stop_pep_i; pep_i++) {
        Index pcb_i = tab_get(Index, &ctx->pep_i_to_pcb_i, pep_i);
        Index pcb_i_plus_1 = tab_get(Index, &ctx->pep_i_to_pcb_i, pep_i + 1);
        Size n_aas = pcb_i_plus_1 - pcb_i;
        Tab pcb_block = tab_subset(&ctx->pcbs, pcb_i, n_aas);

        Counts counts = dytsim_one_pep(ctx, rng, pep_i, &pcb_block, n_aas);
        n_dyts += counts.n_new_dyts;
        n_dytpeps += counts.n_new_dytpeps;
    }
    if(ctx->count_only) {
        tab_set(&ctx->out_counts, 0, &n_dyts);
        tab_set(&ctx->out_counts, 1, &n_dytpeps);
    }
    return NULL;
}

char *copy_results(DytSimContext *ctx, DytType *dyts, DytPepType *dytpeps) {
    // dyts and dytpeps are encoded in structs (see DytRec, DytPepRec)
    // and thus need to be copied one row at a time into the final
    // contiguous output buffers.

    Size n_bytes_per_row;

    // dyt_dump_all(&ctx->_dytrecs, ctx->n_channels, ctx->n_cycles);

    Index n_dytrecs = ctx->_dytrecs.n_rows;
    n_bytes_per_row = sizeof(DytType) * ctx->n_channels * ctx->n_cycles;
    for(Index dytrec_i = 0; dytrec_i < n_dytrecs; dytrec_i++) {
        DytRec *dytrec = tab_ptr(DytRec, &ctx->_dytrecs, dytrec_i);
        DytType *dst =
            &dyts[n_bytes_per_row * (dytrec->dyt_i + 1)]; // +1 to account for
                                                          // nul row
        memcpy(dst, dytrec->chcy_dyt_counts, n_bytes_per_row);
    }

    // dytpep_dump_all(&ctx->_dytpeps);
    // trace("\n");

    Index n_dytpeps = ctx->_dytpeps.n_rows;
    n_bytes_per_row = sizeof(DytPepType) * 3;

    // Reserved row
    DytPepType *dst = &dytpeps[0];
    dst[0] = 0;
    dst[1] = 0;
    dst[2] = 0;
    for(Index dytpep_i = 0; dytpep_i < n_dytpeps; dytpep_i++) {
        DytPepRec *dytpep = tab_ptr(DytPepRec, &ctx->_dytpeps, dytpep_i);
        DytPepType *dst =
            &dytpeps[3 * (dytpep_i + 1)]; // +1 because of dytpeps reserved row
        dst[0] = dytpep->dyt_i +
                 1; // + 1 because all dyts got shifted for the nul row
        dst[1] = dytpep->pep_i;
        dst[2] = dytpep->n_reads;
    }

    return NULL;
}

void context_dump(DytSimContext *ctx) {
    printf("DytSimContext:\n");
    printf("  n_peps=%" PRIu64 "\n", ctx->n_peps);
    printf("  n_cycles=%" PRIu64 "\n", ctx->n_cycles);
    printf("  n_samples=%" PRIu64 "\n", ctx->n_samples);
    printf("  n_channels=%" PRIu64 "\n", ctx->n_channels);
    // printf("ret=%" PRIu64 "\n", ret);

    printf("  pi_cyclic_block=%" PRIu64 "\n", ctx->pi_cyclic_block);
    printf("  pi_initial_block=%" PRIu64 "\n", ctx->pi_initial_block);
    printf("  pi_detach=%" PRIu64 "\n", ctx->pi_detach);
    printf("  pi_edman_success=%" PRIu64 "\n", ctx->pi_edman_success);
    // Some are left out
}

Uint64 xrng_uint64(xRNG *xrng) {
    return xnext(xrng);
}