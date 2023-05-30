#include "math.h"
#include "memory.h"
#include "pthread.h"
#include "stdarg.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"
#include "c_common.h"
#include "peak_find.h"

Bool any(Size *arr, Size size) {
    for(Size i = 0; i < size; i++) {
        if(arr[i]) {
            return 1;
        }
    }
    return 0;
}

char *peak_find(PeakFindContext *ctx, Index reg_i) {
    ensure_only_in_debug(ctx->sub_locs_tab.base != NULL, "Bad out_sublocs");
    const Size n_channels = ctx->n_channels;
    Index ch_i;

    const Size im_mea = ctx->im_mea;
    Size mask_shape[2] = {im_mea, im_mea};
    U8Arr msk_arr = u8arr_malloc(2, mask_shape);
    Uint8 *msk = u8arr_ptr1(&msk_arr, 0);

    U16Arr *ch_ims = &ctx->ch_ims;

    ensure(is_pow_2(im_mea), "im_mea must be a power of 2");
    Uint64 y_shift = pow_2_to_shift(im_mea);

    Index stack_top = 0;
    const Size stack_size = 256;
    const Size stack_size_minus_4 = 256 - 4;
    Uint32 stack[stack_size];
#define PUSH(a, b)                                          \
    stack[stack_top] = ((a & 0xFFFF) << 16) | (b & 0xFFFF); \
    stack_top++;
#define POP(a, b)                              \
    stack_top--;                               \
    a = (stack[stack_top] & 0xFFFF0000) >> 16; \
    b = (stack[stack_top] & 0xFFFF);

    Index off = 0;
    Size hits[N_MAX_CHANNELS];

// In this context, a hit is any pixel value greater than 0
#define CMP(_x, _y)                                \
    for(Index i = 0; i < n_channels; i++) {        \
        hits[i] = *u16arr_ptr3(ch_ims, i, _y, _x); \
    }

    for(Index y = 0; y < im_mea; y++) {
        for(Index x = 0; x < im_mea; x++) {
            if(msk[y * im_mea + x] == 0) {
                CMP(x, y); // Sets hits

                // debug
                //*u64arr_ptr2(&ctx->out_debug, y, x) = hits > 0 ? 1 : 0;

                if(any(hits, n_channels)) {
                    // Seed found, flood
                    Uint64 accum_loc_y = 0;
                    Uint64 accum_loc_x = 0;
                    Uint64 loc_n_pixels = 0;
                    Bool ambiguous = 0;

                    // This array will be set to the first loc_id we find in
                    // each channel. If we find another loc_id in the same peak
                    // area in the sam channel, we know that there are >2 peaks
                    // in the same channel in the same area, so we toss it out.
                    Uint16 channel_loc_ids[N_MAX_CHANNELS];
                    for(Index i = 0; i < n_channels; i++) {
                        channel_loc_ids[i] = hits[i];
                    }

                    PUSH(y, x);
                    while(stack_top > 0) {
                        // POP and EVALUATE the predicate of the pixel on the
                        // stack
                        Uint16 _y, _x;
                        POP(_y, _x);
                        if(0 <= _x && _x < im_mea && 0 <= _y && _y < im_mea) {
                            Uint8 *_msk = u8arr_ptr2(&msk_arr, _y, _x);
                            if(_msk[0] == 0 && stack_top < stack_size_minus_4) {
                                CMP(_x, _y);

                                for(Index i = 0; i < n_channels; i++) {
                                    // If we found a new peak for a channel, and
                                    // we haven't found one for that channel yet
                                    if(hits[i] != 0 &&
                                       channel_loc_ids[i] == 0) {
                                        channel_loc_ids[i] = hits[i];
                                    }

                                    if(hits[i] != 0 &&
                                       hits[i] != channel_loc_ids[i]) {
                                        // We found a different loc_id in the
                                        // same peak area in the same channel.
                                        // This means this peak is ambiguous
                                        ambiguous = 1;
                                    }
                                }

                                if(any(hits, n_channels)) {
                                    *u64arr_ptr2(&ctx->out_debug, _y, _x) = 1;
                                    accum_loc_y += _y;
                                    accum_loc_x += _x;
                                    loc_n_pixels += 1;

                                    _msk[0] = 1;
                                    PUSH(_y + 1, _x);
                                    PUSH(_y - 1, _x);
                                    PUSH(_y, _x + 1);
                                    PUSH(_y, _x - 1);
                                }
                            }
                        }
                    }

                    Size n_rows_free = tab_n_rows_remain(&ctx->sub_locs_tab);
                    if(n_rows_free > 0) {
                        SubLoc sub_loc;
                        // TODO: memory leak here! Ask Zack if there's a way to
                        // allocate a fixed size array in a fixupstruct
                        for(int i = 0; i < n_channels; i++) {
                            sub_loc.loc_ids[i] = channel_loc_ids[i];
                        }
                        sub_loc.ambiguous = ambiguous;
                        tab_add(&ctx->sub_locs_tab, &sub_loc, NULL);
                    } else {
                        Uint64 *out_warning =
                            u64arr_ptr1(&ctx->out_warning_n_locs_overflow, 0);
                        *out_warning = 1;
                        goto break_scan;
                    }
                }
            }
        }
    }

break_scan:

    u8arr_free(&msk_arr);

    return NULL;
}

char *context_init(PeakFindContext *ctx) {
    return NULL;
}

char *context_free(PeakFindContext *ctx) {
    return NULL;
}
