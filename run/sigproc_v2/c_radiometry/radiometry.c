#include "math.h"
#include "memory.h"
#include "pthread.h"
#include "stdarg.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"
#include "c_common.h"
#include "radiometry.h"

#define PI2 (2.0 * M_PI)

void _dump_vec(Float64 *vec, int width, int height, char *msg) {
    trace("VEC %s [\n", msg);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            fprintf(_log, "%4.4f, ", vec[y * width + x]);
        }
        fprintf(_log, "\n");
    }
    fprintf(_log, "]\n");
    fflush(_log);
}

void psf_im(
    Float64 center_x, Float64 center_y, Float64 sigma_x, Float64 sigma_y,
    Float64 rho, Float64 *pixels, Size mea) {
    center_x -= 0.5;
    center_y -= 0.5;

    Float64 sgxs = sigma_x * sigma_x;
    Float64 sgys = sigma_y * sigma_y;
    Float64 rs = rho * rho;
    Float64 omrs = 1.0 - rs;
    Float64 tem_a = 1.0 / (sigma_x * sigma_y * omrs);
    Float64 denom = 2.0 * (rho - 1.0) * (rho + 1.0) * sgxs * sgys;
    Float64 numer_const = -2.0 * rho * sigma_x * sigma_y;
    Float64 linear_term = tem_a * sqrt(omrs);

    Float64 *dst = pixels;
    for(int i = 0; i < mea; i++) {
        Float64 y = (Float64)i;
        Float64 ympy = y - center_y;
        for(int j = 0; j < mea; j++) {
            Float64 x = (Float64)j;
            Float64 xmpx = x - center_x;
            // clang-format off
            *dst++ = (
                linear_term * exp(
                    (
                        numer_const * xmpx * ympy
                        + sgxs * ympy * ympy
                        + sgys * xmpx * xmpx
                    ) / denom
                ) / PI2
            );
            // clang-format on
        }
    }
}

Float64 *_get_psf_at_loc(RadiometryContext *ctx, Float64 loc_x, Float64 loc_y) {
    Index x_i = floor(ctx->n_divs * loc_x / ctx->width);
    Index y_i = floor(ctx->n_divs * loc_y / ctx->height);
    ensure_only_in_debug(
        0.0 <= loc_x && loc_x < ctx->width, "loc x out of bounds");
    ensure_only_in_debug(
        0.0 <= loc_y && loc_y < ctx->height, "loc x out of bounds");
    ensure_only_in_debug(0 <= x_i && x_i < ctx->n_divs, "x_i out of bounds");
    ensure_only_in_debug(0 <= y_i && y_i < ctx->n_divs, "y_i out of bounds");
    return f64arr_ptr2(&ctx->reg_psf_samples, y_i, x_i);
}

int _float_cmp(const void *a, const void *b) {
    Float64 delt = *(Float64 *)a - *(Float64 *)b;
    return delt < 0.0 ? -1 : (delt > 0.0 ? 1 : 0);
}

char *radiometry_field_stack_one_peak(RadiometryContext *ctx, Index peak_i) {
    /*
    Each cycle is sub-pixel aligned, but each peak can be at
    an arbitrary fractional offset (which has already been determine
    by the caller).

    Each region has gauss params (sigma_x, sigma_y, rho).

    Thus, every peak, every cycle has its own Gaussian parameters.

    Possible future optimizations:
        * Realistically there's probably only 0.1 of a pixel of precision
          in the position, sigmas, and rho so we could pre-compute these
          PSF images and just look them up instead of rebuilding them at every
          peak/channel/cycle/
    */
    Float64 *im = f64arr_ptr1(&ctx->cy_ims, 0);

    // Position
    Size n_cycles = ctx->n_cycles;
    Size mea = ctx->peak_mea;
    Size mea_sq = mea * mea;
    Float64 half_mea = (Float64)mea / 2.0;

    // loc is the location in image coordinates
    Float64 *loc_p = f64arr_ptr1(&ctx->locs, peak_i);
    Float64 loc_x = loc_p[1];
    Float64 loc_y = loc_p[0];

    if(!in_bounds(loc_x, 0, ctx->width)) {
        return NULL;
    }
    if(!in_bounds(loc_y, 0, ctx->height)) {
        return NULL;
    }

    // corner is the lower left pixel coordinate in image coordinates
    // where the (mea, mea) sub-image will be extracted
    // Add 0.5 to round up as opposed to floor to keep the spots more centered
    Index corner_x = floor(loc_x - half_mea + 0.5);
    Index corner_y = floor(loc_y - half_mea + 0.5);

    if(!in_bounds(corner_x, 0, ctx->width - mea)) {
        return NULL;
    }
    if(!in_bounds(corner_y, 0, ctx->height - mea)) {
        return NULL;
    }

    // center is the location relative to the the corner
    Float64 center_x = loc_x - (Float64)corner_x;
    Float64 center_y = loc_y - (Float64)corner_y;

    if(!in_bounds(center_x, 0, mea)) {
        return NULL;
    }
    if(!in_bounds(center_y, 0, mea)) {
        return NULL;
    }

    // Shape
    Index n_divs_minus_one = ctx->n_divs - 1;

    Float64 *psf_pixels = (Float64 *)malloc(sizeof(Float64) * mea_sq);
    ensure(psf_pixels != NULL, "malloc failed");

    Float64 *dat_pixels = (Float64 *)malloc(sizeof(Float64) * mea_sq);
    ensure(dat_pixels != NULL, "malloc failed");

    Float64 *bg_pixels = (Float64 *)malloc(sizeof(Float64) * mea_sq);
    ensure(bg_pixels != NULL, "malloc failed");

    // This constant was tuned by hand so that in a noisy free
    // environment we got within 0.1% of the correct answer and
    // left significant pixels for estimating the background.
    const Float64 psf_mask_thresh = 0.001;

    for(Index cy_i = 0; cy_i < n_cycles; cy_i++) {
        Float64 *psf_params = _get_psf_at_loc(ctx, loc_x, loc_y);
        Float64 sigma_x = psf_params[0];
        Float64 sigma_y = psf_params[1];
        Float64 rho = psf_params[2];

        psf_im(
            center_x, center_y, sigma_x, sigma_y, rho, psf_pixels,
            ctx->peak_mea);

        // COPY the data into a contiguous buffer and compute the background
        // stats simultaneously Since the PSF is normalized to unit volume we
        // use the const value psf_mask_thresh to determine the mask from
        // foreground to background.

        Float64 *dat_p;
        Float64 *psf_p = psf_pixels;
        Float64 *dst_p = dat_pixels;
        Float64 *bg_p = bg_pixels;
        Float64 bg_sum = 0.0;
        for(Index y = 0; y < mea; y++) {
            dat_p = f64arr_ptr3(&ctx->cy_ims, cy_i, corner_y + y, corner_x);
            for(Index x = 0; x < mea; x++) {
                if(*psf_p < psf_mask_thresh) {
                    // Background pixel
                    *bg_p++ = *dat_p;
                    bg_sum += *dat_p;
                }
                *dst_p++ = *dat_p++;
                psf_p++;
            }
        }

        Float64 bg_median = 0.0;
        Float64 bg_std = 0.0;
        int n_bg_samples = bg_p - bg_pixels;
        if(n_bg_samples > 0) {
            // Compute median and std of bg samples
            qsort(bg_pixels, n_bg_samples, sizeof(Float64), _float_cmp);
            bg_median = bg_pixels[n_bg_samples / 2];

            Float64 *bg_p = bg_pixels;
            Float64 bg_var = 0.0;
            for(Index i = 0; i < n_bg_samples; i++) {
                Float64 minus_mean = *bg_p++ - bg_median;
                bg_var += minus_mean * minus_mean;
            }
            bg_std = sqrt(bg_var);
        }

        // REMOVE the bg_median
        dat_p = dat_pixels;
        for(Index i = 0; i < mea_sq; i++) {
            *dat_p++ -= bg_median;
        }

        // SIGNAL
        Float64 psf_sum_square = 0.0;
        Float64 signal = 0.0;
        psf_p = psf_pixels;
        dat_p = dat_pixels;
        for(Index i = 0; i < mea_sq; i++) {
            Float64 psf_times_dat = *psf_p * *dat_p;
            signal += psf_times_dat;
            psf_sum_square += *psf_p * *psf_p;
            psf_p++;
            dat_p++;
        }
        signal /= psf_sum_square;

        // RESIDUALS mean
        Float64 residual_mean = 0.0;
        psf_p = psf_pixels;
        dat_p = dat_pixels;
        for(Index i = 0; i < mea_sq; i++) {
            Float64 residual = *dat_p - signal * *psf_p;
            residual_mean += residual;
            psf_p++;
            dat_p++;
        }
        residual_mean /= (Float64)mea_sq;

        // RESIDUALS variance
        Float64 residual_var = 0.0;
        psf_p = psf_pixels;
        dat_p = dat_pixels;
        for(Index i = 0; i < mea_sq; i++) {
            Float64 residual = *dat_p - signal * *psf_p;
            Float64 mean_centered = residual - residual_mean;
            residual_var += mean_centered * mean_centered;
            psf_p++;
            dat_p++;
        }
        residual_var /= (Float64)mea_sq;

        // NOISE
        Float64 noise = sqrt(residual_var / psf_sum_square);

        Float64 *out = f64arr_ptr2(&ctx->out_radiometry, peak_i, cy_i);
        out[0] = signal;
        out[1] = noise;
        out[2] = bg_median;
        out[3] = bg_std;
    }

    free(psf_pixels);
    free(dat_pixels);
    free(bg_pixels);

    return NULL;
}

char *radiometry_field_stack_peak_batch(
    RadiometryContext *ctx, Index peak_start_i, Index peak_stop_i) {
    // Wrap the radiometry_field_stack_one_peak in a loop to avoid all the
    // calling overhead per-peak
    for(Index peak_i = peak_start_i; peak_i < peak_stop_i; peak_i++) {
        char *err = radiometry_field_stack_one_peak(ctx, peak_i);
        if(err != NULL) {
            return err;
        }
    }
    return NULL;
}

char *context_init(RadiometryContext *ctx) {
    return NULL;
}

char *context_free(RadiometryContext *ctx) {
    return NULL;
}
