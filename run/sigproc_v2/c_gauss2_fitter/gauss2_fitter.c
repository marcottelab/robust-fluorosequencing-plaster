#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "time.h"
#include "string.h"
#include "levmar.h"
#include "c_common.h"
#include "gauss2_fitter.h"
#ifndef LM_DBL_PREC
    #error This program assumes that levmar has been compiled with double precision.
#endif

// A Gaussian 2D fitter using Levenberg-Marquardt algorithm
// implemented with levmar-2.6. See http://users.ics.forth.gr/~lourakis/levmar/
// Thanks to John Haven Davis for the Jacobian

double dlevmar_opts[LM_OPTS_SZ] = {
    LM_INIT_MU,
    // scale factor for initial mu
    1e-15,
    // stopping threshold for ||J^T e||_inf
    1e-15,
    // stopping threshold for ||Dp||^2
    1e-20,
    // stopping threshold for ||e||^2
    LM_DIFF_DELTA
    // step used in difference approximation to the Jacobian.
    // If delta<0, the Jacobian is approximated  with central differences
    // which are more accurate (but slower!) compared to the forward differences
    // employed by default. Set to NULL for defaults to be used.
};

char *dlevmar_stop_reasons[] = {
    "Unknown",
    "stopped by small gradient J^T e",
    "stopped by small Dp",
    "stopped by itmax",
    "singular matrix. Restart from current p with increased mu",
    "no further error reduction is possible. Restart with increased mu",
    "stopped by small ||e||_2",
    "stopped by invalid (i.e. NaN or Inf) 'func' values. This is a user error",
};

char *get_dlevmar_stop_reason_from_info(np_float64 *info) {
    int reason = (int)info[6];
    if(0 <= reason && reason < 8) {
        return dlevmar_stop_reasons[reason];
    }
    return dlevmar_stop_reasons[0];
}

void dump_pixels(char *msg, double *pixels) {
    trace("%s: PIXELS [\n", msg);
    for(int i = 0; i < 11 * 11; i++) {
        if(i % 11 == 0) {
            fprintf(_log, "\n");
        }
        fprintf(_log, "%f, ", pixels[i]);
    }
    fprintf(_log, "]\n\n");
}

void dump_params(char *msg, double *params) {
    trace("%s: PARAMS [\n", msg);
    for(int i = 0; i < PARAM_N_FULL_PARAMS; i++) {
        fprintf(_log, "%f, ", params[i]);
    }
    fprintf(_log, "]\n\n");
    trace("%s: END PARAMS\n", msg);
}

void dump_jacobian(char *msg, double *jac, int cnt) {
    trace("%s: JAC [\n", msg);
    for(int i = 0; i < cnt; i++) {
        if(i % 7 == 0) {
            fprintf(_log, "\n");
        }
        if(i % (7 * 11) == 0) {
            fprintf(_log, "\n");
        }
        fprintf(_log, "%f, ", jac[i]);
    }
    fprintf(_log, "]\n\n");
}

void dump_info(double *info) {
    /*
    info[0] = ||e||_2 at initial p.
    info[1-4] = [ ||e||_2, ||J^T e||_inf,  ||Dp||_2, \mu/max[J^T J]_ii ], all
    computed at estimated p. info[5] = number of iterations, info[6] = reason
    for terminating: (See dlevmar_stop_reasons for string constants) 0 - Unknown
        1 - stopped by small gradient J^T e
        2 - stopped by small Dp
        3 - stopped by itmax
        4 - singular matrix. Restart from current p with increased \mu
        5 - no further error reduction is possible. Restart with increased mu
        6 - stopped by small ||e||_2
        7 - stopped by invalid (i.e. NaN or Inf) "func" values; a user error
    info[7] = number of function evaluations
    info[8] = number of Jacobian evaluations
    info[9] = number of linear systems solved, i.e. number of attempts for
    reducing error
    */
    trace("INFO:\n");
    trace("  e^2 at p0: %f\n", info[0]);
    trace("  mu?: %f\n", info[4]);
    trace("  n_iter: %f\n", info[5]);
    trace("  reason to stop: %s\n", get_dlevmar_stop_reason_from_info(info));
}

void gauss_2d(double *params, double *pixels, int m, int n, void *data) {
    // Arguments:
    //   p: parameters of the 2D Gaussian: array [amp, sig_x, sig_y, pos_x,
    //   pos_y, rho, offset] pixels: Destination buffer that will contain the
    //   function evaluation given the parameters m: number of parameters
    //   (length of p) n: number of data points data: data

    double amp = params[0];
    double sig_x = params[1];
    double sig_y = params[2];
    double pos_x = params[3] - 0.5;
    double pos_y = params[4] - 0.5;
    double rho = params[5];
    double offset = params[6];

    double pi2 = 2.0 * M_PI;
    double sgxs = sig_x * sig_x;
    double sgys = sig_y * sig_y;
    double rs = rho * rho;
    double omrs = 1.0 - rs;
    double tem_a = 1.0 / (sig_x * sig_y * omrs);
    double denom = 2.0 * (rho - 1.0) * (rho + 1.0) * sgxs * sgys;
    double numer_const = -2.0 * rho * sig_x * sig_y;
    double linear_term = amp * tem_a * sqrt(omrs);

    int mea = (int)sqrt(n);

    double *dst = pixels;
    for(int i = 0; i < mea; i++) {
        double y = (double)i;
        double ympy = y - pos_y;
        for(int j = 0; j < mea; j++) {
            double x = (double)j;
            double xmpx = x - pos_x;
            *dst++ =
                (offset + linear_term *
                              exp((numer_const * xmpx * ympy +
                                   sgxs * ympy * ympy + sgys * xmpx * xmpx) /
                                  denom) /
                              pi2);
        }
    }

    //    dump_params("model func", params);
    //    dump_pixels("model func", pixels);
}

void jac_gauss_2d(double *params, double *dst_jac, int m, int n, void *data) {
    // Arguments:
    //   p: parameters array [amp, sig_x, sig_y, pos_x, pos_y, rho, offset]
    //   dst_jac: destination buffer to hold the Jacobian
    //   m: number of parameters (length of p)
    //   n: number of data points
    //   data: data

    double amp = params[0];
    double sig_x = params[1];
    double sig_y = params[2];
    double pos_x = params[3] - 0.5;
    double pos_y = params[4] - 0.5;
    double rho = params[5];
    double offset = params[6];

    double pi2 = 2.0 * M_PI;
    double sgxs = sig_x * sig_x;
    double sgys = sig_y * sig_y;
    double rs = rho * rho;
    double omrs = 1.0 - rs;
    double tem_a = 1.0 / (sig_x * sig_y * omrs);
    double denom = 2.0 * (rho - 1) * (rho + 1) * sgxs * sgys;
    double linear_term = -2.0 * rho * sig_x * sig_y;
    int mea = (int)sqrt((double)n);

    double *dst = dst_jac;
    for(int i = 0; i < mea; i++) {
        double y = (double)i;
        double ympy = y - pos_y;
        for(int j = 0; j < mea; j++) {
            double x = (double)j;
            double xmpx = x - pos_x;
            double tem_b = tem_a * sqrt(omrs) *
                           exp((linear_term * xmpx * ympy + sgxs * ympy * ympy +
                                sgys * xmpx * xmpx) /
                               denom) /
                           pi2;

            *dst++ = tem_b;

            double tem_ab_amp = tem_a * tem_b * amp;
            double xmpxy = xmpx * ympy;

            *dst++ = tem_ab_amp *
                     (-omrs * sgxs * sig_y - rho * sig_x * xmpxy +
                      sig_y * xmpx * xmpx) /
                     sgxs;

            *dst++ = -tem_ab_amp *
                     (omrs * sig_x * sgys + rho * sig_y * xmpxy -
                      sig_x * ympy * ympy) /
                     sgys;

            *dst++ = tem_a * (-rho * sig_x * ympy + sig_y * xmpx) * tem_b *
                     amp / sig_x;

            *dst++ = -tem_ab_amp * (rho * sig_y * xmpx - sig_x * ympy) / sig_y;

            *dst++ = -tem_a * tem_b *
                     (rho * (-omrs * sgys + ympy * ympy) * sgxs -
                      sig_y * (2.0 - omrs) * xmpxy * sig_x +
                      rho * sgys * xmpx * xmpx) *
                     amp / (sig_x * omrs * sig_y);

            *dst++ = 1.0;
        }
    }

    //    dump_jacobian("jac", dst_jac, mea * mea * 7);
}

int fit_gauss_2d(
    np_float64 *pixels, np_int64 mea, np_float64 params[PARAM_N_FIT_PARAMS],
    np_float64 *info, np_float64 *covar) {
    /*
    Fit a 2D Gaussian given a square array of pixels with length of size "mea"

    Arguments:
        pixels:
            Square array of pixels (mea x mea)
        mea:
            number of pixels on each side of the square pixels
        params:
            An array [7] of the initial guess of the parameters.
            Will be overwritten with the fit parameters.
            Order:
                amplitude, sigma_x, sigma_y, pos_x, pos_y, rho, offset
        info:
            An array [10] (or NULL if you don't need this info) which contains
    the fitting info: info[0] = ||e||_2 at initial p. info[1-4] = [ ||e||_2,
    ||J^T e||_inf,  ||Dp||_2, \mu/max[J^T J]_ii ], all computed at estimated p.
                 info[5] = number of iterations,
                 info[6] = reason for terminating:
                    (See dlevmar_stop_reasons for string constants)
                    0 - Unknown
                    1 - stopped by small gradient J^T e
                    2 - stopped by small Dp
                    3 - stopped by itmax
                    4 - singular matrix. Restart from current p with increased
    \mu 5 - no further error reduction is possible. Restart with increased mu 6
    - stopped by small ||e||_2 7 - stopped by invalid (i.e. NaN or Inf) "func"
    values; a user error info[7] = number of function evaluations info[8] =
    number of Jacobian evaluations info[9] = number of linear systems solved,
    i.e. number of attempts for reducing error covar: A array [7x7] (or NULL)
    will be filled in with the fitter's covariance estimate on parameters.

    Returns:
        0 on success, -1 on any sort of error
    */

    ensure(sizeof(np_float64) == 8, "np_float64 wrong size");
    int n_pixels = mea * mea;
    int ret = 0;

    ret = dlevmar_der(
        gauss_2d, jac_gauss_2d, (double *)params, (double *)pixels,
        PARAM_N_FIT_PARAMS, n_pixels, N_MAX_ITERATIONS, dlevmar_opts,
        (double *)info, NULL,
        // This is a working buffer that will be allocated if NULL
        // I timed it and it made no difference so it seemed easier
        // to let levmar handle malloc/free.
        (double *)covar, NULL);

    // Without jacobian, useful for sanity checking
    //    ret = dlevmar_dif(
    //        gauss_2d,
    //        params,
    //        pixels, PARAM_N_FIT_PARAMS, n_pixels,
    //        N_MAX_ITERATIONS, dlevmar_opts, info,
    //        NULL,  // See above about working buffer
    //        covar, NULL
    //    );

    return ret > 0 ? 0 : -1;
}

int fit_gauss_2d_on_float_image(
    np_float64 *im, np_int64 im_w, np_int64 im_h, np_int64 center_x,
    np_int64 center_y, np_int64 mea, np_float64 params[PARAM_N_FIT_PARAMS],
    np_float64 *info, np_float64 *covar, np_float64 *noise) {
    /*
    Like fit_gauss_2d but operates on an image with specified dimensions.
    Skip if the peak is even partially over-edges.

    Arguments:
        im, im_h, im_w: A float image of im_h, im_w pixels
        center_y, center_x: Coordinate of the center where the peak is extracted
        mea: length of side of square area to extract (must be odd)
        *: See fit_gauss_2d

    Returns:
         0 if success
        -1 if the fitter returned an error
        -2 if any portion of the request is outside the bounds of im.
    */

    ensure(
        (mea & 1) == 1,
        "Mea must be odd"); // Must be odd so that there is a center pixel
    int half_mea = mea / 2;
    int n_pixels = mea * mea;
    int ret = 0;

    double *pixels = (double *)alloca(sizeof(double) * n_pixels);
    int top = center_y - half_mea;
    int bot = center_y + half_mea;
    int lft = center_x - half_mea;
    int rgt = center_x + half_mea;

    if(top < 0 || bot >= im_h || lft < 0 || rgt >= im_w) {
        return -2;
    }

    ensure((bot - top + 1) * (rgt - lft + 1) == n_pixels, "size mismatch");

    // COPY from the image to a linear array of values
    double min_val = 1e10;
    double max_val = 0.0;
    double *dst = pixels;
    for(int y = top; y <= bot; y++) {
        np_float64 *src = &im[y * im_w + lft];
        for(int x = lft; x <= rgt; x++) {
            double pix = *src++;
            *dst++ = pix;
            min_val = min(min_val, pix);
            max_val = max(max_val, pix);
        }
    }

    // Handle special cases where the initial guess based on data.
    if(params[PARAM_AMP] == 0.0) {
        // Use peak width and max_val to guess amp
        params[PARAM_AMP] = 2.0 * 3.141592654 * params[PARAM_SIGMA_X] *
                            params[PARAM_SIGMA_Y] * (max_val - min_val);
    }

    if(params[PARAM_OFFSET] == 0.0) {
        // Use min_val to guess offfset
        params[PARAM_OFFSET] = min_val;
    }

    // Similarly, the min
    params[PARAM_OFFSET] = min_val;

    // dump_pixels("entrypoint", pixels);
    // dump_params("entrypoint", params);

    ret = dlevmar_der(
        gauss_2d, jac_gauss_2d, params, pixels, PARAM_N_FIT_PARAMS, n_pixels,
        N_MAX_ITERATIONS, dlevmar_opts, (double *)info, NULL,
        // This is a working buffer that will be allocated if NULL
        // I timed it and it made no difference so it seemed easier
        // to let levmar handle malloc/free.
        (double *)covar, NULL);

    int success = ret >= 0;
    *noise = 0.0;

    // dump_info(info);
    // trace("%s\n", get_dlevmar_stop_reason_from_info(info));

    // ret is the number of iterations (>=0) if successful other a negative
    // value
    if(success) {
        // RENDER out the fit and subtract to get residuals
        double *model_pixels = (double *)alloca(sizeof(double) * n_pixels);
        gauss_2d(params, model_pixels, PARAM_N_FIT_PARAMS, n_pixels, NULL);

        np_float64 residual_mean = 0.0;
        for(int i = 0; i < n_pixels; i++) {
            double *data = &pixels[i];
            double *model = &model_pixels[i];
            double residual = *data - *model;
            residual_mean += residual;
        }
        residual_mean /= (np_float64)n_pixels;

        np_float64 sse = 0.0;
        for(int i = 0; i < n_pixels; i++) {
            double *data = &pixels[i];
            double *model = &model_pixels[i];
            double residual = (*data - *model) - residual_mean;
            sse += residual * residual;
        }

        *noise = sqrt(sse);
    }

    return success ? 0 : ret;
}

char *gauss2_check() {
    check_and_return(sizeof(np_int64) == 8, "np_int64 not 8 bytes");
    check_and_return(sizeof(np_float64) == 8, "np_float64 not 8 bytes");
}

int validate_im(np_float64 *im, np_int64 im_w, np_int64 im_h) {
    for(int i = 0; i < im_w * im_h; i++) {
        if(isnan(im[i])) {
            return 0;
        }
    }
    return 1;
}

char *fit_array_of_gauss_2d_on_float_image(
    np_float64 *im, np_int64 im_w, np_int64 im_h, np_int64 mea,
    np_int64 n_peaks, np_int64 *center_x, np_int64 *center_y,
    np_float64 *params,     // This is a full set (PARAM_N_FULL_PARAMS columns)
    np_float64 *std_params, // This is a parallel set of params (same size) for
                            // the std. of the fit
    np_int64 *fails) {
    /*
    Call fit_gauss_2d_on_float_image on an array of peak locations

    Arguments:
        See fit_gauss_2d_on_float_image
        n_peaks: number of elements in the center_y and center_x arrays
        center_y, center_x: Peak positions
        params: n_peaks * N_FULL_PARAMS doubles expected to be initialized to a
    guess of the parameters std_params: n_peaks * N_FULL_PARAMS doubles expected
    to be initialized to a guess of the parameters fails: n_peaks. Will be zero
    if success or non-zero on any sort of failure

    Returns:
        Number of failures
    */

    np_float64 info[N_INFO_ELEMENTS];

    double covar[PARAM_N_FIT_PARAMS][PARAM_N_FIT_PARAMS];

    // check_and_return(validate_im(im, im_w, im_h), "Invalid im");

    np_int64 n_fails = 0;
    for(np_int64 peak_i = 0; peak_i < n_peaks; peak_i++) {
        np_float64 *p = &params[peak_i * PARAM_N_FULL_PARAMS];
        np_float64 noise = 0.0;

        if(center_x[peak_i] >= 0 && center_y[peak_i] >= 0) {
            // SKIP negative which is a sentinel for "DO NOT FIT"

            // trace("PEAK_I=%ld\n", peak_i);
            int res = fit_gauss_2d_on_float_image(
                im, im_w, im_h, center_x[peak_i], center_y[peak_i], mea, p,
                info, &covar[0][0], &noise);

            params[peak_i * PARAM_N_FULL_PARAMS + PARAM_NOISE] = noise;

            // COPY stdev of the covar. into the std_params (from its diagonal)
            np_float64 *dst_std = &std_params[peak_i * PARAM_N_FULL_PARAMS];

            // Note that this is only traversing the fit_params...
            for(int i = 0; i < PARAM_N_FIT_PARAMS; i++) {
                *dst_std++ = sqrt(covar[i][i]);
            }

            int failed_to_converge = (int)info[6] == 3;
            int failed = (res != 0 || failed_to_converge) ? 1 : 0;
            n_fails += failed;
            fails[peak_i] = failed;
        } else {
            // Skipped peaks are reported as "fails" so that they
            // will be NaN'd out by the caller, but they are technically fails
            // so we don't increment n_fails.
            fails[peak_i] = 1;
        }
    }

    return (char *)NULL;
}

char *synth_image(
    np_float64 *im, np_int64 im_w, np_int64 im_h, np_int64 peak_mea,
    np_int64 n_peaks,
    np_float64 *params // An array: (n_peaks, PARAM_N_FIT_PARAMS)
) {
    Float64 half_mea = (Float64)peak_mea / 2.0;

    ensure(sizeof(np_float64) == 8, "np_float64 wrong size");
    int n_pixels = peak_mea * peak_mea;

    double *model_pixels = (double *)alloca(sizeof(double) * n_pixels);
    Float64 *working_params =
        (Float64 *)alloca(sizeof(Float64) * PARAM_N_FIT_PARAMS);
    Float64 *src_params = params;

    for(Index peak_i = 0; peak_i < n_peaks; peak_i++) {
        memcpy(
            working_params, src_params, sizeof(Float64) * PARAM_N_FIT_PARAMS);
        src_params += PARAM_N_FIT_PARAMS;

        // Position
        Float64 loc_x = working_params[PARAM_CENTER_X];
        Float64 loc_y = working_params[PARAM_CENTER_Y];
        Float64 amp = working_params[PARAM_AMP];

        if(!in_bounds(loc_x, 0, im_w)) {
            continue;
        }
        if(!in_bounds(loc_y, 0, im_h)) {
            continue;
        }

        // corner is the lower left pixel coordinate in image coordinates
        // where the (peak_mea, peak_mea) sub-image will be extracted
        // Add 0.5 to round up as opposed to floor to keep the spots more
        // centered
        Index corner_x = floor(loc_x - half_mea + 0.5);
        Index corner_y = floor(loc_y - half_mea + 0.5);

        if(!in_bounds(corner_x, 0, im_w - peak_mea)) {
            continue;
        }
        if(!in_bounds(corner_y, 0, im_h - peak_mea)) {
            continue;
        }

        // center is the location relative to the the corner
        Float64 center_x = loc_x - (Float64)corner_x;
        Float64 center_y = loc_y - (Float64)corner_y;

        if(!in_bounds(center_x, 0, peak_mea)) {
            continue;
        }
        if(!in_bounds(center_y, 0, peak_mea)) {
            continue;
        }

        working_params[PARAM_CENTER_X] = center_x;
        working_params[PARAM_CENTER_Y] = center_y;

        gauss_2d(
            working_params, model_pixels, PARAM_N_FIT_PARAMS, n_pixels, NULL);

        for(Index y = 0; y < peak_mea; y++) {
            Float64 *src = &model_pixels[y * peak_mea];
            Float64 *dst = &im[(corner_y + y) * im_w + corner_x];
            for(Index x = 0; x < peak_mea; x++) {
                *dst += *src;
                dst++;
                src++;
            }
        }
    }
}
