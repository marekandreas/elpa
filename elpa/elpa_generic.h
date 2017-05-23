#pragma once

/**
 * \todo document elpa_set()
 */
#define elpa_set(e, name, value, error) _Generic((value), \
                int: \
                  elpa_set_integer, \
                \
                double: \
                  elpa_set_double \
        )(e, name, value, error)

/**
 * \todo document elpa_solve()
 */
#define elpa_solve(handle, a, ev, q, error) _Generic((a), \
                double*: \
                  elpa_solve_d, \
                \
                float*: \
                  elpa_solve_f, \
                \
                double complex*: \
                  elpa_solve_dc, \
                \
                float complex*: \
                  elpa_solve_fc \
        )(handle, a, ev, q, error)


/**
 * \todo document elpa_cholesky()
 */
#define elpa_cholesky(handle, a, error) _Generic((a), \
                double*: \
                  elpa_cholesky_d, \
                \
                float*: \
                  elpa_cholesky_f, \
                \
                double complex*: \
                  elpa_cholesky_dc, \
                \
                float complex*: \
                  elpa_cholesky_fc \
        )(handle, a, error)


/**
 * \todo document elpa_invert_triangular()
 */
#define elpa_invert_triangular(handle, a, error) _Generic((a), \
                double*: \
                  elpa_invert_trm_d, \
                \
                float*: \
                  elpa_invert_trm_f, \
                \
                double complex*: \
                  elpa_invert_trm_dc, \
                \
                float complex*: \
                  elpa_invert_trm_fc \
        )(handle, a, error)

