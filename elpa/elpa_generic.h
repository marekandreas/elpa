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
                  elpa_solve_real_double, \
                \
                float*: \
                  elpa_solve_real_single, \
                \
                double complex*: \
                  elpa_solve_complex_double, \
                \
                float complex*: \
                  elpa_solve_complex_single \
        )(handle, a, ev, q, error)
