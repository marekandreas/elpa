## DEPRECATED FEATURES OF *ELPA* ##

This file contains a list of features, which have been replaced by other options.
Thus these features are considered as deprecated, and will be removed at some point
in the (near) future from the *ELPA* library.

### A) Deprecated interfaces:###

With the release of ELPA 2017.05.001 a new, more general API for the library has
been published. All new features of ELPA will only be accesible via this new interface.
For ease of transion, the API as defined in release ELPA 2016.11.001 has been frozen in
and will be still supported for some time, albeit without any new features.

Independent of the freezing in of the old, legacy API from the release 2016.11.001 the
following listed interfaces will be removed at some time.

In order to unfiy the namespace of the *ELPA* public interfaces, several interfaces
have been replaced by new names. The old interfaces will be removed

Deprecated interface             Replacement                            Comment
==================================================================================================
get_elpa_row_col_coms            elpa_get_communicators                (removed since 2017.11.001)
get_elpa_communicators           elpa_get_communicators                (removed since 2017.11.001)
solve_evp_real                   elpa_solve_evp_real_1stage_double     (removed since 2017.11.001)
solve_evp_complex                elpa_solve_evp_complex_1stage_double  (removed since 2017.11.001)
solve_evp_real_1stage            elpa_solve_evp_real_1stage_double
solve_evp_complex_1stage         elpa_solve_evp_complex_1stage_double
solve_evp_real_2stage            elpa_solve_evp_real_2stage_double
solve_evp_complex_2stage         elpa_solve_evp_complex_2stage_double
mult_at_b_real                   elpa_mult_at_b_real_double
mult_ah_b_complex                elpa_mult_ah_b_complex_double
invert_trm_real                  elpa_invert_trm_real_double
invert_trm_complex               elpa_invert_trm_complex_double
cholesky_real                    elpa_cholesky_real_double
cholesky_complex                 elpa_cholesky_complex_double

For all symbols also the corresponding "_single" routines are available



### B) Runtime options ###
At the moment no runtime options are deprecated. However, future options will only be available via the new
interface
