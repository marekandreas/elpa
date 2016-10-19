## DEPRECATED FEATURES OF *ELPA* ##

This file contains a list of features, which have been replaced by other options.
Thus these features are considered as deprecated, and will be removed at some point
in the (near) future from the *ELPA* library.

### A) Deprecated interfaces:###

In order to unfiy the namespace of the *ELPA* public interfaces, several interfaces
have been replaced by new names. The old interfaces will be removed

Deprecated interface             Replacement
===================================================
get_elpa_row_col_coms            elpa_get_communicators
get_elpa_communicators           elpa_get_communicators
solve_evp_real                   elpa_solve_evp_real_1stage
solve_evp_complex                elpa_solve_evp_complex_1stage
solve_evp_real_1stage            elpa_solve_evp_real_1stage
solve_evp_complex_1stage         elpa_solve_evp_complex_1stage
solve_evp_real_2stage            elpa_solve_evp_real_2stage
solve_evp_complex_2stage         elpa_solve_evp_complex_2stage
mult_at_b_real                   elpa_mult_at_b_real
mult_ah_b_complex                elpa_mult_ah_b_complex
invert_trm_real                  elpa_invert_trm_real
invert_trm_complex               elpa_invert_trm_complex
cholesky_real                    elpa_cholesky_real
cholesky_complex                 elpa_cholesky_complex


### B) Runtime options ###
At the moment no runtime options are deprecated
