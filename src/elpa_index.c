//    Copyright 2017, L. Hüdepohl and A. Marek, MPCDF
//
//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//    This particular source code file contains additions, changes and
//    enhancements authored by Intel Corporation which is not part of
//    the ELPA consortium.
//
//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
//    Authors: L. Huedepohl and A. Marek, MPCDF
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <elpa/elpa.h>
#include "elpa_index.h"

#include "config.h"

#ifdef WITH_OPENMP_TRADITIONAL
#include <omp.h>
#endif

int max_threads_glob;
int set_max_threads_glob=0;

static int enumerate_identity(elpa_index_t index, int i);
static int cardinality_bool(elpa_index_t index);
static int valid_bool(elpa_index_t index, int n, int new_value);

static int number_of_matrix_layouts(elpa_index_t index);
static int matrix_layout_enumerate(elpa_index_t index, int i);
static int matrix_layout_is_valid(elpa_index_t index, int n, int new_value);
static const char* elpa_matrix_layout_name(int layout);

static int number_of_solvers(elpa_index_t index);
static int solver_enumerate(elpa_index_t index, int i);
static int solver_is_valid(elpa_index_t index, int n, int new_value);
static const char* elpa_solver_name(int solver);

static int number_of_real_kernels(elpa_index_t index);
static int real_kernel_enumerate(elpa_index_t index, int i);
static int real_kernel_is_valid(elpa_index_t index, int n, int new_value);
static const char *real_kernel_name(int kernel);

static int number_of_complex_kernels(elpa_index_t index);
static int complex_kernel_enumerate(elpa_index_t index, int i);
static int complex_kernel_is_valid(elpa_index_t index, int n, int new_value);
static const char *complex_kernel_name(int kernel);

static int band_to_full_cardinality(elpa_index_t index);
static int band_to_full_enumerate(elpa_index_t index, int i);
static int band_to_full_is_valid(elpa_index_t index, int n, int new_value);

static int stripewidth_real_cardinality(elpa_index_t index);
static int stripewidth_real_enumerate(elpa_index_t index, int i);
static int stripewidth_real_is_valid(elpa_index_t index, int n, int new_value);

static int internal_nblk_cardinality(elpa_index_t index);
static int internal_nblk_enumerate(elpa_index_t index, int i);
static int internal_nblk_is_valid(elpa_index_t index, int n, int new_value);

static int stripewidth_complex_cardinality(elpa_index_t index);
static int stripewidth_complex_enumerate(elpa_index_t index, int i);
static int stripewidth_complex_is_valid(elpa_index_t index, int n, int new_value);

static int omp_threads_cardinality(elpa_index_t index);
static int omp_threads_enumerate(elpa_index_t index, int i);
static int omp_threads_is_valid(elpa_index_t index, int n, int new_value);

static int max_stored_rows_cardinality(elpa_index_t index);
static int max_stored_rows_enumerate(elpa_index_t index, int i);
static int max_stored_rows_is_valid(elpa_index_t index, int n, int new_value);

static int min_tile_size_cardinality(elpa_index_t index);
static int min_tile_size_enumerate(elpa_index_t index, int i);
static int min_tile_size_is_valid(elpa_index_t index, int n, int new_value);

#ifdef WITH_GPU_VERSION
int gpu_count();
#endif

static int use_gpu_id_cardinality(elpa_index_t index);
static int use_gpu_id_enumerate(elpa_index_t index, int i);
static int use_gpu_id_is_valid(elpa_index_t index, int n, int new_value);

static int valid_with_gpu(elpa_index_t index, int n, int new_value);
static int valid_with_gpu_elpa1(elpa_index_t index, int n, int new_value);
static int valid_with_gpu_elpa2(elpa_index_t index, int n, int new_value);

static int intermediate_bandwidth_cardinality(elpa_index_t index);
static int intermediate_bandwidth_enumerate(elpa_index_t index, int i);
static int intermediate_bandwidth_is_valid(elpa_index_t index, int n, int new_value);

static int cannon_buffer_size_cardinality(elpa_index_t index);
static int cannon_buffer_size_enumerate(elpa_index_t index, int i);
static int cannon_buffer_size_is_valid(elpa_index_t index, int n, int new_value);

static int na_is_valid(elpa_index_t index, int n, int new_value);
static int nev_is_valid(elpa_index_t index, int n, int new_value);
static int bw_is_valid(elpa_index_t index, int n, int new_value);
static int output_build_config_is_valid(elpa_index_t index, int n, int new_value);
static int gpu_is_valid(elpa_index_t index, int n, int new_value);
static int skewsymmetric_is_valid(elpa_index_t index, int n, int new_value);

static int is_positive(elpa_index_t index, int n, int new_value);

static int elpa_float_string_to_value(char *name, char *string, float *value);
static int elpa_float_value_to_string(char *name, float value, const char **string);

static int elpa_double_string_to_value(char *name, char *string, double *value);
static int elpa_double_value_to_string(char *name, double value, const char **string);

#define BASE_ENTRY(option_name, option_description, once_value, readonly_value, print_flag_value) \
                .base = { \
                        .name = option_name, \
                        .description = option_description, \
                        .once = once_value, \
                        .readonly = readonly_value, \
                        .env_default = "ELPA_DEFAULT_" option_name, \
                        .env_force = "ELPA_FORCE_" option_name, \
                        .print_flag = print_flag_value, \
                }

#define INT_PARAMETER_ENTRY(option_name, option_description, valid_func, print_flag) \
        { \
                BASE_ENTRY(option_name, option_description, 1, 0, print_flag), \
                .valid = valid_func, \
        }

#define BOOL_ENTRY(option_name, option_description, default, tune_level, tune_domain, print_flag) \
        { \
                BASE_ENTRY(option_name, option_description, 0, 0, print_flag), \
                .default_value = default, \
                .autotune_level = tune_level, \
                .autotune_domain = tune_domain, \
                .cardinality = cardinality_bool, \
                .enumerate = enumerate_identity, \
                .valid = valid_bool, \
        }

#define INT_ENTRY(option_name, option_description, default, tune_level, tune_domain, card_func, enumerate_func, valid_func, to_string_func, print_flag) \
        { \
                BASE_ENTRY(option_name, option_description, 0, 0, print_flag), \
                .default_value = default, \
                .autotune_level = tune_level, \
                .autotune_domain = tune_domain, \
                .cardinality = card_func, \
                .enumerate = enumerate_func, \
                .valid = valid_func, \
                .to_string = to_string_func, \
        }

#define INT_ANY_ENTRY(option_name, option_description, print_flag) \
        { \
                BASE_ENTRY(option_name, option_description, 0, 0, print_flag), \
        }

/* The order here is important! Tunable options that are dependent on other
 * tunable options must appear later in the list than their prerequisites */
static const elpa_index_int_entry_t int_entries[] = {
        INT_PARAMETER_ENTRY("na", "Global matrix has size (na * na)", na_is_valid, PRINT_STRUCTURE),
        INT_PARAMETER_ENTRY("nev", "Number of eigenvectors to be computed, 0 <= nev <= na", nev_is_valid, PRINT_STRUCTURE),
        INT_PARAMETER_ENTRY("nblk", "Block size of scalapack block-cyclic distribution", is_positive, PRINT_STRUCTURE),
        INT_PARAMETER_ENTRY("local_nrows", "Number of matrix rows stored on this process", NULL, PRINT_NO),
        INT_PARAMETER_ENTRY("local_ncols", "Number of matrix columns stored on this process", NULL, PRINT_NO),
        INT_PARAMETER_ENTRY("process_row", "Process row number in the 2D domain decomposition", NULL, PRINT_NO),
        INT_PARAMETER_ENTRY("process_col", "Process column number in the 2D domain decomposition", NULL, PRINT_NO),
        INT_PARAMETER_ENTRY("process_id", "Process rank", NULL, PRINT_NO),
        INT_PARAMETER_ENTRY("num_process_rows", "Number of process row number in the 2D domain decomposition", NULL, PRINT_STRUCTURE),
        INT_PARAMETER_ENTRY("num_process_cols", "Number of process column number in the 2D domain decomposition", NULL, PRINT_STRUCTURE),
        INT_PARAMETER_ENTRY("num_processes", "Total number of processes", NULL, PRINT_STRUCTURE),
        INT_PARAMETER_ENTRY("bandwidth", "If specified, a band matrix with this bandwidth is expected as input; bandwidth must be multiply of nblk", bw_is_valid, PRINT_YES),
        INT_ANY_ENTRY("mpi_comm_rows", "Communicator for inter-row communication", PRINT_NO),
        INT_ANY_ENTRY("mpi_comm_cols", "Communicator for inter-column communication", PRINT_NO),
        INT_ANY_ENTRY("mpi_comm_parent", "Parent communicator", PRINT_NO),
        INT_ANY_ENTRY("blacs_context", "BLACS context", PRINT_NO),
//#ifdef REDISTRIBUTE_MATRIX
        INT_ENTRY("internal_nblk", "Internally used block size of scalapack block-cyclic distribution", 0, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_ANY, \
                   internal_nblk_cardinality, internal_nblk_enumerate, internal_nblk_is_valid, NULL, PRINT_YES),
//#endif
#ifdef STORE_BUILD_CONFIG
        INT_ENTRY("output_build_config", "Output the build config", 0, ELPA_AUTOTUNE_NOT_TUNABLE, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cardinality_bool, enumerate_identity, output_build_config_is_valid, NULL, PRINT_NO),
#endif
	INT_ENTRY("matrix_order","Order of the matrix layout", COLUMN_MAJOR_ORDER, ELPA_AUTOTUNE_NOT_TUNABLE, ELPA_AUTOTUNE_DOMAIN_ANY, \
                         number_of_matrix_layouts, matrix_layout_enumerate, matrix_layout_is_valid, elpa_matrix_layout_name, PRINT_YES), \
        INT_ENTRY("solver", "Solver to use", ELPA_SOLVER_1STAGE, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        number_of_solvers, solver_enumerate, solver_is_valid, elpa_solver_name, PRINT_YES),
        INT_ENTRY("gpu", "Use GPU acceleration", 0, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cardinality_bool, enumerate_identity, gpu_is_valid, NULL, PRINT_YES),
        INT_ENTRY("is_skewsymmetric", "Matrix is skewsymmetic", 0, ELPA_AUTOTUNE_NOT_TUNABLE, 0,
                        cardinality_bool, enumerate_identity, skewsymmetric_is_valid, NULL, PRINT_YES),
        //default of gpu ussage for individual phases is 1. However, it is only evaluated, if GPU is used at all, which first has to be determined
        //by the parameter gpu and presence of the device
        INT_ENTRY("gpu_tridiag", "Use GPU acceleration for ELPA1 tridiagonalization", 1, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cardinality_bool, enumerate_identity, valid_with_gpu_elpa1, NULL, PRINT_YES),
        INT_ENTRY("gpu_solve_tridi", "Use GPU acceleration for ELPA solve tridi", 1, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cardinality_bool, enumerate_identity, valid_with_gpu, NULL, PRINT_YES),
        INT_ENTRY("gpu_trans_ev", "Use GPU acceleration for ELPA1 trans ev", 1, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cardinality_bool, enumerate_identity, valid_with_gpu_elpa1, NULL, PRINT_YES),
        INT_ENTRY("gpu_bandred", "Use GPU acceleration for ELPA2 band reduction", 1, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cardinality_bool, enumerate_identity, valid_with_gpu_elpa2, NULL, PRINT_YES),
	//not yet ported to GPU
        //INT_ENTRY("gpu_tridiag_band", "Use GPU acceleration for ELPA2 tridiagonalization", 1, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_ANY, \
        //                cardinality_bool, enumerate_identity, valid_with_gpu_elpa2, NULL, PRINT_YES),
        INT_ENTRY("gpu_trans_ev_tridi_to_band", "Use GPU acceleration for ELPA2 trans_ev_tridi_to_band", 1, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cardinality_bool, enumerate_identity, valid_with_gpu_elpa2, NULL, PRINT_YES),
        INT_ENTRY("gpu_trans_ev_band_to_full", "Use GPU acceleration for ELPA2 trans_ev_band_to_full", 1, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cardinality_bool, enumerate_identity, valid_with_gpu_elpa2, NULL, PRINT_YES),
	INT_ENTRY("use_gpu_id", "Calling MPI task will use this gpu id", -99, ELPA_AUTOTUNE_NOT_TUNABLE, ELPA_AUTOTUNE_DOMAIN_ANY, \
		  use_gpu_id_cardinality, use_gpu_id_enumerate, use_gpu_id_is_valid, NULL, PRINT_YES), 
        INT_ENTRY("real_kernel", "Real kernel to use if 'solver' is set to ELPA_SOLVER_2STAGE", ELPA_2STAGE_REAL_DEFAULT, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, \
                        number_of_real_kernels, real_kernel_enumerate, real_kernel_is_valid, real_kernel_name, PRINT_YES),
        INT_ENTRY("complex_kernel", "Complex kernel to use if 'solver' is set to ELPA_SOLVER_2STAGE", ELPA_2STAGE_COMPLEX_DEFAULT, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_COMPLEX, \
                        number_of_complex_kernels, complex_kernel_enumerate, complex_kernel_is_valid, complex_kernel_name, PRINT_YES),

        INT_ENTRY("min_tile_size", "Minimal tile size used internally in elpa1_tridiag and elpa2_bandred", 0, ELPA_AUTOTUNE_NOT_TUNABLE, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        min_tile_size_cardinality, min_tile_size_enumerate, min_tile_size_is_valid, NULL, PRINT_YES),
        INT_ENTRY("intermediate_bandwidth", "Specifies the intermediate bandwidth in ELPA2 full->banded step. Must be a multiple of nblk", 0, ELPA_AUTOTUNE_NOT_TUNABLE, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        intermediate_bandwidth_cardinality, intermediate_bandwidth_enumerate, intermediate_bandwidth_is_valid, NULL, PRINT_YES),

        INT_ENTRY("blocking_in_band_to_full", "Loop blocking, default 3", 3, ELPA_AUTOTUNE_EXTENSIVE, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        band_to_full_cardinality, band_to_full_enumerate, band_to_full_is_valid, NULL, PRINT_YES),
        INT_ENTRY("stripewidth_real", "Stripewidth_real, default 48. Must be a multiple of 4", 48, ELPA_AUTOTUNE_EXTENSIVE, ELPA_AUTOTUNE_DOMAIN_REAL, \
                        stripewidth_real_cardinality, stripewidth_real_enumerate, stripewidth_real_is_valid, NULL, PRINT_YES),
        INT_ENTRY("stripewidth_complex", "Stripewidth_complex, default 96. Must be a multiple of 8", 96, ELPA_AUTOTUNE_EXTENSIVE, ELPA_AUTOTUNE_DOMAIN_COMPLEX, \
                        stripewidth_complex_cardinality, stripewidth_complex_enumerate, stripewidth_complex_is_valid, NULL, PRINT_YES),

        INT_ENTRY("max_stored_rows", "Maximum number of stored rows used in ELPA 1 backtransformation, default 63", 63, ELPA_AUTOTUNE_EXTENSIVE, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        max_stored_rows_cardinality, max_stored_rows_enumerate, max_stored_rows_is_valid, NULL, PRINT_YES),
#ifdef WITH_OPENMP_TRADITIONAL
        INT_ENTRY("omp_threads", "OpenMP threads used in ELPA, default 1", 1, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        omp_threads_cardinality, omp_threads_enumerate, omp_threads_is_valid, NULL, PRINT_YES),
#else
        INT_ENTRY("omp_threads", "OpenMP threads used in ELPA, default 1", 1, ELPA_AUTOTUNE_NOT_TUNABLE, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        omp_threads_cardinality, omp_threads_enumerate, omp_threads_is_valid, NULL, PRINT_YES),
#endif
        INT_ENTRY("cannon_buffer_size", "Increasing the buffer size might make it faster, but costs memory", 0, ELPA_AUTOTUNE_NOT_TUNABLE, ELPA_AUTOTUNE_DOMAIN_ANY, \
                        cannon_buffer_size_cardinality, cannon_buffer_size_enumerate, cannon_buffer_size_is_valid, NULL, PRINT_YES),
        //BOOL_ENTRY("qr", "Use QR decomposition, only used for ELPA_SOLVER_2STAGE, real case", 0, ELPA_AUTOTUNE_MEDIUM, ELPA_AUTOTUNE_DOMAIN_REAL),
        BOOL_ENTRY("qr", "Use QR decomposition, only used for ELPA_SOLVER_2STAGE, real case", 0, ELPA_AUTOTUNE_NOT_TUNABLE, ELPA_AUTOTUNE_DOMAIN_REAL, PRINT_YES),
        BOOL_ENTRY("timings", "Enable time measurement", 0, ELPA_AUTOTUNE_NOT_TUNABLE, 0, PRINT_YES),
        BOOL_ENTRY("debug", "Emit verbose debugging messages", 0, ELPA_AUTOTUNE_NOT_TUNABLE, 0, PRINT_YES),
        BOOL_ENTRY("print_flops", "Print FLOP rates on task 0", 0, ELPA_AUTOTUNE_NOT_TUNABLE, 0, PRINT_YES),
        BOOL_ENTRY("measure_performance", "Also measure with flops (via papi) with the timings", 0, ELPA_AUTOTUNE_NOT_TUNABLE, 0, PRINT_YES),
        BOOL_ENTRY("check_pd", "Check eigenvalues to be positive", 0, ELPA_AUTOTUNE_NOT_TUNABLE, 0, PRINT_YES),
        BOOL_ENTRY("output_pinning_information", "Print the pinning information", 0, ELPA_AUTOTUNE_NOT_TUNABLE, 0, PRINT_YES),
        BOOL_ENTRY("cannon_for_generalized", "Whether to use Cannons algorithm for the generalized EVP", 1, ELPA_AUTOTUNE_NOT_TUNABLE, 0, PRINT_YES),
};

#define READONLY_FLOAT_ENTRY(option_name, option_description) \
        { \
                BASE_ENTRY(option_name, option_description, 0, 1, 0) \
        }

#define FLOAT_ENTRY(option_name, option_description, default, print_flag) \
        { \
                BASE_ENTRY(option_name, option_description, 0, 0, print_flag), \
                .default_value = default, \
        }

static const elpa_index_float_entry_t float_entries[] = {
        FLOAT_ENTRY("thres_pd_single", "Threshold to define ill-conditioning, default 0.00001", 0.00001, PRINT_YES),
};

#define READONLY_DOUBLE_ENTRY(option_name, option_description) \
        { \
                BASE_ENTRY(option_name, option_description, 0, 1, 0) \
        }

#define DOUBLE_ENTRY(option_name, option_description, default, print_flag) \
        { \
                BASE_ENTRY(option_name, option_description, 0, 0, print_flag), \
                .default_value = default, \
        }

static const elpa_index_double_entry_t double_entries[] = {
        DOUBLE_ENTRY("thres_pd_double", "Threshold to define ill-conditioning, default 0.00001", 0.00001, PRINT_YES),
};

void elpa_index_free(elpa_index_t index) {
#define FREE_OPTION(TYPE, ...) \
        free(index->TYPE##_options.values); \
        free(index->TYPE##_options.is_set); \
        free(index->TYPE##_options.notified);

        FOR_ALL_TYPES(FREE_OPTION);

        free(index);
}

static int compar(const void *a, const void *b) {
        return strcmp(((elpa_index_int_entry_t *) a)->base.name,
                      ((elpa_index_int_entry_t *) b)->base.name);
}

#define IMPLEMENT_FIND_ENTRY(TYPE, ...) \
        static int find_##TYPE##_entry(char *name) { \
                elpa_index_##TYPE##_entry_t *entry; \
                elpa_index_##TYPE##_entry_t key = { .base = {.name = name} } ; \
                size_t nmembers = nelements(TYPE##_entries); \
                entry = lfind((const void*) &key, (const void *) TYPE##_entries, &nmembers, sizeof(elpa_index_##TYPE##_entry_t), compar); \
                if (entry) { \
                        return (entry - &TYPE##_entries[0]); \
                } else { \
                        return -1; \
                } \
        }
FOR_ALL_TYPES(IMPLEMENT_FIND_ENTRY)


#define IMPLEMENT_GETENV(TYPE, PRINTF_SPEC, ...) \
        static int getenv_##TYPE(elpa_index_t index, const char *env_variable, enum NOTIFY_FLAGS notify_flag, int n, TYPE *value, const char *error_string) { \
                int err; \
                char *env_value = getenv(env_variable); \
                if (env_value) { \
                        err = elpa_##TYPE##_string_to_value(TYPE##_entries[n].base.name, env_value, value); \
                        if (err != ELPA_OK) { \
                                fprintf(stderr, "ELPA: Error interpreting environment variable %s with value '%s': %s\n", \
                                                TYPE##_entries[n].base.name, env_value, elpa_strerr(err)); \
                        } else {\
                                const char *value_string = NULL; \
                                if (elpa_##TYPE##_value_to_string(TYPE##_entries[n].base.name, *value, &value_string) == ELPA_OK) { \
                                        if (!(index->TYPE##_options.notified[n] & notify_flag)) { \
                                                if (elpa_index_is_printing_mpi_rank(index)) { \
                                                        fprintf(stderr, "ELPA: %s '%s' is set to %s due to environment variable %s\n", \
                                                                      error_string, TYPE##_entries[n].base.name, value_string, env_variable); \
                                                } \
                                                index->TYPE##_options.notified[n] |= notify_flag; \
                                        } \
                                } else { \
                                        if (elpa_index_is_printing_mpi_rank(index)) { \
                                                fprintf(stderr, "ELPA: %s '%s' is set to '" PRINTF_SPEC "' due to environment variable %s\n", \
                                                        error_string, TYPE##_entries[n].base.name, *value, env_variable);\
                                        } \
                                } \
                                return 1; \
                        } \
                } \
                return 0; \
        }
FOR_ALL_TYPES(IMPLEMENT_GETENV)


#define IMPLEMENT_GET_FUNCTION(TYPE, PRINTF_SPEC, SCANF_SPEC, ERROR_VALUE) \
        TYPE elpa_index_get_##TYPE##_value(elpa_index_t index, char *name, int *error) { \
                TYPE ret; \
                if (sizeof(TYPE##_entries) == 0) { \
                        return ELPA_ERROR_ENTRY_NOT_FOUND; \
                } \
                int n = find_##TYPE##_entry(name); \
                if (n >= 0) { \
                        int from_env = 0; \
                        if (!TYPE##_entries[n].base.once && !TYPE##_entries[n].base.readonly) { \
                                from_env = getenv_##TYPE(index, TYPE##_entries[n].base.env_force, NOTIFY_ENV_FORCE, n, &ret, "Option"); \
                        } \
                        if (!from_env) { \
                                ret = index->TYPE##_options.values[n]; \
                        } \
                        if (error != NULL) { \
                                *error = ELPA_OK; \
                        } \
                        return ret; \
                } else { \
                        if (error != NULL) { \
                                *error = ELPA_ERROR_ENTRY_NOT_FOUND; \
                        } \
                        return ERROR_VALUE; \
                } \
        }
FOR_ALL_TYPES(IMPLEMENT_GET_FUNCTION)


#define IMPLEMENT_LOC_FUNCTION(TYPE, ...) \
        TYPE* elpa_index_get_##TYPE##_loc(elpa_index_t index, char *name) { \
                if (sizeof(TYPE##_entries) == 0) { \
                        return NULL; \
                } \
                int n = find_##TYPE##_entry(name); \
                if (n >= 0) { \
                        return &index->TYPE##_options.values[n]; \
                } else { \
                        return NULL; \
                } \
        }
FOR_ALL_TYPES(IMPLEMENT_LOC_FUNCTION)


#define IMPLEMENT_SET_FUNCTION(TYPE, PRINTF_SPEC, ...) \
        int elpa_index_set_##TYPE##_value(elpa_index_t index, char *name, TYPE value) { \
                if (sizeof(TYPE##_entries) == 0) { \
                        return ELPA_ERROR_ENTRY_NOT_FOUND; \
                } \
                int n = find_##TYPE##_entry(name); \
                if (n < 0) { \
                        return ELPA_ERROR_ENTRY_NOT_FOUND; \
                }; \
                if (TYPE##_entries[n].valid != NULL) { \
                        if(!TYPE##_entries[n].valid(index, n, value)) { \
                                return ELPA_ERROR_ENTRY_INVALID_VALUE; \
                        }; \
                } \
                if (TYPE##_entries[n].base.once & index->TYPE##_options.is_set[n]) { \
                        return ELPA_ERROR_ENTRY_ALREADY_SET; \
                } \
                if (TYPE##_entries[n].base.readonly) { \
                        return ELPA_ERROR_ENTRY_READONLY; \
                } \
                index->TYPE##_options.values[n] = value; \
                index->TYPE##_options.is_set[n] = 1; \
                return ELPA_OK; \
        }
FOR_ALL_TYPES(IMPLEMENT_SET_FUNCTION)

#define IMPLEMENT_SET_FROM_LOAD_FUNCTION(TYPE, PRINTF_SPEC, ...) \
        int elpa_index_set_from_load_##TYPE##_value(elpa_index_t index, char *name, TYPE value, int explicit) { \
                if (sizeof(TYPE##_entries) == 0) { \
                        return ELPA_ERROR_ENTRY_NOT_FOUND; \
                } \
                int n = find_##TYPE##_entry(name); \
                if (n < 0) { \
                        return ELPA_ERROR_ENTRY_NOT_FOUND; \
                }; \
                index->TYPE##_options.values[n] = value; \
                if(explicit) \
                        index->TYPE##_options.is_set[n] = 1; \
                return ELPA_OK; \
        }
FOR_ALL_TYPES(IMPLEMENT_SET_FROM_LOAD_FUNCTION)


#define IMPLEMENT_IS_SET_FUNCTION(TYPE, ...) \
        int elpa_index_##TYPE##_value_is_set(elpa_index_t index, char *name) { \
                if (sizeof(TYPE##_entries) == 0) { \
                        return ELPA_ERROR_ENTRY_NOT_FOUND; \
                } \
                int n = find_##TYPE##_entry(name); \
                if (n >= 0) { \
                        if (index->TYPE##_options.is_set[n]) { \
                                return 1; \
                        } else { \
                                return 0; \
                        } \
                } else { \
                        return ELPA_ERROR_ENTRY_NOT_FOUND; \
                } \
        }
FOR_ALL_TYPES(IMPLEMENT_IS_SET_FUNCTION)


int elpa_index_value_is_set(elpa_index_t index, char *name) {
        int res = ELPA_ERROR;

#define RET_IF_SET(TYPE, ...) \
        res = elpa_index_##TYPE##_value_is_set(index, name); \
        if (res >= 0) { \
                return res; \
        }

        FOR_ALL_TYPES(RET_IF_SET)

        fprintf(stderr, "ELPA Error: Could not find entry '%s'\n", name);
        return res;
}

int elpa_index_int_is_valid(elpa_index_t index, char *name, int new_value) {
        int n = find_int_entry(name); \
        if (n >= 0) { \
                if (int_entries[n].valid == NULL) {
                        return ELPA_OK;
                } else {
                        return int_entries[n].valid(index, n, new_value) ? ELPA_OK : ELPA_ERROR;
                }
        }
        return ELPA_ERROR_ENTRY_NOT_FOUND;
}

int elpa_int_value_to_string(char *name, int value, const char **string) {
        int n = find_int_entry(name);
        if (n < 0) {
                return ELPA_ERROR_ENTRY_NOT_FOUND;
        }
        if (int_entries[n].to_string == NULL) {
                return ELPA_ERROR_ENTRY_NO_STRING_REPRESENTATION;
        }
        *string = int_entries[n].to_string(value);
        return ELPA_OK;
}


int elpa_int_value_to_strlen(char *name, int value) {
        const char *string = NULL;
        elpa_int_value_to_string(name, value, &string);
        if (string == NULL) {
                return 0;
        } else {
                return strlen(string);
        }
}


int elpa_index_int_value_to_strlen(elpa_index_t index, char *name) {
        int n = find_int_entry(name);
        if (n < 0) {
                return 0;
        }
        return elpa_int_value_to_strlen(name, index->int_options.values[n]);
}


int elpa_int_string_to_value(char *name, char *string, int *value) {
        int n = find_int_entry(name);
        if (n < 0) {
                return ELPA_ERROR_ENTRY_NOT_FOUND;
        }

        if (int_entries[n].to_string == NULL) {
                int val, ret;
                ret = sscanf(string, "%d", &val);
                if (ret == 1) {
                        *value = val;
                        return ELPA_OK;
                } else {
                        return ELPA_ERROR_ENTRY_INVALID_VALUE;
                }
        }

        for (int i = 0; i < int_entries[n].cardinality(NULL); i++) {
                int candidate = int_entries[n].enumerate(NULL, i);
                if (strcmp(string, int_entries[n].to_string(candidate)) == 0) {
                        *value = candidate;
                        return ELPA_OK;
                }
        }
        return ELPA_ERROR_ENTRY_INVALID_VALUE;
}

int elpa_float_string_to_value(char *name, char *string, float *value) {
        float val;
        int ret = sscanf(string, "%lf", &val);
        if (ret == 1) {
                *value = val;
                return ELPA_OK;
        } else {
                /* \todo: remove */
                fprintf(stderr, "ELPA: DEBUG: Could not parse float value '%s' for option '%s'\n", string, name);
                return ELPA_ERROR_ENTRY_INVALID_VALUE;
        }
}

int elpa_float_value_to_string(char *name, float value, const char **string) {
        return ELPA_ERROR_ENTRY_NO_STRING_REPRESENTATION;
}

int elpa_double_string_to_value(char *name, char *string, double *value) {
        double val;
        int ret = sscanf(string, "%lf", &val);
        if (ret == 1) {
                *value = val;
                return ELPA_OK;
        } else {
                /* \todo: remove */
                fprintf(stderr, "ELPA: DEBUG: Could not parse double value '%s' for option '%s'\n", string, name);
                return ELPA_ERROR_ENTRY_INVALID_VALUE;
        }
}

int elpa_double_value_to_string(char *name, double value, const char **string) {
        return ELPA_ERROR_ENTRY_NO_STRING_REPRESENTATION;
}

int elpa_option_cardinality(char *name) {
        int n = find_int_entry(name);
        if (n < 0 || !int_entries[n].cardinality) {
                return ELPA_ERROR_ENTRY_NOT_FOUND;
        }
        return int_entries[n].cardinality(NULL);
}

int elpa_option_enumerate(char *name, int i) {
        int n = find_int_entry(name);
        if (n < 0 || !int_entries[n].enumerate) {
                return 0;
        }
        return int_entries[n].enumerate(NULL, i);
}


/* Helper functions for simple int entries */
static int cardinality_bool(elpa_index_t index) {
        return 2;
}

static int valid_bool(elpa_index_t index, int n, int new_value) {
        return (0 <= new_value) && (new_value < 2);
}

static int enumerate_identity(elpa_index_t index, int i) {
        return i;
}

/* Helper functions for specific options */

#define NAME_CASE(name, value, ...) \
        case value: \
                return #name;

#define VALID_CASE(name, value) \
        case value: \
                return 1;

#define VALID_CASE_3(name, value, available, other_checks) \
        case value: \
                return available && (other_checks(value));

static const char* elpa_matrix_layout_name(int layout) {
	switch(layout) {
		ELPA_FOR_ALL_MATRIX_LAYOUTS(NAME_CASE)
		default:
			return "(Invalid matrix layout)";
	}
}

static int number_of_matrix_layouts(elpa_index_t index) {
        return ELPA_NUMBER_OF_MATRIX_LAYOUTS;
}

static int matrix_layout_enumerate(elpa_index_t index, int i) {
#define OPTION_RANK(name, value, ...) \
        +(value >= sizeof(array_of_size_value)/sizeof(int) ? 0 : 1)

#define EMPTY()
#define DEFER1(m) m EMPTY()
#define EVAL(...) __VA_ARGS__

#define ENUMERATE_CASE(name, value, ...) \
        { const int array_of_size_value[value]; \
        case 0 DEFER1(INNER_ITERATOR)()(OPTION_RANK): \
                return value; }

        switch(i) {
#define INNER_ITERATOR() ELPA_FOR_ALL_MATRIX_LAYOUTS
                EVAL(ELPA_FOR_ALL_MATRIX_LAYOUTS(ENUMERATE_CASE))
#undef INNER_ITERATOR
                default:
                        return 0;
        }
}

static int matrix_layout_is_valid(elpa_index_t index, int n, int new_value) {
        switch(new_value) {
                ELPA_FOR_ALL_MATRIX_LAYOUTS(VALID_CASE)
                default:
                        return 0;
        }
}

static const char* elpa_solver_name(int solver) {
        switch(solver) {
                ELPA_FOR_ALL_SOLVERS(NAME_CASE)
                default:
                        return "(Invalid solver)";
        }
}

static int number_of_solvers(elpa_index_t index) {
        return ELPA_NUMBER_OF_SOLVERS;
}

static int solver_enumerate(elpa_index_t index, int i) {
#define OPTION_RANK(name, value, ...) \
        +(value >= sizeof(array_of_size_value)/sizeof(int) ? 0 : 1)

#define EMPTY()
#define DEFER1(m) m EMPTY()
#define EVAL(...) __VA_ARGS__

#define ENUMERATE_CASE(name, value, ...) \
        { const int array_of_size_value[value]; \
        case 0 DEFER1(INNER_ITERATOR)()(OPTION_RANK): \
                return value; }

        switch(i) {
#define INNER_ITERATOR() ELPA_FOR_ALL_SOLVERS
                EVAL(ELPA_FOR_ALL_SOLVERS(ENUMERATE_CASE))
#undef INNER_ITERATOR
                default:
                        return 0;
        }
}


static int solver_is_valid(elpa_index_t index, int n, int new_value) {
        switch(new_value) {
                ELPA_FOR_ALL_SOLVERS(VALID_CASE)
                default:
                        return 0;
        }
}

static int number_of_real_kernels(elpa_index_t index) {
        return ELPA_2STAGE_NUMBER_OF_REAL_KERNELS;
}

static int real_kernel_enumerate(elpa_index_t index,int i) {
        switch(i) {
#define INNER_ITERATOR() ELPA_FOR_ALL_2STAGE_REAL_KERNELS
                EVAL(ELPA_FOR_ALL_2STAGE_REAL_KERNELS(ENUMERATE_CASE))
#undef INNER_ITERATOR
                default:
                        return 0;
        }
}

static const char *real_kernel_name(int kernel) {
        switch(kernel) {
                ELPA_FOR_ALL_2STAGE_REAL_KERNELS(NAME_CASE)
                default:
                        return "(Invalid real kernel)";
        }
}

#define REAL_GPU_KERNEL_ONLY_WHEN_GPU_IS_ACTIVE(kernel_number) \
        kernel_number == ELPA_2STAGE_REAL_GPU ? gpu_is_active : 1

static int real_kernel_is_valid(elpa_index_t index, int n, int new_value) {
        int solver = elpa_index_get_int_value(index, "solver", NULL);
        if (solver == ELPA_SOLVER_1STAGE) {
                return new_value == ELPA_2STAGE_REAL_DEFAULT;
        }
        int gpu_is_active = elpa_index_get_int_value(index, "gpu", NULL);
        switch(new_value) {
                ELPA_FOR_ALL_2STAGE_REAL_KERNELS(VALID_CASE_3, REAL_GPU_KERNEL_ONLY_WHEN_GPU_IS_ACTIVE)
                default:
                        return 0;
        }
}

static int number_of_complex_kernels(elpa_index_t index) {
        return ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS;
}


static int complex_kernel_enumerate(elpa_index_t index,int i) {
        switch(i) {
#define INNER_ITERATOR() ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS
                EVAL(ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS(ENUMERATE_CASE))
#undef INNER_ITERATOR
                default:
                        return 0;
        }
}

static const char *complex_kernel_name(int kernel) {
        switch(kernel) {
                ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS(NAME_CASE)
                default:
                        return "(Invalid complex kernel)";
        }
}

#define COMPLEX_GPU_KERNEL_ONLY_WHEN_GPU_IS_ACTIVE(kernel_number) \
        kernel_number == ELPA_2STAGE_COMPLEX_GPU ? gpu_is_active : 1

static int complex_kernel_is_valid(elpa_index_t index, int n, int new_value) {
        int solver = elpa_index_get_int_value(index, "solver", NULL);
        if (solver == ELPA_SOLVER_1STAGE) {
                return new_value == ELPA_2STAGE_COMPLEX_DEFAULT;
        }
        int gpu_is_active = elpa_index_get_int_value(index, "gpu", NULL);
        switch(new_value) {
                ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS(VALID_CASE_3, COMPLEX_GPU_KERNEL_ONLY_WHEN_GPU_IS_ACTIVE)
                default:
                        return 0;
        }
}

static const char* elpa_autotune_level_name(int level) {
        switch(level) {
                ELPA_FOR_ALL_AUTOTUNE_LEVELS(NAME_CASE)
                default:
                        return "(Invalid autotune level)";
        }
}

static const char* elpa_autotune_domain_name(int domain) {
        switch(domain) {
                ELPA_FOR_ALL_AUTOTUNE_DOMAINS(NAME_CASE)
                default:
                        return "(Invalid autotune domain)";
        }
}

static int na_is_valid(elpa_index_t index, int n, int new_value) {
        return new_value > 0;
}

static int nev_is_valid(elpa_index_t index, int n, int new_value) {
        if (!elpa_index_int_value_is_set(index, "na")) {
                return 0;
        }
        return 0 <= new_value && new_value <= elpa_index_get_int_value(index, "na", NULL);
}

static int is_positive(elpa_index_t index, int n, int new_value) {
        return new_value > 0;
}

static int bw_is_valid(elpa_index_t index, int n, int new_value) {
        int na;
        if (elpa_index_int_value_is_set(index, "na") != 1) {
                return 0;
        }

        na = elpa_index_get_int_value(index, "na", NULL);
        return (0 <= new_value) && (new_value < na);
}

static int output_build_config_is_valid(elpa_index_t index, int n, int new_value) {
        return new_value == 0 || new_value == 1;
}

static int gpu_is_valid(elpa_index_t index, int n, int new_value) {
        return new_value == 0 || new_value == 1;
}

static int skewsymmetric_is_valid(elpa_index_t index, int n, int new_value) {
        return new_value == 0 || new_value == 1;
}

static int band_to_full_cardinality(elpa_index_t index) {
	return 10;
}
static int band_to_full_enumerate(elpa_index_t index, int i) {
	return i+1;
}

static int internal_nblk_is_valid(elpa_index_t index, int n, int new_value) {
        return (0 <= new_value);
}
static int internal_nblk_cardinality(elpa_index_t index) {
	return 9;
}

static int internal_nblk_enumerate(elpa_index_t index, int i) {
	switch(i) {
	  case 0:
	    return 2;
	  case 1:
	    return 4;
	  case 2:
	    return 8;
	  case 3:
	    return 16;
	  case 4:
	    return 32;
	  case 5:
	    return 64;
	  case 6:
	    return 128;
	  case 7:
	    return 256;
	  case 8:
	    return 1024;
	}
}

// TODO shouldnt it be only for ELPA2??
static int band_to_full_is_valid(elpa_index_t index, int n, int new_value) {
	int max_block=10;
        return (1 <= new_value) && (new_value <= max_block);
}

static int stripewidth_real_cardinality(elpa_index_t index) {
	return 17;
}

static int stripewidth_complex_cardinality(elpa_index_t index) {
	return 17;
}

static int stripewidth_real_enumerate(elpa_index_t index, int i) {
	switch(i) {
	  case 0:
	    return 32;
	  case 1:
	    return 36;
	  case 2:
	    return 40;
	  case 3:
	    return 44;
	  case 4:
	    return 48;
	  case 5:
	    return 52;
	  case 6:
	    return 56;
	  case 7:
	    return 60;
	  case 8:
	    return 64;
	  case 9:
	    return 68;
	  case 10:
	    return 72;
	  case 11:
	    return 76;
	  case 12:
	    return 80;
	  case 13:
	    return 84;
	  case 14:
	    return 88;
	  case 15:
	    return 92;
	  case 16:
	    return 96;
	}
}

static int stripewidth_complex_enumerate(elpa_index_t index, int i) {
	switch(i) {
	  case 0:
	    return 48;
	  case 1:
	    return 56;
	  case 2:
	    return 64;
	  case 3:
	    return 72;
	  case 4:
	    return 80;
	  case 5:
	    return 88;
	  case 6:
	    return 96;
	  case 7:
	    return 104;
	  case 8:
	    return 112;
	  case 9:
	    return 120;
	  case 10:
	    return 128;
	  case 11:
	    return 136;
	  case 12:
	    return 144;
	  case 13:
	    return 152;
	  case 14:
	    return 160;
	  case 15:
	    return 168;
	  case 16:
	    return 176;
	}
}

static int stripewidth_real_is_valid(elpa_index_t index, int n, int new_value) {
	return (32 <= new_value) && (new_value <= 96);
}

static int stripewidth_complex_is_valid(elpa_index_t index, int n, int new_value) {
	return (48 <= new_value) && (new_value <= 176);
}

static int omp_threads_cardinality(elpa_index_t index) {
	int max_threads;
#ifdef WITH_OPENMP_TRADITIONAL
	if (set_max_threads_glob == 0) {
		max_threads_glob = omp_get_max_threads();
		set_max_threads_glob = 1;
	}
#else
	max_threads_glob = 1;
	set_max_threads_glob = 1;
#endif
	max_threads = max_threads_glob;
	return max_threads;
}

static int omp_threads_enumerate(elpa_index_t index, int i) {
        return i + 1;
}

static int omp_threads_is_valid(elpa_index_t index, int n, int new_value) {
        int max_threads;
#ifdef WITH_OPENMP_TRADITIONAL
	if (set_max_threads_glob == 0) {
		max_threads_glob = omp_get_max_threads();
		set_max_threads_glob = 1;
	}
#else
	max_threads_glob = 1;
	set_max_threads_glob = 1;
#endif
	max_threads = max_threads_glob;
        return (1 <= new_value) && (new_value <= max_threads);
}


static int valid_with_gpu(elpa_index_t index, int n, int new_value) {
        int gpu_is_active = elpa_index_get_int_value(index, "gpu", NULL);
        if (gpu_is_active == 1) {
                return ((new_value == 0 ) || (new_value == 1));
        }
        else {
                return new_value == 0;
        }
}

static int valid_with_gpu_elpa1(elpa_index_t index, int n, int new_value) {
        int solver = elpa_index_get_int_value(index, "solver", NULL);
        int gpu_is_active = elpa_index_get_int_value(index, "gpu", NULL);
        if ((solver == ELPA_SOLVER_1STAGE) && (gpu_is_active == 1)) {
                return ((new_value == 0 ) || (new_value == 1));
        }
        else {
                return new_value == 0;
        }
}

static int valid_with_gpu_elpa2(elpa_index_t index, int n, int new_value) {
        int solver = elpa_index_get_int_value(index, "solver", NULL);
        int gpu_is_active = elpa_index_get_int_value(index, "gpu", NULL);
        if ((solver == ELPA_SOLVER_2STAGE) && (gpu_is_active == 1)) {
                return ((new_value == 0 ) || (new_value == 1));
        }
        else {
                return new_value == 0;
        }
}

static int max_stored_rows_cardinality(elpa_index_t index) {
	return 8;
}

static int max_stored_rows_enumerate(elpa_index_t index, int i) {
	switch(i) {
	  case 0:
	    return 15;
	  case 1:
	    return 31;
	  case 2:
	    return 47;
	  case 3:
	    return 63;
	  case 4:
	    return 79;
	  case 5:
	    return 95;
	  case 6:
	    return 111;
	  case 7:
	    return 127;
	}
}

static int max_stored_rows_is_valid(elpa_index_t index, int n, int new_value) {
        int solver = elpa_index_get_int_value(index, "solver", NULL);
        if (solver == ELPA_SOLVER_2STAGE) {
                return new_value == 15;
        } else {
                return (15 <= new_value) && (new_value <= 127);
        }
}

static int use_gpu_id_cardinality(elpa_index_t index) {
#ifdef WITH_GPU_VERSION
	int count;
	count = gpu_count();
        if (count == -1000) {
          fprintf(stderr, "Querrying GPUs failed! Set GPU count = 0\n");
	return 0;
        }
	return count;
#else
	return 0;
#endif
}

static int use_gpu_id_enumerate(elpa_index_t index, int i) {
        fprintf(stderr, "use_gpu_id_is_enumerate should never be called. please report this bug\n");
        return i;
}

static int use_gpu_id_is_valid(elpa_index_t index, int n, int new_value) {
#ifdef WITH_GPU_VERSION
	int count;
	count = gpu_count();
        if (count == -1000) {
          fprintf(stderr, "Querrying GPUs failed! Return with error\n");
	  return 0 == 1 ;
	} else {
          return (0 <= new_value) && (new_value <= count);
	}
#else
	return 0 == 0;
#endif

}

// TODO: this shoudl definitely be improved (too many options to test in autotuning)
static const int TILE_SIZE_STEP = 128;

static int min_tile_size_cardinality(elpa_index_t index) {
        int na;
        if(index == NULL)
                return 0;
        if (elpa_index_int_value_is_set(index, "na") != 1) {
                return 0;
        }
        na = elpa_index_get_int_value(index, "na", NULL);
        return na/TILE_SIZE_STEP;
}

static int min_tile_size_enumerate(elpa_index_t index, int i) {
        return (i+1) * TILE_SIZE_STEP;
}

static int min_tile_size_is_valid(elpa_index_t index, int n, int new_value) {
       return new_value % TILE_SIZE_STEP == 0;
}

static int intermediate_bandwidth_cardinality(elpa_index_t index) {
        int na, nblk;
        if(index == NULL)
                return 0;
        if (elpa_index_int_value_is_set(index, "na") != 1) {
                return 0;
        }
        na = elpa_index_get_int_value(index, "na", NULL);

        if (elpa_index_int_value_is_set(index, "nblk") != 1) {
                return 0;
        }
        nblk = elpa_index_get_int_value(index, "nblk", NULL);

        return na/nblk;
}

static int intermediate_bandwidth_enumerate(elpa_index_t index, int i) {
        int nblk;
        if(index == NULL)
                return 0;
        if (elpa_index_int_value_is_set(index, "nblk") != 1) {
                return 0;
        }
        nblk = elpa_index_get_int_value(index, "nblk", NULL);

        return (i+1) * nblk;
}

static int intermediate_bandwidth_is_valid(elpa_index_t index, int n, int new_value) {
        int na, nblk;
        if (elpa_index_int_value_is_set(index, "na") != 1) {
                return 0;
        }
        na = elpa_index_get_int_value(index, "na", NULL);

        if (elpa_index_int_value_is_set(index, "nblk") != 1) {
                return 0;
        }
        nblk = elpa_index_get_int_value(index, "nblk", NULL);

        int solver = elpa_index_get_int_value(index, "solver", NULL);
        if (solver == ELPA_SOLVER_1STAGE) {
                return new_value == nblk;
        } else {
                if((new_value <= 1 ) || (new_value > na ))
                  return 0;
                if(new_value % nblk != 0) {
                  fprintf(stderr, "intermediate bandwidth has to be multiple of nblk\n");
                  return 0;
                }
        }
}

static int cannon_buffer_size_cardinality(elpa_index_t index) {
        return 2;
}

static int cannon_buffer_size_enumerate(elpa_index_t index, int i) {
        int np_rows;
        if(index == NULL)
                return 0;
        if (elpa_index_int_value_is_set(index, "num_process_rows") != 1) {
                return 0;
        }
        np_rows = elpa_index_get_int_value(index, "num_process_rows", NULL);

        // TODO: 0 is both error code and legal value?
        if(i == 0)
          return 0;
        else
          return np_rows - 1;
}

static int cannon_buffer_size_is_valid(elpa_index_t index, int n, int new_value) {
        int np_rows;
        if(index == NULL)
                return 0;
        if (elpa_index_int_value_is_set(index, "num_process_rows") != 1) {
                return 0;
        }
        np_rows = elpa_index_get_int_value(index, "num_process_rows", NULL);

        return ((new_value >= 0) && (new_value < np_rows));
}

elpa_index_t elpa_index_instance() {
        elpa_index_t index = (elpa_index_t) calloc(1, sizeof(struct elpa_index_struct));

#define ALLOCATE(TYPE, PRINTF_SPEC, ...) \
        index->TYPE##_options.values = (TYPE*) calloc(nelements(TYPE##_entries), sizeof(TYPE)); \
        index->TYPE##_options.is_set = (int*) calloc(nelements(TYPE##_entries), sizeof(int)); \
        index->TYPE##_options.notified = (int*) calloc(nelements(TYPE##_entries), sizeof(int)); \
        for (int n = 0; n < nelements(TYPE##_entries); n++) { \
                TYPE default_value = TYPE##_entries[n].default_value; \
                if (!TYPE##_entries[n].base.once && !TYPE##_entries[n].base.readonly) { \
                        getenv_##TYPE(index, TYPE##_entries[n].base.env_default, NOTIFY_ENV_DEFAULT, n, &default_value, "Default for option"); \
                } \
                index->TYPE##_options.values[n] = default_value; \
        }

        FOR_ALL_TYPES(ALLOCATE)

        return index;
}

static int is_tunable_but_overriden(elpa_index_t index, int i, int autotune_level, int autotune_domain) {
        return (int_entries[i].autotune_level != 0) &&
               (int_entries[i].autotune_level <= autotune_level) &&
               (int_entries[i].autotune_domain & autotune_domain) &&
               (index->int_options.is_set[i]);
}

static int is_tunable(elpa_index_t index, int i, int autotune_level, int autotune_domain) {
        return (int_entries[i].autotune_level != 0) &&
               (int_entries[i].autotune_level <= autotune_level) &&
               (int_entries[i].autotune_domain & autotune_domain) &&
               (!index->int_options.is_set[i]);
}

int elpa_index_autotune_cardinality(elpa_index_t index, int autotune_level, int autotune_domain) {
        int N = 1;

        for (int i = 0; i < nelements(int_entries); i++) { \
                if (is_tunable(index, i, autotune_level, autotune_domain)) {
                        N *= int_entries[i].cardinality(index);
                }
        }
        return N;
}

void elpa_index_print_int_parameter(elpa_index_t index, char* buff, int i)
{
        int value = index->int_options.values[i];
        sprintf(buff, "%s = ", int_entries[i].base.name);
        if (int_entries[i].to_string) {
                sprintf(buff, "%s%d -> %s\n", buff, value, int_entries[i].to_string(value));
        } else {
                sprintf(buff, "%s%d\n", buff, value);
        }
}

int elpa_index_set_autotune_parameters(elpa_index_t index, int autotune_level, int autotune_domain, int current) {
        int current_cpy = current;
        char buff[100];
        int debug = elpa_index_get_int_value(index, "debug", NULL);

        //if(elpa_index_is_printing_mpi_rank(index)) fprintf(stderr, "***Trying a new autotuning index %d\n", current);
        for (int i = 0; i < nelements(int_entries); i++) {
                if (is_tunable(index, i, autotune_level, autotune_domain)) {
                        int value = int_entries[i].enumerate(index, current_cpy % int_entries[i].cardinality(index));
                        //if(elpa_index_is_printing_mpi_rank(index)) fprintf(stderr, "  * val[%d] = %d -> %d\n", i, current_cpy % int_entries[i].cardinality(index), value);
                        /* Try to set option i to that value */
                        if (int_entries[i].valid(index, i, value)) {
                                index->int_options.values[i] = value;
                        } else {
                                //if(elpa_index_is_printing_mpi_rank(index)) fprintf(stderr, "  *NOT VALID becaluse of i %d (%s) and value %d translated to %d\n", i, int_entries[i].base.name, current_cpy % int_entries[i].cardinality(index), value);
                                return 0;
                        }
                        current_cpy /= int_entries[i].cardinality(index);
                }
        }
        if (debug == 1 && elpa_index_is_printing_mpi_rank(index)) {
                fprintf(stderr, "\n*** AUTOTUNING: setting a new combination of parameters, idx %d ***\n", current);
                elpa_index_print_autotune_parameters(index, autotune_level, autotune_domain);
                fprintf(stderr, "***\n\n");
        }

        /* Could set all values */
        return 1;
}

int elpa_index_print_autotune_parameters(elpa_index_t index, int autotune_level, int autotune_domain) {
        char buff[100];
        if (elpa_index_is_printing_mpi_rank(index)) {
                for (int i = 0; i < nelements(int_entries); i++) {
                        if (is_tunable(index, i, autotune_level, autotune_domain)) {
                                elpa_index_print_int_parameter(index, buff, i);
                                fprintf(stderr, "%s", buff);
                        }
                }
        }
        return 1;
}

int elpa_index_print_autotune_state(elpa_index_t index, int autotune_level, int autotune_domain, int min_loc,
                                    double min_val, int current, int cardinality, char* file_name) {
        char buff[100];
        elpa_index_t index_best;
        int min_loc_cpy = min_loc;
        FILE *f;

        // get index with the currently best parameters
        index_best = elpa_index_instance();

        if(min_loc_cpy > -1){
                for (int i = 0; i < nelements(int_entries); i++) {
                        if (is_tunable(index, i, autotune_level, autotune_domain)) {

                                int value = int_entries[i].enumerate(index, min_loc_cpy % int_entries[i].cardinality(index));
                                /* we are setting the value for output only, we do not need to check consistency */
                                index_best->int_options.values[i] = value;
                                min_loc_cpy /= int_entries[i].cardinality(index);
                        }
                }
        }
        if (elpa_index_is_printing_mpi_rank(index)) {
                int output_to_file = (strlen(file_name) > 0);
                if(output_to_file) {
                        f = fopen(file_name, "w");
                        if(f == NULL){
                                fprintf(stderr, "Cannot open file %s in elpa_index_print_autotune_state\n", file_name);
                                return 0;
                        }
                }
                else {
                        f = stdout;
                }

                if(!output_to_file)
                        fprintf(f, "\n");
                fprintf(f, "*** AUTOTUNING STATE ***\n");
                fprintf(f, "** This is the state of the autotuning object\n");
                fprintf(f, "autotune_level = %d -> %s\n", autotune_level, elpa_autotune_level_name(autotune_level));
                fprintf(f, "autotune_domain = %d -> %s\n", autotune_domain, elpa_autotune_domain_name(autotune_domain));
                fprintf(f, "autotune_cardinality = %d\n", cardinality);
                fprintf(f, "current_idx = %d\n", current);
                fprintf(f, "best_idx = %d\n", min_loc);
                fprintf(f, "best_time = %g\n", min_val);
                if(min_loc_cpy > -1) {
                        fprintf(f, "** The following parameters are autotuned with so far the best values\n");
                        for (int i = 0; i < nelements(int_entries); i++) {
                                if (is_tunable(index, i, autotune_level, autotune_domain)) {
                                        elpa_index_print_int_parameter(index_best, buff, i);
                                        fprintf(f, "%s", buff);
                                }
                        }
                        fprintf(f, "** The following parameters would be autotuned on the selected autotuning level, but were overridden by the set() method\n");
                        for (int i = 0; i < nelements(int_entries); i++) {
                                if (is_tunable_but_overriden(index, i, autotune_level, autotune_domain)) {
                                        elpa_index_print_int_parameter(index, buff, i);
                                        fprintf(f, "%s", buff);
                                }
                        }
                }else{
                        fprintf(f, "** No output after first step\n");
                }
                fprintf(f, "*** END OF AUTOTUNING STATE ***\n");

                if(output_to_file)
                        fclose(f);
        }
        elpa_index_free(index_best);

        return 1;
}

const int LEN =1000;

#define IMPLEMENT_LOAD_LINE(TYPE, PRINTF_SPEC, SCANF_SPEC, ...) \
        static int load_##TYPE##_line(FILE* f, const char* expected, TYPE* val) { \
                char line[LEN], s[LEN]; \
                int error = 0; \
                TYPE n; \
                if(fgets(line, LEN, f) == NULL){ \
                        fprintf(stderr, "Loading autotuning state error: line is not there\n"); \
                        error = 1; \
                } else{ \
                        sscanf(line, "%s = " SCANF_SPEC "\n", s, &n); \
                        if(strcmp(s, expected) != 0){ \
                                fprintf(stderr, "Loading autotuning state error: expected %s, got %s\n", expected, s); \
                                error = 1;\
                        } else{ \
                                *val = n; \
                        } \
                } \
                if(error){ \
                        fprintf(stderr, "Autotuning state file corrupted\n"); \
                        return 0; \
                } \
                return 1; \
        }
FOR_ALL_TYPES(IMPLEMENT_LOAD_LINE)

int elpa_index_load_autotune_state(elpa_index_t index, int* autotune_level, int* autotune_domain, int* min_loc,
                                    double* min_val, int* current, int* cardinality, char* file_name) {
        char line[LEN];
        FILE *f;

        //TODO: should be broadcasted, instead of read on all ranks
        //if(elpa_index_is_printing_mpi_rank(index)){
                f = fopen(file_name, "r");

                if (f == NULL) {
                        fprintf(stderr, "Cannont open file %s\n", file_name);
                        return(0);
                }


                if(fgets(line, LEN, f) == NULL) return 0;
                if(fgets(line, LEN, f) == NULL) return 0;
                if(! load_int_line(f, "autotune_level", autotune_level)) return 0;
                if(! load_int_line(f, "autotune_domain", autotune_domain)) return 0;
                if(! load_int_line(f, "autotune_cardinality", cardinality)) return 0;
                if(! load_int_line(f, "current_idx", current)) return 0;
                if(! load_int_line(f, "best_idx", min_loc)) return 0;
                if(! load_double_line(f, "best_time", min_val)) return 0;
                fclose(f);
       // }

        return 1;
}

const char STRUCTURE_PARAMETERS[] = "* Parameters describing structure of the computation:\n";
const char EXPLICIT_PARAMETERS[] = "* Parameters explicitly set by the user:\n";
const char DEFAULT_PARAMETERS[] = "* Parameters with default or environment value:\n";

int elpa_index_print_settings(elpa_index_t index, char *file_name) {
        const int LEN =10000;
        char out_structure[LEN], out_set[LEN], out_defaults[LEN], out_nowhere[LEN], buff[100];
        char (*out)[LEN];
        FILE *f;

        sprintf(out_structure, "%s", STRUCTURE_PARAMETERS);
        sprintf(out_set, "%s", EXPLICIT_PARAMETERS);
        sprintf(out_defaults, "%s", DEFAULT_PARAMETERS);
        sprintf(out_nowhere, "Not to be printed:\n");
        if(elpa_index_is_printing_mpi_rank(index)){
                for (int i = 0; i < nelements(int_entries); i++) {
                        if(int_entries[i].base.print_flag == PRINT_STRUCTURE) {
                                out = &out_structure;
                        } else if(int_entries[i].base.print_flag == PRINT_YES && index->int_options.is_set[i]) {
                                out = &out_set;
                        } else if(int_entries[i].base.print_flag == PRINT_YES && !index->int_options.is_set[i]) {
                                out = &out_defaults;
                        } else
                                out = &out_nowhere;
                        elpa_index_print_int_parameter(index, buff, i);
                        sprintf(*out, "%s%s", *out, buff);
                }
                int output_to_file = (strlen(file_name) > 0);
                if(output_to_file) {
                        f = fopen(file_name, "w");
                        if(f == NULL){
                                fprintf(stderr, "Cannot open file %s in elpa_index_print_settings\n", file_name);
                                return 0;
                        }
                }
                else {
                        f = stdout;
                }

                fprintf(f, "*** ELPA STATE ***\n");
                fprintf(f, "%s%s%s", out_structure, out_set, out_defaults);
                fprintf(f, "*** END OF ELPA STATE ***\n");
                if(output_to_file)
                        fclose(f);
        }

        return 1;
}

int elpa_index_load_settings(elpa_index_t index, char *file_name) {
        const int LEN = 1000;
        char line[LEN], s[LEN];
        int n;
        FILE *f;
        int skip, explicit;

        //TODO: should be broadcasted, instead of read on all ranks
        //if(elpa_index_is_printing_mpi_rank(index)){
                f = fopen(file_name, "r");

                if (f == NULL) {
                        fprintf(stderr, "Cannont open file %s\n", file_name);
                        return(0);
                }

                skip = 1;
                explicit = 0;

                while ((fgets(line, LEN, f)) != NULL) {
                        if(strcmp(line, EXPLICIT_PARAMETERS) == 0){
                                skip = 0;
                                explicit = 1;
                        }
                        if(strcmp(line, DEFAULT_PARAMETERS) == 0){
                                skip = 0;
                                explicit = 0;
                        }

                        if(line[0] != '\n' && line[0] != '*'){
                                sscanf(line, "%s = %d\n", s, &n);
                                if(! skip){
                                        int error = elpa_index_set_from_load_int_value(index, s, n, explicit);
                                }
                        }
                }
                fclose(f);
       // }

        return 1;
}


int elpa_index_is_printing_mpi_rank(elpa_index_t index)
{
  int process_id;
  if(elpa_index_int_value_is_set(index, "process_id")){
    process_id = elpa_index_get_int_value(index, "process_id", NULL);
    return (process_id == 0);
  }
  printf("Warning: process_id not set, printing on all MPI ranks. This can happen with legacy API.");
  return 1;
}
