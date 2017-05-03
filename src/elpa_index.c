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
#include <elpa/elpa.h>
#include "elpa_index.h"

#include <execinfo.h>

static int enumerate_identity(int i);
static int cardinality_bool(void);
static int valid_bool(elpa_index_t index, int n, int new_value);

static int elpa_number_of_solvers();
static int elpa_solver_enumerate(int i);
static int elpa_solver_is_valid(elpa_index_t index, int n, int new_value);
static const char* elpa_solver_name(int solver);

static int elpa_2stage_number_of_real_kernels();
static int elpa_2stage_real_kernel_enumerate(int i);
static int elpa_2stage_real_kernel_is_valid(elpa_index_t index, int n, int new_value);
static const char *elpa_2stage_real_kernel_name(int kernel);

static int elpa_2stage_number_of_complex_kernels();
static int elpa_2stage_complex_kernel_enumerate(int i);
static int elpa_2stage_complex_kernel_is_valid(elpa_index_t index, int n, int new_value);
static const char *elpa_2stage_complex_kernel_name(int kernel);

int elpa_index_double_string_to_value(char *name, char *string, double *value);
int elpa_index_double_value_to_string(char *name, double value, const char **string);

#define BASE_ENTRY(option_name, option_description, once_value, readonly_value) \
                .base = { \
                        .name = option_name, \
                        .description = option_description, \
                        .once = once_value, \
                        .readonly = readonly_value, \
                        .env_default = "ELPA_DEFAULT_" option_name, \
                        .env_force = "ELPA_FORCE_" option_name, \
                }

#define INT_PARAMETER_ENTRY(option_name, option_description) \
        (elpa_index_int_entry_t) { \
                BASE_ENTRY(option_name, option_description, 1, 0), \
        }

#define BOOL_ENTRY(option_name, option_description, default) \
        (elpa_index_int_entry_t) { \
                BASE_ENTRY(option_name, option_description, 0, 0), \
                .default_value = default, \
                .cardinality = cardinality_bool, \
                .enumerate = enumerate_identity, \
                .valid = valid_bool, \
        }

#define INT_LIST_ENTRY(option_name, option_description, default, card_func, enumerate_func, valid_func, to_string_func) \
        (elpa_index_int_entry_t) { \
                BASE_ENTRY(option_name, option_description, 0, 0), \
                .default_value = default, \
                .cardinality = card_func, \
                .enumerate = enumerate_func, \
                .valid = valid_func, \
                .to_string = to_string_func, \
        }

#define INT_ANY_ENTRY(option_name, option_description) \
        (elpa_index_int_entry_t) { \
                BASE_ENTRY(option_name, option_description, 0, 0), \
        }

static const elpa_index_int_entry_t int_entries[] = {
        INT_PARAMETER_ENTRY("na", "Global matrix has size (na * na)"),
        INT_PARAMETER_ENTRY("nev", "Number of eigenvectors to be computed, 0 <= nev <= na"),
        INT_PARAMETER_ENTRY("nblk", "Block size of scalapack block-cyclic distribution"),
        INT_PARAMETER_ENTRY("local_nrows", "Number of matrix rows stored on this process"),
        INT_PARAMETER_ENTRY("local_ncols", "Number of matrix columns stored on this process"),
        INT_PARAMETER_ENTRY("process_row", "Process row number in the 2D domain decomposition"),
        INT_PARAMETER_ENTRY("process_col", "Process column number in the 2D domain decomposition"),
        INT_ANY_ENTRY("mpi_comm_rows", "Communicator for inter-row communication"),
        INT_ANY_ENTRY("mpi_comm_cols", "Communicator for inter-column communication"),
        INT_ANY_ENTRY("mpi_comm_parent", "Parent communicator"),
        INT_LIST_ENTRY("solver", "Solver to use", ELPA_SOLVER_1STAGE, \
                        elpa_number_of_solvers, elpa_solver_enumerate, elpa_solver_is_valid, elpa_solver_name),
        INT_LIST_ENTRY("real_kernel", "Real kernel to use if 'solver' is set to ELPA_SOLVER_2STAGE", ELPA_2STAGE_REAL_DEFAULT, \
                        elpa_2stage_number_of_real_kernels, elpa_2stage_real_kernel_enumerate, \
                        elpa_2stage_real_kernel_is_valid, elpa_2stage_real_kernel_name),
        INT_LIST_ENTRY("complex_kernel", "Complex kernel to use if 'solver' is set to ELPA_SOLVER_2STAGE", ELPA_2STAGE_COMPLEX_DEFAULT, \
                        elpa_2stage_number_of_complex_kernels, elpa_2stage_complex_kernel_enumerate, \
                        elpa_2stage_complex_kernel_is_valid, elpa_2stage_complex_kernel_name),
        BOOL_ENTRY("qr", "Use QR decomposition, only used for ELPA_SOLVER_2STAGE, real case", 0),
        BOOL_ENTRY("gpu", "Use GPU acceleration", 0),
        BOOL_ENTRY("summary_timings", "Emit some debugging timing informaion", 0),
        BOOL_ENTRY("debug", "Emit verbose debugging messages", 0),
};

#define READONLY_DOUBLE_ENTRY(option_name, option_description) \
        (elpa_index_double_entry_t) { \
                BASE_ENTRY(option_name, option_description, 0, 1) \
        }

static const elpa_index_double_entry_t double_entries[] = {
        READONLY_DOUBLE_ENTRY("time_evp_fwd", "Time for forward step [s]"),
        READONLY_DOUBLE_ENTRY("time_evp_solve", "Time for solve step [s]"),
        READONLY_DOUBLE_ENTRY("time_evp_back", "Time for back-substitution step [s]"),
};

void elpa_index_free(elpa_index_t index) {
#define FREE_OPTION(TYPE, ...) \
        free(index->TYPE##_options.values); \
        free(index->TYPE##_options.is_set); \
        free(index->TYPE##_options.notified);

        FOR_ALL_TYPES(FREE_OPTION);

        free(index);
}

static int compar(const void *key, const void *member) {
        const char *name = (const char *) key;
        elpa_index_int_entry_t *entry = (elpa_index_int_entry_t *) member;

        int l1 = strlen(entry->base.name);
        int l2 = strlen(name);
        if (l1 != l2) {
                return 1;
        }
        if (strncmp(name, entry->base.name, l1) == 0) {
                return 0;
        } else {
                return 1;
        }
}

#define IMPLEMENT_FIND_ENTRY(TYPE, ...) \
        static int find_##TYPE##_entry(char *name) { \
                elpa_index_##TYPE##_entry_t *entry; \
                size_t nmembers = nelements(TYPE##_entries); \
                entry = lfind((const void*) name, (const void *) TYPE##_entries, &nmembers, sizeof(elpa_index_##TYPE##_entry_t), compar); \
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
                        err = elpa_index_##TYPE##_string_to_value(TYPE##_entries[n].base.name, env_value, value); \
                        if (err != ELPA_OK) { \
                                fprintf(stderr, "ELPA: Error interpreting environment variable %s with value '%s': %s\n", \
                                                TYPE##_entries[n].base.name, env_value, elpa_strerr(err)); \
                        } else {\
                                const char *value_string = NULL; \
                                if (elpa_index_##TYPE##_value_to_string(TYPE##_entries[n].base.name, *value, &value_string) == ELPA_OK) { \
                                        if (!(index->TYPE##_options.notified[n] & notify_flag)) { \
                                                fprintf(stderr, "ELPA: %s '%s' is set to %s due to environment variable %s\n", \
                                                                error_string, TYPE##_entries[n].base.name, value_string, env_variable); \
                                                index->TYPE##_options.notified[n] |= notify_flag; \
                                        } \
                                } else { \
                                        fprintf(stderr, "ELPA: %s '%s' is set to '" PRINTF_SPEC "' due to environment variable %s\n", \
                                                        error_string, TYPE##_entries[n].base.name, *value, env_variable);\
                                } \
                                return 1; \
                        } \
                } \
                return 0; \
        }
FOR_ALL_TYPES(IMPLEMENT_GETENV)


#define IMPLEMENT_GET_FUNCTION(TYPE, PRINTF_SPEC, ERROR_VALUE) \
        TYPE elpa_index_get_##TYPE##_value(elpa_index_t index, char *name, int *error) { \
                char *env_value = NULL; \
                TYPE ret; \
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
                        } else { \
                                fprintf(stderr, "elpa_index_get_" #TYPE "(): No entry '%s' and you did not check for errors!\n", name); \
                        } \
                        return ERROR_VALUE; \
                } \
        }
FOR_ALL_TYPES(IMPLEMENT_GET_FUNCTION)


#define IMPLEMENT_LOC_FUNCTION(TYPE, ...) \
        TYPE* elpa_index_get_##TYPE##_loc(elpa_index_t index, char *name) { \
                int n = find_##TYPE##_entry(name); \
                if (n >= 0) { \
                        return &index->TYPE##_options.values[n]; \
                } else { \
                        return NULL; \
                } \
        }
FOR_ALL_TYPES(IMPLEMENT_LOC_FUNCTION)


#define IMPLEMENT_SET_FUNCTION(TYPE, ...) \
        int elpa_index_set_##TYPE##_value(elpa_index_t index, char *name, TYPE value) { \
                int n = find_##TYPE##_entry(name); \
                if (n < 0) { \
                        return ELPA_ERROR_ENTRY_NOT_FOUND; \
                }; \
                if (TYPE##_entries[n].valid != NULL) { \
                        if(!TYPE##_entries[n].valid(index, n, value)) { \
                                return ELPA_ERROR_INVALID_VALUE; \
                        }; \
                } \
                if (TYPE##_entries[n].base.once & index->TYPE##_options.is_set[n]) { \
                        return ELPA_ERROR_VALUE_ALREADY_SET; \
                } \
                index->TYPE##_options.values[n] = value; \
                index->TYPE##_options.is_set[n] = 1; \
                return ELPA_OK; \
        }
FOR_ALL_TYPES(IMPLEMENT_SET_FUNCTION)


#define IMPLEMENT_IS_SET_FUNCTION(TYPE, ...) \
        int elpa_index_##TYPE##_value_is_set(elpa_index_t index, char *name) { \
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

        printf("ERROR: Could not find entry '%s'\n");
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

int elpa_index_int_value_to_string(char *name, int value, const char **string) {
        int n = find_int_entry(name);
        if (n < 0) {
                return ELPA_ERROR_ENTRY_NOT_FOUND;
        }
        if (int_entries[n].to_string == NULL) {
                return ELPA_ERROR_NO_STRING_REPRESENTATION;
        }
        *string = int_entries[n].to_string(value);
        return ELPA_OK;
}

const char *elpa_index_int_value_to_string_helper(char *name, int value) {
        const char *string = NULL;
        elpa_index_int_value_to_string(name, value, &string);
        if (string == NULL) {
                string = "";
        }
        return string;
}

int elpa_index_int_string_to_value(char *name, char *string, int *value) {
        int n = find_int_entry(name);
        if (n < 0) {
                return ELPA_ERROR_ENTRY_NOT_FOUND;
        }

        if (int_entries[n].to_string == NULL) {
                int val, ret;
                ret = sscanf(string, "%d", &val);
                if (ret == strlen(string)) {
                        *value = val;
                        return ELPA_OK;
                } else {
                        return ELPA_ERROR_INVALID_VALUE;
                }
        }

        for (int i = 0; i < int_entries[n].cardinality(); i++) {
                int candidate = int_entries[n].enumerate(i);
                if (strcmp(string, int_entries[n].to_string(candidate)) == 0) {
                        *value = candidate;
                        return ELPA_OK;
                }
        }
        return ELPA_ERROR_INVALID_VALUE;
}

int elpa_index_double_string_to_value(char *name, char *string, double *value) {
        double val;
        int ret = sscanf(string, "%lf", &val);
        if (ret == strlen(string)) {
                *value = val;
                return ELPA_OK;
        } else {
                /* \todo: remove */
                fprintf(stderr, "ELPA: DEBUG: Could not parse double value '%s' for option '%s'\n", string, name);
                return ELPA_ERROR_INVALID_VALUE;
        }
}

int elpa_index_double_value_to_string(char *name, double value, const char **string) {
        return ELPA_ERROR_NO_STRING_REPRESENTATION;
}

int elpa_index_cardinality(elpa_index_t index, char *name) {
        int n = find_int_entry(name);
        if (n < 0 || !int_entries[n].cardinality) {
                return ELPA_ERROR_ENTRY_NOT_FOUND;
        }
        return int_entries[n].cardinality();
}

int elpa_index_enumerate(elpa_index_t index, char *name, int i) {
        int n = find_int_entry(name);
        if (n < 0 || !int_entries[n].enumerate) {
                return 0;
        }
        return int_entries[n].enumerate(i);
}



/* Helper functions for simple int entries */
static int cardinality_bool(void) {
        return 2;
}

static int valid_bool(elpa_index_t index, int n, int new_value) {
        return (0 <= new_value) && (new_value < 2);
}

static int enumerate_identity(int i) {
        return i;
}

/* Helper functions for specific options */

#define NAME_CASE(name, value, ...) \
        case value: \
                return #name;

#define VALID_CASE(name, value) \
        case value: \
                return 1;

#define VALID_CASE_3(name, value, available) \
        case value: \
                return available;

static const char* elpa_solver_name(int solver) {
        switch(solver) {
                ELPA_FOR_ALL_SOLVERS(NAME_CASE)
                default:
                        return "(Invalid solver)";
        }
}

static int elpa_number_of_solvers() {
        return ELPA_NUMBER_OF_SOLVERS;
}

static int elpa_solver_enumerate(int i) {
#define OPTION_RANK(name, value, ...) \
        +(value > sizeof(array_of_size_value)/sizeof(int) ? 1 : 0)

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


static int elpa_solver_is_valid(elpa_index_t index, int n, int new_value) {
        switch(new_value) {
                ELPA_FOR_ALL_SOLVERS(VALID_CASE)
                default:
                        return 0;
        }
}

static int elpa_2stage_number_of_real_kernels() {
        return ELPA_2STAGE_NUMBER_OF_REAL_KERNELS;
}

static int elpa_2stage_real_kernel_enumerate(int i) {
        switch(i) {
#define INNER_ITERATOR() ELPA_FOR_ALL_2STAGE_REAL_KERNELS
                EVAL(ELPA_FOR_ALL_2STAGE_REAL_KERNELS(ENUMERATE_CASE))
#undef INNER_ITERATOR
                default:
                        return 0;
        }
}

static const char *elpa_2stage_real_kernel_name(int kernel) {
        switch(kernel) {
                ELPA_FOR_ALL_2STAGE_REAL_KERNELS(NAME_CASE)
                default:
                        return "(Invalid real kernel)";
        }
}

static int elpa_2stage_real_kernel_is_valid(elpa_index_t index, int n, int new_value) {
        switch(new_value) {
                ELPA_FOR_ALL_2STAGE_REAL_KERNELS(VALID_CASE_3)
                default:
                        return 0;
        }
}

static int elpa_2stage_number_of_complex_kernels() {
        return ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS;
}


static int elpa_2stage_complex_kernel_enumerate(int i) {
        switch(i) {
#define INNER_ITERATOR() ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS
                EVAL(ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS(ENUMERATE_CASE))
#undef INNER_ITERATOR
                default:
                        return 0;
        }
}

static const char *elpa_2stage_complex_kernel_name(int kernel) {
        switch(kernel) {
                ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS(NAME_CASE)
                default:
                        return "(Invalid complex kernel)";
        }
}

static int elpa_2stage_complex_kernel_is_valid(elpa_index_t index, int n, int new_value) {
        switch(new_value) {
                ELPA_FOR_ALL_2STAGE_COMPLEX_KERNELS(VALID_CASE_3)
                default:
                        return 0;
        }
}

elpa_index_t elpa_index_instance() {
        char *env_value = NULL;
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

