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

#include "elpa_index.h"
#include "elpa_constants_private.h"

elpa_index_t elpa_index_allocate(
                int n_int_entries, elpa_index_int_entry_t int_entries[n_int_entries],
                int n_double_entries, elpa_index_double_entry_t double_entries[n_double_entries]) {

        elpa_index_t index = (elpa_index_t) calloc(1, sizeof(struct elpa_index_struct));

        /* Integer entries */
        index->n_int_entries = n_int_entries;
        index->int_values = (int*) calloc(index->n_int_entries, sizeof(int));
        index->int_value_is_set = (int*) calloc(index->n_int_entries, sizeof(int));
        index->int_entries = int_entries;
        for (int i = 0; i < index->n_int_entries; i++) {
                index->int_values[i] = int_entries[i].default_value;
        }

        /* Double entries */
        index->n_double_entries = n_double_entries;
        index->double_values = (double*) calloc(index->n_double_entries, sizeof(double));
        index->double_value_is_set = (int*) calloc(index->n_int_entries, sizeof(int));
        index->double_entries = double_entries;
        for (int i = 0; i < index->n_double_entries; i++) {
                index->double_values[i] = 0.0;
        }

        return index;
}

void elpa_index_free(elpa_index_t index) {
        free(index->int_values);
        free(index->int_value_is_set);
        free(index->double_values);
        free(index->double_value_is_set);
        free(index);
}

static int compar(const void *key, const void *member) {
        const char *name = (const char *) key;
        elpa_index_int_entry_t *entry = (elpa_index_int_entry_t *) member;

        int l1 = strlen(entry->name);
        int l2 = strlen(name);
        if (l1 != l2) {
                return 1;
        }
        if (strncmp(name, entry->name, l1) == 0) {
                return 0;
        } else {
                return 1;
        }
}

static int find_int_entry(elpa_index_t index, const char *name) {
        elpa_index_int_entry_t *entry;
        size_t nmembers = index->n_int_entries;
        entry = lfind((const void*) name, (const void *) index->int_entries, &nmembers, sizeof(elpa_index_int_entry_t), compar);
        if (entry) {
                return (entry - &index->int_entries[0]);
        } else {
                return -1;
        }
}

int elpa_index_get_int_value(elpa_index_t index, const char *name, int *success) {
        int n = find_int_entry(index, name);
        if (n >= 0) {
                if (success != NULL) {
                        *success = ELPA_OK;
                }
                 return index->int_values[n];
        } else {
                if (success != NULL) {
                        *success = ELPA_ERROR;
                } else {
                        fprintf(stderr, "ELPA: No such entry '%s' and you did not check for errors, returning -1!\n", name);
                }
                return -1;
        }
}

int* elpa_index_get_int_loc(elpa_index_t index, const char *name) {
        int n = find_int_entry(index, name);
        if (n >= 0) {
                return &index->int_values[n];
        } else {
                return NULL;
        }
}

int elpa_index_set_int_value(elpa_index_t index, const char *name, int value) {
        int n = find_int_entry(index, name);
        int res = ELPA_ERROR;
        if (n >= 0) {
                res = index->int_entries[n].valid(index, n, value);
                if (res == ELPA_OK) {
                        index->int_values[n] = value;
                        index->int_value_is_set[n] = 1;
                }
        }
        return res;
}

int elpa_index_int_value_is_set(elpa_index_t index, const char *name) {
        int n = find_int_entry(index, name);
        if (n >= 0) {
                if (index->int_value_is_set[n]) {
                        return ELPA_OK;
                } else {
                        return ELPA_NO;
                }
        } else {
                return ELPA_ERROR;
        }
}


static const char *invalid_name = "(INVALID NAME)";
static const char *invalid_value = "(INVALID VALUE)";

const char* elpa_index_get_int_repr(elpa_index_t index, const char *name) {
        int n = find_int_entry(index, name);
        if (n >= 0) {
                if (index->int_entries[n].fixed_range_reprs != NULL) {
                        return index->int_entries[n].repr(index, n);
                } else {
                        return NULL;
                }
        } else {
                return invalid_name;
        }
}

/* Helper functions for simple int entries */

static int is_in_fixed_range(elpa_index_t index, int n, int new_value) {
        if (new_value >= 0 && new_value < index->int_entries[n].cardinality(index, n)) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

static int enumerate_fixed_range(elpa_index_t index, int n, unsigned int i) {
        return i;
}

static int is_in_fixed_range_plus_one(elpa_index_t index, int n, int new_value) {
        if (new_value >= 1 && new_value <= index->int_entries[n].cardinality(index, n)) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

static int enumerate_fixed_range_plus_one(elpa_index_t index, int n, unsigned int i) {
        return i + 1;
}

static int cardinality_fixed_range(elpa_index_t index, int n) {
        return index->int_entries[n].fixed_range;
}


static const char* repr_fixed_range(elpa_index_t index, int n) {
        int value = index->int_values[n];
        return index->int_entries[n].fixed_range_reprs[value];
}


/* The single instance */

#define BOOL_ENTRY(option_name, default) \
        (elpa_index_int_entry_t) { \
                .name = option_name, \
                .default_value = default, \
                .cardinality = cardinality_fixed_range, \
                .enumerate_option = enumerate_fixed_range, \
                .valid = is_in_fixed_range, \
                .fixed_range = 2, \
        }

#define INT_LIST_ENTRY(option_name, default, range, reprs) \
        (elpa_index_int_entry_t) { \
                .name = option_name, \
                .default_value = default, \
                .cardinality = cardinality_fixed_range, \
                .enumerate_option = enumerate_fixed_range, \
                .valid = is_in_fixed_range, \
                .repr = repr_fixed_range, \
                .fixed_range = range, \
                .fixed_range_reprs = reprs, \
        }

#define INT_ANY_ENTRY(option_name) \
        (elpa_index_int_entry_t) { \
                .name = option_name, \
        }

static elpa_index_int_entry_t elpa_index_int_entries[] = {
        BOOL_ENTRY("summary_timings", 0),
        BOOL_ENTRY("debug", 0),
        BOOL_ENTRY("qr", 0),
        BOOL_ENTRY("gpu", 0),
        INT_LIST_ENTRY("solver", ELPA_SOLVER_1STAGE, ELPA_NUMBER_OF_SOLVERS, ELPA_SOLVER_NAMES),
        INT_LIST_ENTRY("real_kernel", ELPA_2STAGE_REAL_DEFAULT, ELPA_2STAGE_NUMBER_OF_REAL_KERNELS, ELPA_2STAGE_REAL_KERNEL_NAMES),
        INT_LIST_ENTRY("complex_kernel", ELPA_2STAGE_COMPLEX_DEFAULT, ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS, ELPA_2STAGE_COMPLEX_KERNEL_NAMES),
        INT_ANY_ENTRY("mpi_comm_rows"),
        INT_ANY_ENTRY("mpi_comm_cols"),
        INT_ANY_ENTRY("mpi_comm_parent"),
};

#define DOUBLE_ENTRY(option_name) \
        (elpa_index_double_entry_t) { \
                .name = option_name, \
        }

static elpa_index_double_entry_t elpa_index_double_entries[] = {
        DOUBLE_ENTRY("time_evp_fwd"),
        DOUBLE_ENTRY("time_evp_solve"),
        DOUBLE_ENTRY("time_evp_back"),
};

/*
 !f> interface
 !f>   function elpa_index_instance() result(index) bind(C, name="elpa_index_instance")
 !f>     import c_ptr
 !f>     type(c_ptr) :: index
 !f>   end function
 !f> end interface
 */
elpa_index_t elpa_index_instance() {
        elpa_index_t index = elpa_index_allocate(
                nelements(elpa_index_int_entries), elpa_index_int_entries,
                nelements(elpa_index_double_entries), elpa_index_double_entries
        );
}
