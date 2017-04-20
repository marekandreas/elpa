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

elpa_index_t elpa_allocate_index(int n_entries, elpa_int_entry_t int_entries[n_entries]) {
        elpa_index_t index = (elpa_index_t) calloc(1, sizeof(struct elpa_index_struct));

        /* Integer entries */
        index->n_int_entries = n_entries;
        index->int_values = (int*) calloc(index->n_int_entries, sizeof(int));
        index->int_entries = int_entries;
        for (int i = 0; i < index->n_int_entries; i++) {
                index->int_values[i] = int_entries[i].default_value;
        }

        return index;
}

void elpa_free_index(elpa_index_t index) {
        free(index->int_values);
        free(index);
}

static int compar(const void *key, const void *member) {
        const char *name = (const char *) key;
        elpa_int_entry_t *entry = (elpa_int_entry_t *) member;

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
        elpa_int_entry_t *entry;
        size_t nmembers = index->n_int_entries;
        entry = lfind((const void*) name, (const void *) index->int_entries, &nmembers, sizeof(elpa_int_entry_t), compar);
        if (entry) {
                return (entry - &index->int_entries[0]);
        } else {
                return -1;
        }
}

int elpa_get_int_entry(elpa_index_t index, const char *name, int *success) {
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
                        fprintf(stderr, "ELPA: No such entry '%s' and you did not check for errors, returning ELPA_INVALID_INT!\n", name);
                }
                return ELPA_INVALID_INT;
        }
}

int elpa_set_int_entry(elpa_index_t index, const char *name, int value) {
        int n = find_int_entry(index, name);
        int res = ELPA_ERROR;
        if (n >= 0) {
                res = index->int_entries[n].valid(index, value);
                if (res == ELPA_OK) {
                        index->int_values[n] = value;
                }
        }
        return res;
}
