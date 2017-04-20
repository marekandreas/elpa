#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <search.h>
#include <elpa/elpa.h>

#define nelements(x) (sizeof(x)/sizeof(x[0]))

/* A simple structure for storing values to a pre-set
 * number of keys */

/* Forward declaration of configuration structure */
typedef struct elpa_index_struct* elpa_index_t;

/* Function pointer type for the cardinality */
typedef int (*cardinality_t)();

/* Function pointer type to enumerate all possible values (if possible) */
typedef const int (*enumerate_int_option_t)(unsigned int n);

/* Function pointer type check validity of option */
typedef int (*valid_int_t)(elpa_index_t index, int value);

typedef struct {
        const char *name;
        int default_value;
        cardinality_t cardinality;
        enumerate_int_option_t enumerate_option;
        valid_int_t valid;
} elpa_int_entry_t;

struct elpa_index_struct {
        int n_int_entries;
        int *int_values;
        elpa_int_entry_t *int_entries;
};

elpa_index_t elpa_allocate_index(int n_entries, elpa_int_entry_t int_entries[n_entries]);

/*
 !f> interface
 !f>   subroutine elpa_free_index(index) bind(C, name="elpa_free_index")
 !f>     import c_ptr
 !f>     type(c_ptr), value :: index
 !f>   end subroutine
 !f> end interface
 */
void elpa_free_index(elpa_index_t index);

/*
 !f> interface
 !f>   function elpa_get_int_entry(index, name, success) result(value) bind(C, name="elpa_get_int_entry")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                 :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     integer(kind=c_int)                :: value
 !f>     integer(kind=c_int), optional      :: success
 !f>   end function
 !f> end interface
 */
int elpa_get_int_entry(elpa_index_t index, const char *name, int *success);

/*
 !f> interface
 !f>   function elpa_set_int_entry(index, name, value) result(success) bind(C, name="elpa_set_int_entry")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     integer(kind=c_int),intent(in), value :: value
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_set_int_entry(elpa_index_t index, const char *name, int value);
