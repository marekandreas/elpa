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

/* Function type for the cardinality */
typedef int (*elpa_index_cardinality_t)(elpa_index_t index, int n);

/* Function type to enumerate all possible values (if possible) */
typedef int (*elpa_index_enumerate_int_option_t)(elpa_index_t index, int n, unsigned int i);

/* Function type to check the validity of a value */
typedef int (*elpa_index_valid_int_t)(elpa_index_t index, int n, int new_value);

/* Function type to give a string representation of a value */
typedef const char* (*elpa_index_repr_int_t)(elpa_index_t index, int n);


typedef struct {
        const char *name;
        int default_value;
        elpa_index_cardinality_t cardinality;
        elpa_index_enumerate_int_option_t enumerate_option;
        elpa_index_valid_int_t valid;
        elpa_index_repr_int_t repr;

        /* For simple, fixed range options: */
        int fixed_range;
        const char **fixed_range_reprs;

} elpa_index_int_entry_t;

typedef struct {
        const char *name;
        double value;
} elpa_index_double_entry_t;

struct elpa_index_struct {
        int n_int_entries;
        int *int_values;
        int *int_value_is_set;
        elpa_index_int_entry_t *int_entries;

        int n_double_entries;
        double *double_values;
        int *double_value_is_set;
        elpa_index_double_entry_t *double_entries;
};

elpa_index_t elpa_index_allocate(
                int n_int_entries, elpa_index_int_entry_t int_entries[n_int_entries],
                int n_double_entries, elpa_index_double_entry_t double_entries[n_double_entries]);

/*
 !f> interface
 !f>   subroutine elpa_index_free(index) bind(C, name="elpa_index_free")
 !f>     import c_ptr
 !f>     type(c_ptr), value :: index
 !f>   end subroutine
 !f> end interface
 */
void elpa_index_free(elpa_index_t index);


/*
 !f> interface
 !f>   function elpa_index_get_int_value(index, name, success) result(value) bind(C, name="elpa_index_get_int_value")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                 :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     integer(kind=c_int)                :: value
 !f>     integer(kind=c_int), optional      :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_get_int_value(elpa_index_t index, const char *name, int *success);


/*
 !f> interface
 !f>   function elpa_index_get_int_loc(index, name) result(loc) bind(C, name="elpa_index_get_int_loc")
 !f>     import c_ptr, c_char
 !f>     type(c_ptr), value                 :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     type(c_ptr)                        :: loc
 !f>   end function
 !f> end interface
 */
int* elpa_index_get_int_loc(elpa_index_t index, const char *name);


/*
 !f> interface
 !f>   function elpa_index_set_int_value(index, name, value) result(success) bind(C, name="elpa_index_set_int_value")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     integer(kind=c_int),intent(in), value :: value
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_set_int_value(elpa_index_t index, const char *name, int value);


/*
 !f> interface
 !f>   function elpa_index_int_value_is_set(index, name) result(success) bind(C, name="elpa_index_int_value_is_set")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_int_value_is_set(elpa_index_t index, const char *name);


/*
 !f> interface
 !f>   function elpa_index_get_double_value(index, name, success) result(value) bind(C, name="elpa_index_get_double_value")
 !f>     import c_ptr, c_int, c_double, c_char
 !f>     type(c_ptr), value                 :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     real(kind=c_double)                :: value
 !f>     integer(kind=c_int), optional      :: success
 !f>   end function
 !f> end interface
 */
double elpa_index_get_double_value(elpa_index_t index, const char *name, int *success);


/*
 !f> interface
 !f>   function elpa_index_get_double_loc(index, name) result(loc) bind(C, name="elpa_index_get_double_loc")
 !f>     import c_ptr, c_char
 !f>     type(c_ptr), value                 :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     type(c_ptr)                        :: loc
 !f>   end function
 !f> end interface
 */
double* elpa_index_get_double_loc(elpa_index_t index, const char *name);


/*
 !f> interface
 !f>   function elpa_index_set_double_value(index, name, value) result(success) bind(C, name="elpa_index_set_double_value")
 !f>     import c_ptr, c_int, c_double, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     real(kind=c_double),intent(in), value :: value
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_set_double_value(elpa_index_t index, const char *name, double value);
