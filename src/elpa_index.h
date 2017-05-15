#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <search.h>
#include <math.h>

#include <elpa/elpa.h>

#define nelements(x) (sizeof(x)/sizeof(x[0]))

#define FOR_ALL_TYPES(X) \
        X(int, "%d", -1) \
        X(double, "%g", NAN)

/* A simple structure for storing values to a pre-set
 * number of keys */

/* Forward declaration of configuration structure */
typedef struct elpa_index_struct* elpa_index_t;

/* Function type for the cardinality */
typedef int (*elpa_index_cardinality_t)(void);

/* Function type to enumerate all possible values, starting from 0 */
typedef int (*elpa_index_enumerate_int_option_t)(int i);

/* Function types to check the validity of a value */
typedef int (*elpa_index_valid_int_t)(elpa_index_t index, int n, int new_value);
typedef int (*elpa_index_valid_double_t)(elpa_index_t index, int n, double new_value);

/* Function type to give a string representation of a value */
typedef const char* (*elpa_index_to_string_int_t)(int n);


typedef struct {
        char *name;
        char *description;
        char *env_default;
        char *env_force;
        int once;
        int readonly;
} elpa_index_entry_t;


typedef struct {
        elpa_index_entry_t base;
        int default_value;
        elpa_index_valid_int_t valid;
        elpa_index_cardinality_t cardinality;
        elpa_index_enumerate_int_option_t enumerate;
        elpa_index_to_string_int_t to_string;
} elpa_index_int_entry_t;


typedef struct {
        elpa_index_entry_t base;
        double default_value;
        elpa_index_valid_double_t valid;
} elpa_index_double_entry_t;

enum NOTIFY_FLAGS {
        NOTIFY_ENV_DEFAULT = (1<<0),
        NOTIFY_ENV_FORCE   = (1<<1),
};

struct elpa_index_struct {
#define STRUCT_MEMBERS(TYPE, ...) \
        struct { \
        TYPE *values; \
        int *is_set; \
        int *notified; \
        } TYPE##_options;
        FOR_ALL_TYPES(STRUCT_MEMBERS)
};


/*
 !f> interface
 !f>   function elpa_index_instance_c() result(index) bind(C, name="elpa_index_instance")
 !f>     import c_ptr
 !f>     type(c_ptr) :: index
 !f>   end function
 !f> end interface
 */
elpa_index_t elpa_index_instance();


/*
 !f> interface
 !f>   subroutine elpa_index_free_c(index) bind(C, name="elpa_index_free")
 !f>     import c_ptr
 !f>     type(c_ptr), value :: index
 !f>   end subroutine
 !f> end interface
 */
void elpa_index_free(elpa_index_t index);


/*
 !f> interface
 !f>   function elpa_index_get_int_value_c(index, name, success) result(value) bind(C, name="elpa_index_get_int_value")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                         :: index
 !f>     character(kind=c_char), intent(in)         :: name(*)
 !f>     integer(kind=c_int), intent(out), optional :: success
 !f>     integer(kind=c_int)                        :: value
 !f>   end function
 !f> end interface
 */
int elpa_index_get_int_value(elpa_index_t index, char *name, int *success);


/*
 !f> interface
 !f>   function elpa_index_set_int_value_c(index, name, value) result(success) bind(C, name="elpa_index_set_int_value")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     integer(kind=c_int),intent(in), value :: value
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_set_int_value(elpa_index_t index, char *name, int value);


/*
 !f> interface
 !f>   function elpa_index_int_value_is_set_c(index, name) result(success) bind(C, name="elpa_index_int_value_is_set")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_int_value_is_set(elpa_index_t index, char *name);


/*
 !f> interface
 !f>   function elpa_index_get_int_loc_c(index, name) result(loc) bind(C, name="elpa_index_get_int_loc")
 !f>     import c_ptr, c_char
 !f>     type(c_ptr), value                 :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     type(c_ptr)                        :: loc
 !f>   end function
 !f> end interface
 */
int* elpa_index_get_int_loc(elpa_index_t index, char *name);


/*
 !f> interface
 !f>   function elpa_index_get_double_value_c(index, name, success) result(value) bind(C, name="elpa_index_get_double_value")
 !f>     import c_ptr, c_int, c_double, c_char
 !f>     type(c_ptr), value                              :: index
 !f>     character(kind=c_char), intent(in)              :: name(*)
 !f>     integer(kind=c_int), intent(out), optional      :: success
 !f>     real(kind=c_double)                             :: value
 !f>   end function
 !f> end interface
 */
double elpa_index_get_double_value(elpa_index_t index, char *name, int *success);


/*
 !f> interface
 !f>   function elpa_index_set_double_value_c(index, name, value) result(success) bind(C, name="elpa_index_set_double_value")
 !f>     import c_ptr, c_int, c_double, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     real(kind=c_double),intent(in), value :: value
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_set_double_value(elpa_index_t index, char *name, double value);


/*
 !f> interface
 !f>   function elpa_index_double_value_is_set_c(index, name) result(success) bind(C, name="elpa_index_double_value_is_set")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_double_value_is_set(elpa_index_t index, char *name);


/*
 !f> interface
 !f>   function elpa_index_get_double_loc_c(index, name) result(loc) bind(C, name="elpa_index_get_double_loc")
 !f>     import c_ptr, c_char
 !f>     type(c_ptr), value                 :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     type(c_ptr)                        :: loc
 !f>   end function
 !f> end interface
 */
double* elpa_index_get_double_loc(elpa_index_t index, char *name);


/*
 !f> interface
 !f>   function elpa_index_value_is_set_c(index, name) result(success) bind(C, name="elpa_index_value_is_set")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_value_is_set(elpa_index_t index, char *name);


/*
 !f> interface
 !f>   function elpa_int_value_to_string_c(name, value, string) &
 !f>              result(error) bind(C, name="elpa_int_value_to_string")
 !f>     import c_int, c_ptr, c_char
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     integer(kind=c_int), intent(in), value :: value
 !f>     type(c_ptr), intent(out) :: string
 !f>     integer(kind=c_int) :: error
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_int_value_to_string(char *name, int value, const char **string);


/*
 !f> interface
 !f>   pure function elpa_int_value_to_strlen_c(name, value) &
 !f>                   result(length) bind(C, name="elpa_int_value_to_strlen")
 !f>     import c_int, c_ptr, c_char
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     integer(kind=c_int), intent(in), value :: value
 !f>     integer(kind=c_int) :: length
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_int_value_to_strlen(char *name, int value);


/*
 !f> interface
 !f>   pure function elpa_index_int_value_to_strlen_c(index, name) &
 !f>                   result(length) bind(C, name="elpa_index_int_value_to_strlen")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     integer(kind=c_int) :: length
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_int_value_to_strlen(elpa_index_t index, char *name);


/*
 !f> interface
 !f>   function elpa_int_string_to_value_c(name, string, value) result(error) bind(C, name="elpa_int_string_to_value")
 !f>     import c_int, c_ptr, c_char
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     character(kind=c_char), intent(in) :: string(*)
 !f>     integer(kind=c_int), intent(out) :: value
 !f>     integer(kind=c_int) :: error
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_int_string_to_value(char *name, char *string, int *value);


/*
 !f> interface
 !f>   function elpa_option_cardinality_c(name) result(n) bind(C, name="elpa_option_cardinality")
 !f>     import c_int, c_char
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     integer(kind=c_int) :: n
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_option_cardinality(char *name);

/*
 !f> interface
 !f>   function elpa_option_enumerate_c(name, i) result(value) bind(C, name="elpa_option_enumerate")
 !f>     import c_int, c_char
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     integer(kind=c_int), intent(in), value :: i
 !f>     integer(kind=c_int) :: value
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_option_enumerate(char *name, int i);


/*
 !f> interface
 !f>   function elpa_index_int_is_valid_c(index, name, new_value) result(success) bind(C, name="elpa_index_int_is_valid")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     integer(kind=c_int), intent(in), value :: new_value
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_int_is_valid(elpa_index_t index, char *name, int new_value);
