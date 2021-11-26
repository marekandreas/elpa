/*
!
!    Copyright 2017, L. Hüdepohl and A. Marek, MPCDF
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
*/
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <search.h>
#include <math.h>

#include "config.h"
#include <elpa/elpa.h>

#define nelements(x) (sizeof(x)/sizeof(x[0]))

#define FOR_ALL_TYPES(X) \
        X(int, "%d", "%d", -1) \
        X(float, "%g", "%lg", NAN) \
        X(double, "%g", "%lg", NAN)

/* A simple structure for storing values to a pre-set
 * number of keys */

/* Forward declaration of configuration structure */
typedef struct elpa_index_struct* elpa_index_t;

/* Function type for the cardinality */
typedef int (*elpa_index_cardinality_t)(elpa_index_t index);

/* Function type to enumerate all possible values, starting from 0 */
typedef int (*elpa_index_enumerate_int_option_t)(elpa_index_t index, int i);

/* Function types to check the validity of a value */
typedef int (*elpa_index_valid_int_t)(elpa_index_t index, int n, int new_value);
typedef int (*elpa_index_valid_float_t)(elpa_index_t index, int n, float new_value);
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
        int print_flag;
} elpa_index_entry_t;


typedef struct {
        elpa_index_entry_t base;
        int default_value;
        int autotune_level_old;
        int autotune_level;
        int autotune_domain;
	int autotune_part;
        elpa_index_valid_int_t valid;
        elpa_index_cardinality_t cardinality;
        elpa_index_enumerate_int_option_t enumerate;
        elpa_index_to_string_int_t to_string;
} elpa_index_int_entry_t;


typedef struct {
        elpa_index_entry_t base;
        float default_value;
        elpa_index_valid_float_t valid;
} elpa_index_float_entry_t;


typedef struct {
        elpa_index_entry_t base;
        double default_value;
        elpa_index_valid_double_t valid;
} elpa_index_double_entry_t;

enum NOTIFY_FLAGS {
        NOTIFY_ENV_DEFAULT = (1<<0),
        NOTIFY_ENV_FORCE   = (1<<1),
};

enum PRINT_FLAGS {
        PRINT_STRUCTURE,
        PRINT_YES,
        PRINT_NO,
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
 !f>   function elpa_index_get_int_value_c(index, name, success) result(value) &
 !f>       bind(C, name="elpa_index_get_int_value")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                         :: index
 !f>     character(kind=c_char), intent(in)         :: name(*)
 !f>#ifdef USE_FORTRAN2008
 !f>     integer(kind=c_int), optional, intent(out) :: success
 !f>#else
 !f>     integer(kind=c_int), intent(out)           :: success
 !f>#endif
 !f>     integer(kind=c_int)                        :: value
 !f>   end function
 !f> end interface
 */
int elpa_index_get_int_value(elpa_index_t index, char *name, int *success);


/*
 !f> interface
 !f>   function elpa_index_set_int_value_c(index, name, value) result(success) &
 !f>       bind(C, name="elpa_index_set_int_value")
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
 !f>   function elpa_index_get_float_value_c(index, name, success) result(value) bind(C, name="elpa_index_get_float_value")
 !f>     import c_ptr, c_int, c_float, c_char
 !f>     type(c_ptr), value                              :: index
 !f>     character(kind=c_char), intent(in)              :: name(*)
 !f>#ifdef USE_FORTRAN2008
 !f>     integer(kind=c_int), intent(out), optional      :: success
 !f>#else
 !f>     integer(kind=c_int), intent(out)                :: success
 !f>#endif
 !f>     real(kind=c_float)                             :: value
 !f>   end function
 !f> end interface
 */
float elpa_index_get_float_value(elpa_index_t index, char *name, int *success);


/*
 !f> interface
 !f>   function elpa_index_set_float_value_c(index, name, value) result(success) &
 !f>       bind(C, name="elpa_index_set_float_value")
 !f>     import c_ptr, c_int, c_float, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     real(kind=c_float),intent(in), value :: value
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_set_float_value(elpa_index_t index, char *name, float value);


/*
 !f> interface
 !f>   function elpa_index_float_value_is_set_c(index, name) result(success) &
 !f>       bind(C, name="elpa_index_float_value_is_set")
 !f>     import c_ptr, c_int, c_char
 !f>     type(c_ptr), value                    :: index
 !f>     character(kind=c_char), intent(in)    :: name(*)
 !f>     integer(kind=c_int)                   :: success
 !f>   end function
 !f> end interface
 */
int elpa_index_float_value_is_set(elpa_index_t index, char *name);


/*
 !f> interface
 !f>   function elpa_index_get_float_loc_c(index, name) result(loc) bind(C, name="elpa_index_get_float_loc")
 !f>     import c_ptr, c_char
 !f>     type(c_ptr), value                 :: index
 !f>     character(kind=c_char), intent(in) :: name(*)
 !f>     type(c_ptr)                        :: loc
 !f>   end function
 !f> end interface
 */
float* elpa_index_get_float_loc(elpa_index_t index, char *name);


/*
 !f> interface
 !f>   function elpa_index_get_double_value_c(index, name, success) result(value) bind(C, name="elpa_index_get_double_value")
 !f>     import c_ptr, c_int, c_double, c_char
 !f>     type(c_ptr), value                              :: index
 !f>     character(kind=c_char), intent(in)              :: name(*)
 !f>#ifdef USE_FORTRAN2008
 !f>     integer(kind=c_int), intent(out), optional      :: success
 !f>#else
 !f>     integer(kind=c_int), intent(out)                :: success
 !f>#endif
 !f>     real(kind=c_double)                             :: value
 !f>   end function
 !f> end interface
 */
double elpa_index_get_double_value(elpa_index_t index, char *name, int *success);


/*
 !f> interface
 !f>   function elpa_index_set_double_value_c(index, name, value) result(success) &
 !f>       bind(C, name="elpa_index_set_double_value")
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
 !f>   function elpa_index_double_value_is_set_c(index, name) result(success) &
 !f>       bind(C, name="elpa_index_double_value_is_set")
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
 !pf> interface
 !pf>   function elpa_int_value_to_string_c(name, value, string) &
 !pf>              result(error) bind(C, name="elpa_int_value_to_string")
 !pf>     import c_int, c_ptr, c_char
 !pf>     character(kind=c_char), intent(in) :: name(*)
 !pf>     integer(kind=c_int), intent(in), value :: value
 !pf>     type(c_ptr), intent(out) :: string
 !pf>     integer(kind=c_int) :: error
 !pf>   end function
 !pf> end interface
 !pf>
 */
int elpa_int_value_to_string(char *name, int value, const char **string);


/*
 !pf> interface
 !pf>   pure function elpa_int_value_to_strlen_c(name, value) &
 !pf>                   result(length) bind(C, name="elpa_int_value_to_strlen")
 !pf>     import c_int, c_ptr, c_char
 !pf>     character(kind=c_char), intent(in) :: name(*)
 !pf>     integer(kind=c_int), intent(in), value :: value
 !pf>     integer(kind=c_int) :: length
 !pf>   end function
 !pf> end interface
 !pf>
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
 !f>   function elpa_index_int_is_valid_c(index, name, new_value) result(success) &
 !f>       bind(C, name="elpa_index_int_is_valid")
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


/*
 !f> interface
 !f>   function elpa_index_autotune_cardinality_c(index, autotune_level, autotune_domain) result(n) &
 !f>       bind(C, name="elpa_index_autotune_cardinality")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in), value :: autotune_level, autotune_domain
 !f>     integer(kind=c_int) :: n
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_autotune_cardinality(elpa_index_t index, int autotune_level, int autotune_domain);


/*
 !f> interface
 !f>   function elpa_index_autotune_cardinality_new_stepping_c(index, &
 !f>            autotune_level, &
 !f>       autotune_domain, autotune_part) result(n) &
 !f>       bind(C, name="elpa_index_autotune_cardinality_new_stepping")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in), value :: autotune_level, &
 !f>                                               autotune_domain
 !f>     integer(kind=c_int), intent(in), value :: autotune_part
 !f>     integer(kind=c_int) :: n
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_autotune_cardinality_new_stepping(elpa_index_t index, int autotune_level, int autotune_domain, int autotune_part);


/*
 !f> interface
 !f>   function elpa_index_set_autotune_parameters_c(index, autotune_level, autotune_domain, n) result(success) &
 !f>       bind(C, name="elpa_index_set_autotune_parameters")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in), value :: autotune_level, autotune_domain, n
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_set_autotune_parameters(elpa_index_t index, int autotune_level, int autotune_domain, int n);


/*
 !f> interface
 !f>   function elpa_index_set_autotune_parameters_new_stepping_c(index, &
 !f>            autotune_level, autotune_domain, autotune_part, n) &
 !f>       result(success) &
 !f>       bind(C, name="elpa_index_set_autotune_parameters_new_stepping")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in), value :: autotune_level, &
 !f>                                               autotune_domain, n
 !f>     integer(kind=c_int), intent(in), value :: autotune_part
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_set_autotune_parameters_new_stepping(elpa_index_t index, int autotune_level, int autotune_domain, int autotune_part, int n);


/*
 !f> interface
 !f>   function elpa_index_print_autotune_parameters_c(index, autotune_level, autotune_domain) result(success) &
 !f>       bind(C, name="elpa_index_print_autotune_parameters")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in), value :: autotune_level, autotune_domain
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_print_autotune_parameters(elpa_index_t index, int autotune_level, int autotune_domain);

/*
 !f> interface
 !f>   function elpa_index_print_autotune_parameters_new_stepping_c(index, &
 !f>       autotune_level, autotune_domain, autotune_part) result(success) &
 !f>       bind(C, name="elpa_index_print_autotune_parameters_new_stepping")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in), value :: autotune_level, &
 !f>                                               autotune_domain
 !f>     integer(kind=c_int), intent(in), value :: autotune_part
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_print_autotune_parameters_new_stepping(elpa_index_t index, int autotune_level, int autotune_domain, int autotune_part);


/*
 !f> interface
 !f>   function elpa_index_print_settings_c(index, file_name) result(success) &
 !f>       bind(C, name="elpa_index_print_settings")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     character(kind=c_char), intent(in)     :: file_name(*)
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_print_settings(elpa_index_t index, char* filename);

/*
 !f> interface
 !f>   function elpa_index_load_settings_c(index, file_name) result(success) &
 !f>       bind(C, name="elpa_index_load_settings")
 !f>     import c_int, c_ptr, c_char
 !f>     type(c_ptr), intent(in), value :: index
 !f>     character(kind=c_char), intent(in)     :: file_name(*)
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_load_settings(elpa_index_t index, char* filename);

/*
 !f> interface
 !f>   function elpa_index_print_autotune_state_c(index, autotune_level, autotune_domain, min_loc, &
 !f>                                              min_val, current, cardinality, file_name) result(success) &
 !f>       bind(C, name="elpa_index_print_autotune_state")
 !f>     import c_int, c_ptr, c_char, c_double
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in), value :: autotune_level, autotune_domain, min_loc, current, cardinality
 !f>     real(kind=c_double), intent(in), value :: min_val
 !f>     character(kind=c_char), intent(in)     :: file_name(*)
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_print_autotune_state(elpa_index_t index, int autotune_level, int autotune_domain, int min_loc,
                                    double min_val, int current, int cardinality, char* filename);


/*
 !f> interface
 !f>   function elpa_index_print_autotune_state_new_stepping_c(index, &
 !f>            autotune_level, autotune_domain, autotune_part, min_loc, &
 !f>            min_val, current, cardinality, solver, file_name) result(success) &
 !f>       bind(C, name="elpa_index_print_autotune_state_new_stepping")
 !f>     import c_int, c_ptr, c_char, c_double
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in), value :: autotune_level, &
 !f>                                               autotune_domain, &
 !f>                                               min_loc, current, &
 !f>                                               cardinality
 !f>     integer(kind=c_int), intent(in), value :: autotune_part, solver
 !f>     real(kind=c_double), intent(in), value :: min_val
 !f>     character(kind=c_char), intent(in)     :: file_name(*)
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_print_autotune_state_new_stepping(elpa_index_t index, int autotune_level, int autotune_domain, int autotune_part, int min_loc,
                                    double min_val, int current, int cardinality, int solver, char* filename);

/*
 !f> interface
 !f>   function elpa_index_load_autotune_state_c(index, autotune_level, autotune_domain, min_loc, &
 !f>                                              min_val, current, cardinality, file_name) result(success) &
 !f>       bind(C, name="elpa_index_load_autotune_state")
 !f>     import c_int, c_ptr, c_char, c_double
 !f>     type(c_ptr), intent(in), value :: index
 !f>     integer(kind=c_int), intent(in) :: autotune_level, autotune_domain, min_loc, current, cardinality
 !f>     real(kind=c_double), intent(in) :: min_val
 !f>     character(kind=c_char), intent(in)     :: file_name(*)
 !f>     integer(kind=c_int) :: success
 !f>   end function
 !f> end interface
 !f>
 */
int elpa_index_load_autotune_state(elpa_index_t index, int* autotune_level, int* autotune_domain, int* min_loc,
                                    double* min_val, int* current, int* cardinality, char* filename);

int elpa_index_is_printing_mpi_rank(elpa_index_t index);
