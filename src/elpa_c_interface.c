#include <elpa/elpa.h>

/*
 !f> interface
 !f>   pure function elpa_strerr_c(elpa_error) result(string) bind(C, name="elpa_strerr")
 !f>     import c_int, c_ptr
 !f>     integer(kind=c_int), intent(in), value :: elpa_error
 !f>     type(c_ptr) :: string
 !f>   end function
 !f> end interface
 */
const char *elpa_strerr(int elpa_error) {
#define NAME_CASE(name, value) \
        case value: \
                return #name;

        switch(elpa_error) {
                ELPA_FOR_ALL_ERRORS(NAME_CASE)
                default:
                        return "(Unknown error code)";
        }
}
