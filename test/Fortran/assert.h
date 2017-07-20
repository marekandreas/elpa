#define stringify_(x) "x"
#define stringify(x) stringify_(x)
#define assert(x) call x_a(x, stringify(x), "F", __LINE__)

#define assert_elpa_ok(error_code) call x_ao(error_code, stringify(error_code), __FILE__, __LINE__)

! vim: syntax=fortran
