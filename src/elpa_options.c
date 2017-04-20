#include "elpa_index.h"

/* summary timings */
static int summary_timings_cardinality() {
        return 2;
}

static const int summary_timings_enumerate_option(unsigned int n) {
        return n;
}

static int summary_timings_valid(elpa_index_t options, int value) {
        if (value >= 0 && value < summary_timings_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}


/* wantDebug */
static int wantDebug_cardinality() {
        return 2;
}

static const int wantDebug_enumerate_option(unsigned int n) {
        return n;
}

static int wantDebug_valid(elpa_index_t options, int value) {
        if (value >= 0 && value < wantDebug_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

/* QR */
static int qr_cardinality() {
        return 2;
}

static const int qr_enumerate_option(unsigned int n) {
        return n;
}

static int qr_valid(elpa_index_t options, int value) {
        if (value >= 0 && value < qr_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

/* GPU */
static int gpu_cardinality() {
        return 2;
}

static const int gpu_enumerate_option(unsigned int n) {
        return n;
}

static int gpu_valid(elpa_index_t options, int value) {
        if (value >= 0 && value < gpu_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

/* Solver */
static int solver_cardinality() {
        return ELPA_NUMBER_OF_SOLVERS;
}

static const int solver_enumerate_option(unsigned int n) {
        return n+1;
}

static int solver_valid(elpa_index_t options, int value) {
        if (value >= 1 && value <= solver_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}


/* Real Kernel */
static int real_kernel_cardinality() {
        return ELPA_2STAGE_NUMBER_OF_REAL_KERNELS;
}

static const int real_kernel_enumerate_option(unsigned int n) {
        return n+1;
}

static int real_kernel_valid(elpa_index_t options, int value) {
        if (value >= 1 && value <= real_kernel_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}


/* Complex Kernel */
static int complex_kernel_cardinality() {
        return ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS;
}

static const int complex_kernel_enumerate_option(unsigned int n) {
        return n+1;
}

static int complex_kernel_valid(elpa_index_t options, int value) {
        if (value >= 1 && value <= complex_kernel_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

/** END OF OPTIONS **/

static elpa_int_entry_t elpa_int_options[] = {
        {"summary_timings", 0, summary_timings_cardinality, summary_timings_enumerate_option, summary_timings_valid},
        {"wantDebug", 0, wantDebug_cardinality, wantDebug_enumerate_option, wantDebug_valid},
        {"qr", 0, qr_cardinality, qr_enumerate_option, qr_valid},
        {"gpu", 0, gpu_cardinality, gpu_enumerate_option, gpu_valid},
        {"solver", ELPA_SOLVER_1STAGE, solver_cardinality, solver_enumerate_option, solver_valid},
        {"real_kernel", ELPA_2STAGE_REAL_DEFAULT, real_kernel_cardinality, real_kernel_enumerate_option, real_kernel_valid},
        {"complex_kernel", ELPA_2STAGE_COMPLEX_DEFAULT, complex_kernel_cardinality, complex_kernel_enumerate_option, complex_kernel_valid},
};

/*
 !f> interface
 !f>   function elpa_allocate_options() result(options) bind(C, name="elpa_allocate_options")
 !f>     import c_ptr
 !f>     type(c_ptr) :: options
 !f>   end function
 !f> end interface
 */
elpa_index_t elpa_allocate_options() {
        elpa_index_t options = elpa_allocate_index(nelements(elpa_int_options), elpa_int_options);
}
