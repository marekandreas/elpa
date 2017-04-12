#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <search.h>

#include <elpa/elpa.h>


#define nelements(x) (sizeof(x)/sizeof(x[0]))

/* Incomplete forward declaration of configuration structure */
typedef struct elpa_options_struct* elpa_options_t;

/* Function pointer type for the cardinality */
typedef int (*cardinality_t)();

/* Function pointer type to enumerate all possible options */
typedef const int (*enumerate_int_option_t)(unsigned int n);

/* Function pointer type check validity of option */
typedef int (*valid_int_t)(elpa_options_t options, int value);

typedef struct {
        const char *name;
        int default_value;
        cardinality_t cardinality;
        enumerate_int_option_t enumerate_option;
        valid_int_t valid;
} elpa_int_option_t;


/** OPTIONS **/


/* wantDebug */
int wantDebug_cardinality() {
        return 2;
}

const int wantDebug_enumerate_option(unsigned int n) {
        return n;
}

int wantDebug_valid(elpa_options_t options, int value) {
        if (value >= 0 && value < wantDebug_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

/* QR */
int qr_cardinality() {
        return 2;
}

const int qr_enumerate_option(unsigned int n) {
        return n;
}

int qr_valid(elpa_options_t options, int value) {
        if (value >= 0 && value < qr_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

/* GPU */
int gpu_cardinality() {
        return 2;
}

const int gpu_enumerate_option(unsigned int n) {
        return n;
}

int gpu_valid(elpa_options_t options, int value) {
        if (value >= 0 && value < gpu_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

/* Solver */
int solver_cardinality() {
        return ELPA_NUMBER_OF_SOLVERS;
}

const int solver_enumerate_option(unsigned int n) {
        return n+1;
}

int solver_valid(elpa_options_t options, int value) {
        if (value >= 1 && value <= solver_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}


/* Real Kernel */
int real_kernel_cardinality() {
        return ELPA_2STAGE_NUMBER_OF_REAL_KERNELS;
}

const int real_kernel_enumerate_option(unsigned int n) {
        return n+1;
}

int real_kernel_valid(elpa_options_t options, int value) {
        if (value >= 1 && value <= real_kernel_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}


/* Complex Kernel */
int complex_kernel_cardinality() {
        return ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS;
}

const int complex_kernel_enumerate_option(unsigned int n) {
        return n+1;
}

int complex_kernel_valid(elpa_options_t options, int value) {
        if (value >= 1 && value <= complex_kernel_cardinality()) {
                return ELPA_OK;
        } else {
                return ELPA_ERROR;
        }
}

/** END OF OPTIONS **/


elpa_int_option_t elpa_int_options[] = {
        {"wantDebug", 0, wantDebug_cardinality, wantDebug_enumerate_option, wantDebug_valid},
        {"qr", 0, qr_cardinality, qr_enumerate_option, qr_valid},
        {"gpu", 0, gpu_cardinality, gpu_enumerate_option, gpu_valid},
        {"solver", ELPA_SOLVER_1STAGE, solver_cardinality, solver_enumerate_option, solver_valid},
        {"real_kernel", ELPA_2STAGE_REAL_DEFAULT, real_kernel_cardinality, real_kernel_enumerate_option, real_kernel_valid},
        {"complex_kernel", ELPA_2STAGE_COMPLEX_DEFAULT, complex_kernel_cardinality, complex_kernel_enumerate_option, complex_kernel_valid},
};

struct elpa_options_struct {
        int int_options[nelements(elpa_int_options)];
};

elpa_options_t elpa_allocate_options() {
        elpa_options_t options = (elpa_options_t) calloc(1, sizeof(struct elpa_options_struct));
        int i;
        for (i = 0; i < nelements(elpa_int_options); i++) {
                options->int_options[i] = elpa_int_options[i].default_value;
        }
        return options;
}

void elpa_free_options(elpa_options_t options) {
        free(options);
}

int compar(const void *key, const void *member) {
        const char *name = (const char *) key;
        elpa_int_option_t *option = (elpa_int_option_t *) member;

        int l1 = strlen(option->name);
        int l2 = strlen(name);
        if (l1 != l2) {
                return 1;
        }
        if (strncmp(name, option->name, l1) == 0) {
                return 0;
        } else {
                return 1;
        }
}

int find_int_option(const char *name) {
        elpa_int_option_t *option;
        size_t nmembers = nelements(elpa_int_options);
        option = lfind((const void*) name, (const void *) &elpa_int_options, &nmembers, sizeof(elpa_int_option_t), compar);
        if (option) {
                return (option - &elpa_int_options[0]);
        } else {
                return -1;
        }
}

int get_int_option(elpa_options_t options, const char *name, int *success) {
        int n = find_int_option(name);
        if (n >= 0) {
                if (success != NULL) {
                        *success = ELPA_OK;
                }
                return options->int_options[n];
        } else {
                if (success != NULL) {
                        *success = ELPA_ERROR;
                } else {
                        fprintf(stderr, "ELPA: No such option '%s' and you did not check for errors, returning ELPA_INVALID_INT!\n", name);
                }
                return ELPA_INVALID_INT;
        }
}

int set_int_option(elpa_options_t options, const char *name, int value) {
        int n = find_int_option(name);
        int res = ELPA_ERROR;
        if (n >= 0) {
                res = elpa_int_options[n].valid(options, value);
                if (res == ELPA_OK) {
                        options->int_options[n] = value;
                }
        }
        return res;
}
