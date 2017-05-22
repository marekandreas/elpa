#ifndef ELPA_H
#define ELPA_H

#include <limits.h>
#include <complex.h>

struct elpa_struct;
typedef struct elpa_struct *elpa_t;

#include <elpa/elpa_constants.h>
#include <elpa/elpa_generated.h>
#include <elpa/elpa_generic.h>

const char *elpa_strerr(int elpa_error);

#endif
