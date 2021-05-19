#ifndef ELPA_H
#define ELPA_H

#include <limits.h>
#include <complex.h>

#include <elpa/elpa_version.h>

struct elpa_struct;
typedef struct elpa_struct *elpa_t;

struct elpa_autotune_struct;
typedef struct elpa_autotune_struct *elpa_autotune_t;


#include <elpa/elpa_constants.h>
#include <elpa/elpa_generated_c_api.h>
#include <elpa/elpa_generated.h>
#include <elpa/elpa_generic.h>

#define ELPA_2STAGE_REAL_GPU    ELPA_2STAGE_REAL_NVIDIA_GPU
#define ELPA_2STAGE_COMPLEX_GPU ELPA_2STAGE_COMPLEX_NVIDIA_GPU

const char *elpa_strerr(int elpa_error);

#endif
