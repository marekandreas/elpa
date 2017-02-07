#include "config-f90.h"
module redist

  public

  contains
#define DOUBLE_PRECISION_REAL 1

#define REAL_DATATYPE rk8
#define BYTESIZE 8
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "precision_macros.h"
#undef PRECISION_SUFFIX
#define PRECISION_SUFFIX "_double"
#define PRECISION double
#include "redist_band.X90"

#undef DOUBLE_PRECISION_REAL
#undef REAL_DATATYPE
#undef BYTESIZE
#undef REALCASE
#undef DOUBLE_PRECISION
#undef PRECISION_SUFFIX
#undef PRECISION


! single precision
#ifdef WANT_SINGLE_PRECISION_REAL

#undef DOUBLE_PRECISION_REAL
#undef DOUBLE_PRECISION
#define REAL_DATATYPE rk4
#define BYTESIZE 4
#define REALCASE 1
#include "precision_macros.h"
#undef PRECISION_SUFFIX
#define PRECISION_SUFFIX "_single"
#undef PRECISION
#define PRECISION single
#include "redist_band.X90"

#undef REAL_DATATYPE
#undef BYTESIZE
#undef REALCASE
#undef PRECISION_SUFFIX
#undef PRECISION

#endif /* WANT_SINGLE_PRECISION_REAL */

! double precision
#define DOUBLE_PRECISION_COMPLEX 1

#define COMPLEX_DATATYPE ck8
#define BYTESIZE 16
#define COMPLEXCASE 1
#define DOUBLE_PRECISION
#include "precision_macros.h"
#undef PRECISION_SUFFIX
#define PRECISION_SUFFIX "_double"
#undef PRECISION
#define PRECISION double
#include "redist_band.X90"

#undef COMPLEX_DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE
#undef DOUBLE_PRECISION
#undef DOUBLE_PRECISION_COMPLEX
#undef PRECISION_SUFFIX
#undef PRECISION

#ifdef WANT_SINGLE_PRECISION_COMPLEX

#undef DOUBLE_PRECISION_COMPLEX
#undef DOUBLE_PRECISION_REAL
#undef DOUBLE_PRECISION
#define COMPLEX_DATATYPE ck4
#define COMPLEXCASE 1
#include "precision_macros.h"
#undef PRECISION_SUFFIX
#define PRECISION_SUFFIX "_single"
#undef PRECISION
#define PRECISION single
#include "redist_band.X90"

#undef COMPLEX_DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE
#undef PRECISION_SUFFIX

#endif /* WANT_SINGLE_PRECISION_COMPLEX */



end module redist

