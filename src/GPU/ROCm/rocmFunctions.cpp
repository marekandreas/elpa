//
//    Copyright 2021 - 2023, A. Marek
//
//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//    This particular source code file contains additions, changes and
//    enhancements authored by Intel Corporation which is not part of
//    the ELPA consortium.
//
//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
//
// --------------------------------------------------------------------------------------------------
//
// This file was written by A. Marek, MPCDF

#include <stdio.h>
#include <math.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <complex.h>

#include "config-f90.h"


#undef SOLVER_status_success
#undef SOLVER_status
#undef SOLVER_handle
#undef SOLVER_FILL_MODE
#undef SOLVER_double_complex
#undef SOLVER_float_complex
#undef BLAS_status
#undef BLAS_handle
//#undef BLAS_float_complex
#undef BLAS_set_stream
#undef BLAS_status_success
#undef BLAS_status_invalid_handle
#undef BLAS_create_handle
#undef BLAS_destroy_handle
#undef BLAS_double_complex
#undef BLAS_float_complex
#undef BLAS_strsm
#undef BLAS_dtrsm
#undef BLAS_ctrsm
#undef BLAS_ztrsm
#undef BLAS_dtrmm
#undef BLAS_strmm
#undef BLAS_ztrmm
#undef BLAS_ctrmm
#undef BLAS_dcopy
#undef BLAS_scopy
#undef BLAS_zcopy
#undef BLAS_ccopy
#undef BLAS_dgemm
#undef BLAS_sgemm
#undef BLAS_zgemm
#undef BLAS_cgemm
#undef BLAS_dgemv
#undef BLAS_sgemv
#undef BLAS_zgemv
#undef BLAS_cgemv
#undef BLAS_dpotrf
#undef BLAS_spotrf
#undef BLAS_cpotrf
#undef BLAS_zpotrf
#undef BLAS_operation
#undef BLAS_operation_none
#undef BLAS_operation_transpose
#undef BLAS_operation_conjugate_transpose
#undef BLAS_operation_none
#undef BLAS_fill
#undef BLAS_fill_lower
#undef BLAS_fill_upper
#undef BLAS_side
#undef BLAS_side_left
#undef BLAS_side_right
#undef BLAS_diagonal
#undef BLAS_diagonal_non_unit
#undef BLAS_diagonal_unit
#undef BLAS_ddot
#undef BLAS_sdot
#undef BLAS_zdot
#undef BLAS_zdotc
#undef BLAS_zdotu
#undef BLAS_cdot
#undef BLAS_cdotu
#undef BLAS_cdotc
#undef BLAS_dscal
#undef BLAS_sscal
#undef BLAS_zscal
#undef BLAS_cscal
#undef BLAS_daxpy
#undef BLAS_saxpy
#undef BLAS_zaxpy
#undef BLAS_caxpy
#undef BLAS_set_pointer_mode
#undef BLAS_get_pointer_mode
#undef BLAS_pointer_mode_host
#undef BLAS_pointer_mode_device
#undef BLAS_pointer_mode

#ifdef HIPBLAS
#define SOLVER_status hipsolverStatus_t
#define SOLVER_status_success HIPSOLVER_STATUS_SUCCESS
#define SOLVER_FILL_MODE hipsolver_fill_mode
#define SOLVER_handle hipsolverHandle_t
#define SOLVER_double_complex hipDoubleComplex
#define SOLVER_float_complex hipFloatComplex
#define BLAS hipblas
#define BLAS_status hipblasStatus_t
#define BLAS_handle hipblasHandle_t
#define BLAS_set_stream hipblasSetStream
#define BLAS_status_success HIPBLAS_STATUS_SUCCESS
#define BLAS_status_invalid_handle HIPBLAS_STATUS_INVALID_VALUE
#define BLAS_create_handle hipblasCreate
#define BLAS_destroy_handle hipblasDestroy
#define BLAS_double_complex hipblasDoubleComplex
#define BLAS_float_complex hipblasComplex
#define BLAS_ctrsm hipblasCtrsm
#define BLAS_ztrsm hipblasZtrsm
#define BLAS_dtrsm hipblasDtrsm
#define BLAS_strsm hipblasStrsm
#define BLAS_ctrmm hipblasCtrmm
#define BLAS_ztrmm hipblasZtrmm
#define BLAS_dtrmm hipblasDtrmm
#define BLAS_strmm hipblasStrmm
#define BLAS_ccopy hipblasCcopy
#define BLAS_zcopy hipblasZcopy
#define BLAS_dcopy hipblasDcopy
#define BLAS_scopy hipblasScopy
#define BLAS_cgemm hipblasCgemm
#define BLAS_zgemm hipblasZgemm
#define BLAS_dgemm hipblasDgemm
#define BLAS_sgemm hipblasSgemm
#define BLAS_cgemv hipblasCgemv
#define BLAS_zgemv hipblasZgemv
#define BLAS_dgemv hipblasDgemv
#define BLAS_sgemv hipblasSgemv
#define BLAS_dpotrf hipsolverDpotrf
#define BLAS_spotrf hipsolverSpotrf
#define BLAS_zpotrf hipsolverZpotrf
#define BLAS_cpotrf hipsolverCpotrf
#define BLAS_operation hipblasOperation_t
#define BLAS_operation_none HIPBLAS_OP_N
#define BLAS_operation_transpose HIPBLAS_OP_T
#define BLAS_operation_conjugate_transpose HIPBLAS_OP_C
#define BLAS_operation_none HIPBLAS_OP_N
#define BLAS_fill hipblasFillMode_t
#define BLAS_fill_lower HIPBLAS_FILL_MODE_LOWER
#define BLAS_fill_upper HIPBLAS_FILL_MODE_UPPER
#define BLAS_side hipblasSideMode_t
#define BLAS_side_left HIPBLAS_SIDE_LEFT
#define BLAS_side_right HIPBLAS_SIDE_RIGHT
#define BLAS_diagonal hipblasDiagType_t
#define BLAS_diagonal_non_unit HIPBLAS_DIAG_NON_UNIT
#define BLAS_diagonal_unit HIPBLAS_DIAG_UNIT
#define BLAS_ddot hipblasDdot
#define BLAS_sdot hipblasSdot
#define BLAS_zdot hipblasZdot
#define BLAS_zdotc hipblasZdotc
#define BLAS_zdotu hipblasZdotu
#define BLAS_cdot hipblasCdot
#define BLAS_cdotc hipblasCdotc
#define BLAS_cdotu hipblasCdotu
#define BLAS_dscal hipblasDscal
#define BLAS_sscal hipblasSscal
#define BLAS_zscal hipblasZscal
#define BLAS_cscal hipblasCscal
#define BLAS_daxpy hipblasDaxpy
#define BLAS_saxpy hipblasSaxpy
#define BLAS_zaxpy hipblasZaxpy
#define BLAS_caxpy hipblasCaxpy
#define BLAS_set_pointer_mode hipblasSetPointerMode
#define BLAS_get_pointer_mode hipblasGetPointerMode
#define BLAS_pointer_mode_host HIPBLAS_POINTER_MODE_HOST
#define BLAS_pointer_mode_device HIPBLAS_POINTER_MODE_DEVICE
#define BLAS_pointer_mode hipblasPointerMode_t
//#define BLAS_float_complex hipblas_float_complex
//#define BLAS_set_stream hipblas_set_stream
#else /* HIPBLAS */
#define SOLVER_status rocblas_status
#define SOLVER_status_success rocblas_status_success
#define SOLVER_FILL_MODE hip_fill_mode
#define SOLVER_handle rocblas_handle
#define SOLVER_double_complex rocblas_double_complex 
#define SOLVER_float_complex rocblas_float_complex
#define BLAS rocblas
#define BLAS_status rocblas_status
#define BLAS_handle rocblas_handle
#define BLAS_set_stream rocblas_set_stream
#define BLAS_status_success rocblas_status_success
#define BLAS_status_invalid_handle rocblas_status_invalid_handle
#define BLAS_status_memory_error rocblas_status_memory_error
#define BLAS_create_handle rocblas_create_handle
#define BLAS_destroy_handle rocblas_destroy_handle
#define BLAS_double_complex rocblas_double_complex
#define BLAS_float_complex rocblas_float_complex
#define BLAS_ctrsm rocblas_ctrsm
#define BLAS_ztrsm rocblas_ztrsm
#define BLAS_dtrsm rocblas_dtrsm
#define BLAS_strsm rocblas_strsm
#define BLAS_ctrmm rocblas_ctrmm
#define BLAS_ztrmm rocblas_ztrmm
#define BLAS_dtrmm rocblas_dtrmm
#define BLAS_strmm rocblas_strmm
#define BLAS_ccopy rocblas_ccopy
#define BLAS_zcopy rocblas_zcopy
#define BLAS_dcopy rocblas_dcopy
#define BLAS_scopy rocblas_scopy
#define BLAS_cgemm rocblas_cgemm
#define BLAS_zgemm rocblas_zgemm
#define BLAS_dgemm rocblas_dgemm
#define BLAS_sgemm rocblas_sgemm
#define BLAS_cgemv rocblas_cgemv
#define BLAS_zgemv rocblas_zgemv
#define BLAS_dgemv rocblas_dgemv
#define BLAS_sgemv rocblas_sgemv
#define BLAS_dpotrf rocsolver_dpotrf
#define BLAS_spotrf rocsolver_spotrf
#define BLAS_zpotrf rocsolver_zpotrf
#define BLAS_cpotrf rocsolver_cpotrf
#define BLAS_operation rocblas_operation
#define BLAS_operation_none rocblas_operation_none
#define BLAS_operation_transpose rocblas_operation_transpose
#define BLAS_operation_conjugate_transpose rocblas_operation_conjugate_transpose
#define BLAS_operation_none rocblas_operation_none
#define BLAS_fill rocblas_fill
#define BLAS_fill_lower rocblas_fill_lower
#define BLAS_fill_upper rocblas_fill_upper
#define BLAS_side rocblas_side
#define BLAS_side_left rocblas_side_left
#define BLAS_side_right rocblas_side_right
#define BLAS_diagonal rocblas_diagonal
#define BLAS_diagonal_non_unit rocblas_diagonal_non_unit
#define BLAS_diagonal_unit rocblas_diagonal_unit
#define BLAS_ddot rocblas_ddot
#define BLAS_sdot rocblas_sdot
#define BLAS_zdot rocblas_zdot
#define BLAS_zdotu rocblas_zdotu
#define BLAS_zdotc rocblas_zdotc
#define BLAS_cdot rocblas_cdot
#define BLAS_cdotc rocblas_cdotc
#define BLAS_cdotu rocblas_cdotu
#define BLAS_dscal rocblas_dscal
#define BLAS_sscal rocblas_sscal
#define BLAS_zscal rocblas_zscal
#define BLAS_cscal rocblas_cscal
#define BLAS_daxpy rocblas_daxpy
#define BLAS_saxpy rocblas_saxpy
#define BLAS_zaxpy rocblas_zaxpy
#define BLAS_caxpy rocblas_caxpy
#define BLAS_set_pointer_mode rocblas_set_pointer_mode
#define BLAS_get_pointer_mode rocblas_get_pointer_mode
#define BLAS_pointer_mode_host rocblas_pointer_mode_host
#define BLAS_pointer_mode_device rocblas_pointer_mode_device
#define BLAS_pointer_mode rocblas_pointer_mode
#endif /* HIPBLAS */

#ifdef HIPBLAS
#include <hipblas/hipblas.h>
#ifdef WITH_AMD_ROCSOLVER
#include <hipsolver/hipsolver.h>
#endif
#else /* HIPBLAS */
#include "rocblas/rocblas.h"
#ifdef WITH_AMD_ROCSOLVER
#include <rocsolver/rocsolver.h>
#endif
#endif /* HIPBLAS */
#include "hip/hip_runtime_api.h"


#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_HIP
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#ifdef WITH_AMD_GPU_VERSION
#include "./rocmFunctions_template.h"
#endif

#ifdef WITH_AMD_ROCSOLVER
#include "./rocsolverFunctions_template.h"
#endif
