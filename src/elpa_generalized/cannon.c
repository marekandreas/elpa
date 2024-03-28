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
//    This particular source code file has been developed within the ELPA-AEO //
//    project, which has been a joint effort of
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Technische Universität München, Lehrstuhl für Theoretische Chemie,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,

//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/ and
//    http://elpa-aeo.mpcdf.mpg.de
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
// Author: Valeriy Manin (Bergische Universität Wuppertal)
// integreated into the ELPA library Pavel Kus, Andeas Marek (MPCDF)

#include "config-f90.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include "../helpers/scalapack_interfaces.h"

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define C_INT_TYPE_PTR long int*
#define C_INT_TYPE long int
#define BLAS_KIND c_int64_t
#else
#define C_INT_TYPE_PTR int*
#define C_INT_TYPE int
#define BLAS_KIND c_int
#endif

#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define C_INT_MPI_TYPE_PTR long int*
#define C_INT_MPI_TYPE long int
#define MPI_KIND c_int64_t
#else
#define C_INT_MPI_TYPE_PTR int*
#define C_INT_MPI_TYPE int
#define MPI_KIND c_int
#endif

#ifdef WITH_NVTX
#include <nvToolsExt.h>
#endif

#ifdef WITH_NVTX
#define NVTX_RANGE_PUSH(msg) nvtxRangePushA(msg)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
// Do nothing if WITH_NVTX is not defined
#define NVTX_RANGE_PUSH(msg) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#endif

#ifdef WITH_NVIDIA_GPU_VERSION
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

int gpuMemcpyHostToDevice;
int gpuMemcpyDeviceToHost;
#include "./gpu_vendor_agnostic_layer.h"

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(int status, const char *file, int line)
{
   if (status != 1) // 1 = success for ELPA GPU interfaces 
   {
      fprintf(stderr,"GPUassert: %s %d\n", file, line);
      exit(!status);
   }
}


// most of the file is not compiled if not using MPI
#ifdef WITH_MPI
#include <mpi.h>

#ifdef NEED_NO_UNDERSCORE_TO_LINK_AGAINST_FORTRAN
#define numroc_ numroc
#define dlacpy_ dlacpy
#define slacpy_ slacpy
#define zlacpy_ zlacpy
#define clacpy_ clacpy
#define pdtran_ pdtran
#define pstran_ pstran
#define pztranc_ pztranc
#define pctranc_ pctranc
#define pdlacpy_ pdlacpy
#define pslacpy_ pslacpy
#define pzlacpy_ pzlacpy
#define pclacpy_ pclacpy
#endif

#ifdef NEED_UNDERSCORE_TO_LINK_AGAINST_FORTRAN
#define numroc_ numroc_
#define dlacpy_ dlacpy_
#define slacpy_ slacpy_
#define zlacpy_ zlacpy_
#define clacpy_ clacpy_
#define pdtran_ pdtran_
#define pstran_ pstran_
#define pztranc_ pztranc_
#define pctranc_ pctranc_
#define pdlacpy_ pdlacpy_
#define pslacpy_ pslacpy_
#define pzlacpy_ pzlacpy_
#define pclacpy_ pclacpy_
#endif



//***********************************************************************************************************

#define REALCASE 1
#define DOUBLE_PRECISION 1
#define cublasXgemm cublasDgemm
#define gpublasXgemm gpublasDgemm
#include "../general/precision_macros.h"
#include "cannon_forw_template.h"
#include "cannon_back_template.h"
#undef cublasXgemm
#undef gpublasXgemm
#undef DOUBLE_PRECISION
#undef REALCASE

/*
!f> interface
!f>   subroutine cannons_reduction_d(A, U, local_rowsCast, local_colsCast, a_desc, Res, toStore, row_comm, col_comm, &
!f>                                  wantDebug, useGPU, gpublasHandle) &
!f>                             bind(C, name="cannons_reduction_c_d")
!f>     use precision
!f>     use, intrinsic :: iso_c_binding
!f>     implicit none
!f>     real(c_double)                :: A(local_rowsCast, local_colsCast), U(local_rowsCast, local_colsCast)
!f>     real(c_double)                :: Res(local_rowsCast, local_colsCast)
!f>     integer(kind=BLAS_KIND)       :: a_desc(9)
!f>     integer(kind=c_int), value    :: local_rowsCast, local_colsCast
!f>     integer(kind=MPI_KIND), value :: row_comm, col_comm, ToStore
!f>     integer(kind=c_int), value    :: wantDebug, useGPU
!f>     integer(kind=c_intptr_t)      :: gpublasHandle
!f>   end subroutine
!f> end interface
*/
void cannons_reduction_c_d(double* A, double* U, int local_rowsCast, int local_colsCast, C_INT_TYPE_PTR a_desc,
                           double *Res, C_INT_MPI_TYPE ToStore, C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                           int wantDebug, int useGPU, intptr_t *gpublasHandle);

/*
!f> interface
!f>   subroutine cannons_triang_rectangular_d(U, B, local_rowsCast, local_colsCast, u_desc, b_desc, Res, row_comm, col_comm, &
!f>                                           wantDebug, useGPU, gpublasHandle) &
!f>                             bind(C, name="cannons_triang_rectangular_c_d")
!f>     use precision
!f>     use, intrinsic :: iso_c_binding
!f>     implicit none
!f>     real(c_double)                :: U(local_rowsCast, local_colsCast), B(local_rowsCast, local_colsCast)
!f>     real(c_double)                :: Res(local_rowsCast, local_colsCast)
!f>     integer(kind=BLAS_KIND)       :: u_desc(9), b_desc(9)
!f>     integer(kind=c_int), value    :: local_rowsCast, local_colsCast
!f>     integer(kind=MPI_KIND), value :: row_comm, col_comm
!f>     integer(kind=c_int), value    :: wantDebug, useGPU
!f>     integer(kind=c_intptr_t)      :: gpublasHandle
!f>   end subroutine
!f> end interface
*/
void cannons_triang_rectangular_c_d(double* U, double* B, int local_rowsCast, int local_colsCast,
                                    C_INT_TYPE_PTR u_desc, C_INT_TYPE_PTR b_desc, double *Res, C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                                    int wantDebug, int useGPU, intptr_t *gpublasHandle);

//***********************************************************************************************************

#define REALCASE 1
#define SINGLE_PRECISION 1
#define cublasXgemm cublasSgemm
#define gpublasXgemm gpublasSgemm
#include "../general/precision_macros.h"
#include "cannon_forw_template.h"
#include "cannon_back_template.h"
#undef cublasXgemm
#undef gpublasXgemm
#undef SINGLE_PRECISION
#undef REALCASE

/*
!f> interface
!f>   subroutine cannons_reduction_f(A, U, local_rowsCast, local_colsCast, a_desc, Res, toStore, row_comm, col_comm,&
!f>                                  wantDebug, useGPU, gpublasHandle) &
!f>                             bind(C, name="cannons_reduction_c_f")
!f>     use precision
!f>     use, intrinsic :: iso_c_binding
!f>     implicit none
!f>     real(c_float)                 :: A(local_rowsCast, local_colsCast), U(local_rowsCast, local_colsCast)
!f>     real(c_float)                 :: Res(local_rowsCast, local_colsCast)
!f>     !type(c_ptr), value           :: A, U, Res
!f>     integer(kind=BLAS_KIND)       :: a_desc(9)
!f>     integer(kind=c_int), value    :: local_rowsCast, local_colsCast
!f>     integer(kind=MPI_KIND), value :: row_comm, col_comm, ToStore
!f>     integer(kind=c_int), value    :: wantDebug, useGPU
!f>     integer(kind=c_intptr_t)      :: gpublasHandle
!f>   end subroutine
!f> end interface
*/
void cannons_reduction_c_f(float* A, float* U, int local_rowsCast, int local_colsCast, C_INT_TYPE_PTR a_desc,
                           float *Res, C_INT_MPI_TYPE ToStore, C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                           int wantDebug, int useGPU, intptr_t *gpublasHandle);

/*
!f> interface
!f>   subroutine cannons_triang_rectangular_f(U, B, local_rowsCast, local_colsCast, u_desc, b_desc, Res, row_comm, col_comm, &
!f>                                           wantDebug, useGPU, gpublasHandle) &
!f>                             bind(C, name="cannons_triang_rectangular_c_f")
!f>     use precision
!f>     use, intrinsic :: iso_c_binding
!f>     implicit none
!f>     real(c_float)                 :: U(local_rowsCast, local_colsCast), B(local_rowsCast, local_colsCast)
!f>     real(c_float)                 :: Res(local_rowsCast, local_colsCast)
!f>     integer(kind=BLAS_KIND)       :: u_desc(9), b_desc(9)
!f>     integer(kind=c_int), value    :: local_rowsCast, local_colsCast
!f>     integer(kind=MPI_KIND), value :: row_comm, col_comm
!f>     integer(kind=c_int), value    :: wantDebug, useGPU
!f>     integer(kind=c_intptr_t)      :: gpublasHandle
!f>   end subroutine
!f> end interface
*/
void cannons_triang_rectangular_c_f(float* U, float* B, int local_rowsCast, int local_colsCast,
                                    C_INT_TYPE_PTR u_desc, C_INT_TYPE_PTR b_desc, float *Res, C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                                    int wantDebug, int useGPU, intptr_t *gpublasHandle);

//***********************************************************************************************************

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#define cublasXgemm cublasZgemm
#define gpublasXgemm gpublasZgemm
#include "../general/precision_macros.h"
#include "cannon_forw_template.h"
#include "cannon_back_template.h"
#undef cublasXgemm
#undef gpublasXgemm
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

/*
!f> interface
!f>   subroutine cannons_reduction_dc(A, U, local_rowsCast, local_colsCast, a_desc, Res, toStore, row_comm, col_comm, &
!f>                                   wantDebug, useGPU, gpublasHandle) &
!f>                             bind(C, name="cannons_reduction_c_dc")
!f>     use precision
!f>     use, intrinsic :: iso_c_binding
!f>     implicit none
!f>     complex(c_double)             :: A(local_rowsCast, local_colsCast), U(local_rowsCast, local_colsCast)
!f>     complex(c_double)             :: Res(local_rowsCast, local_colsCast)
!f>     integer(kind=BLAS_KIND)       :: a_desc(9)
!f>     integer(kind=c_int), value    :: local_rowsCast, local_colsCast
!f>     integer(kind=MPI_KIND), value :: row_comm, col_comm, ToStore
!f>     integer(kind=c_int), value    :: wantDebug, useGPU
!f>     integer(kind=c_intptr_t)      :: gpublasHandle
!f>   end subroutine
!f> end interface
*/
void cannons_reduction_c_dc(double complex* A, double complex* U, int local_rowsCast, int local_colsCasr, C_INT_TYPE_PTR a_desc,
                            double complex *Res, C_INT_MPI_TYPE ToStore, C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                            int wantDebug, int useGPU, intptr_t *gpublasHandle);

/*
!f> interface
!f>   subroutine cannons_triang_rectangular_dc(U, B, local_rowsCast, local_colsCast, u_desc, b_desc, Res, row_comm, col_comm, &
!f>                                            wantDebug, useGPU, gpublasHandle) &
!f>                             bind(C, name="cannons_triang_rectangular_c_dc")
!f>     use precision
!f>     use, intrinsic :: iso_c_binding
!f>     implicit none
!f>     complex(c_double)             :: U(local_rowsCast, local_colsCast), B(local_rowsCast, local_colsCast)
!f>     complex(c_double)             :: Res(local_rowsCast, local_colsCast)
!f>     integer(kind=BLAS_KIND)       :: u_desc(9), b_desc(9)
!f>     integer(kind=c_int), value    :: local_rowsCast, local_colsCast
!f>     integer(kind=MPI_KIND), value :: row_comm, col_comm
!f>     integer(kind=c_int), value    :: wantDebug, useGPU
!f>     integer(kind=c_intptr_t)      :: gpublasHandle
!f>   end subroutine
!f> end interface
*/
void cannons_triang_rectangular_c_dc(double complex* U, double complex* B, int local_rowsCast, int local_colsCast,
                                     C_INT_TYPE_PTR u_desc, C_INT_TYPE_PTR b_desc, double complex *Res, C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                                     int wantDebug, int useGPU, intptr_t *gpublasHandle);
//***********************************************************************************************************

#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#define cublasXgemm cublasCgemm
#define gpublasXgemm gpublasCgemm
#include "../general/precision_macros.h"
#include "cannon_forw_template.h"
#include "cannon_back_template.h"
#undef cublasXgemm
#undef gpublasXgemm
#undef SINGLE_PRECISION
#undef COMPLEXCASE

/*
!f> interface
!f>   subroutine cannons_reduction_fc(A, U, local_rowsCast, local_colsCast, a_desc, Res, toStore, row_comm, col_comm, &
!f>                                   wantDebug, useGPU, gpublasHandle) &
!f>                             bind(C, name="cannons_reduction_c_fc")
!f>     use precision
!f>     use, intrinsic :: iso_c_binding
!f>     implicit none
!f>     complex(c_float)              :: A(local_rowsCast, local_colsCast), U(local_rowsCast, local_colsCast)
!f>     complex(c_float)              :: Res(local_rowsCast, local_colsCast)
!f>     !type(c_ptr), value           :: A, U, Res
!f>     integer(kind=BLAS_KIND)       :: a_desc(9)
!f>     integer(kind=c_int), value    :: local_rowsCast, local_colsCast
!f>     integer(kind=MPI_KIND), value :: row_comm, col_comm, ToStore
!f>     integer(kind=c_int), value    :: wantDebug, useGPU
!f>     integer(kind=c_intptr_t)      :: gpublasHandle
!f>   end subroutine
!f> end interface
*/

void cannons_reduction_c_fc(float complex* A, float complex* U, int local_rowsCast, int local_colsCast, C_INT_TYPE_PTR a_desc,
                            float complex *Res, C_INT_MPI_TYPE ToStore, C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                            int wantDebug, int useGPU, intptr_t *gpublasHandle);

/*
!f> interface
!f>   subroutine cannons_triang_rectangular_fc(U, B, local_rowsCast, local_colsCast, u_desc, b_desc, Res, row_comm, col_comm, &
!f>                                            wantDebug, useGPU, gpublasHandle) &
!f>                             bind(C, name="cannons_triang_rectangular_c_fc")
!f>     use precision
!f>     use, intrinsic :: iso_c_binding
!f>     implicit none
!f>     complex(c_float)              :: U(local_rowsCast, local_colsCast), B(local_rowsCast, local_colsCast)
!f>     complex(c_float)              :: Res(local_rowsCast, local_colsCast)
!f>     integer(kind=BLAS_KIND)       :: u_desc(9), b_desc(9)
!f>     integer(kind=c_int), value    :: local_rowsCast, local_colsCast
!f>     integer(kind=MPI_KIND), value :: row_comm, col_comm
!f>     integer(kind=c_int), value    :: wantDebug, useGPU
!f>     integer(kind=c_intptr_t)      :: gpublasHandle
!f>   end subroutine
!f> end interface
*/
void cannons_triang_rectangular_c_fc(float complex* U, float complex* B, int local_rowsCast, int local_colsCast,
                                     C_INT_TYPE_PTR u_desc, C_INT_TYPE_PTR b_desc, float complex *Res, C_INT_MPI_TYPE row_comm, C_INT_MPI_TYPE col_comm,
                                     int wantDebug, int useGPU, intptr_t *gpublasHandle);
#endif
