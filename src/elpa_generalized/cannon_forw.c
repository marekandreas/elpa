#include "config-f90.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// most of the file is not compiled if not using MPI
#ifdef WITH_MPI
#include <mpi.h>

//#include <elpa/elpa.h>
//#include <elpa/elpa_generated.h>
//#include <elpa/elpa_constants.h>
//#include <elpa/elpa_generated_legacy.h>
//#include <elpa/elpa_generic.h>
//#include <elpa/elpa_legacy.h>
//
void pdlacpy_(char*, int*, int*, double*, int*, int*, int*, double*, int*, int*, int*);
void dlacpy_(char*, int*, int*, double*, int*, double*, int*);
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 
void pdtran_(int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
//void pdelset_(double*, int*, int*, int*, double*);
//void pdsymm_(char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
//void pdpotrf_(char*, int*, double*, int*, int*, int*, int*);
//void pdsyngst_(int*, char*, int*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*);
//void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
int numroc_(int*, int*, int*, int*, int*);
//void set_up_blacsgrid_f1(int, int*, int*, int*, int*, int*, int*, int*);
//void pdtrtrs_(char*, char*, char*, int*, int*, double*, int*, int*, int*, double*, int*, int*, int*, int*);
//void pdsyevr_(char*, char*, char*, int*, double*, int*, int*, int*, int*, int*, int*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, int*);

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "cannon_forw_template.c"
#undef DOUBLE_PRECISION
#undef REALCASE

//***********************************************************************************************************
/*
!f> interface
!f>   subroutine cannons_reduction(A, U, local_rows, local_cols, a_desc, Res, toStore, row_comm, col_comm) &
!f>                             bind(C, name="cannons_reduction_c_d")
!f>     use, intrinsic :: iso_c_binding
!f>     real(c_double)                        :: A(local_rows, local_cols), U(local_rows, local_cols), Res(local_rows, local_cols)
!f>     !type(c_ptr), value                   :: A, U, Res
!f>     integer(kind=c_int)                   :: a_desc(9)
!f>     integer(kind=c_int),value             :: local_rows, local_cols
!f>     integer(kind=c_int),value     ::  row_comm, col_comm, ToStore
!f>   end subroutine
!f> end interface
*/
void cannons_reduction_c_d(double* A, double* U, int local_rows, int local_cols, int* a_desc,
                         double *Res, int ToStore, int row_comm, int col_comm);

#else
void cannons_reduction_c_d(double* A, double* U, int local_rows, int local_cols, int* a_desc,
                         double *Res, int ToStore, int row_comm, int col_comm)
{
}

#endif
