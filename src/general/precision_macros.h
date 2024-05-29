#if 0
!    Copyright 2016, P. Kus
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
!
#endif

#ifdef REALCASE
#undef DOUBLE_PRECISION_REAL
#undef SINGLE_PRECSION_REAL
#undef  MATH_DATATYPE
#undef  BLAS_TRANS_OR_CONJ
#undef  BLAS_CHAR
#undef  BLAS_CHAR_AND_SY_OR_HE
#undef  PRECISION_CHAR
#undef  PRECISION
#undef  SPECIAL_COMPLEX_DATATYPE
#undef  PRECISION_STR
#undef  REAL_DATATYPE
#undef  C_REAL_DATATYPE

#undef C_GEMM
#undef C_LACPY
#undef C_PLACPY
#undef C_PTRAN

#undef  PRECISION_TRTRI
#undef  PRECISION_POTRF
#undef  PRECISION_TRSM
#undef  PRECISION_GEMV
#undef  PRECISION_GERC
#undef  PRECISION_TRMV
#undef  PRECISION_GEMM
#undef  PRECISION_TRMM
#undef  PRECISION_HERK
#undef  PRECISION_SYRK
#undef  PRECISION_SYMV
#undef  PRECISION_SYMM
#undef  PRECISION_HEMV
#undef  PRECISION_HER2
#undef  PRECISION_SYR2
#undef  PRECISION_SYR2K
#undef  PRECISION_GEQRF
#undef  PRECISION_STEDC
#undef  PRECISION_STEQR
#undef  PRECISION_LAMRG
#undef  PRECISION_LAMCH
#undef  PRECISION_LAPY2
#undef  PRECISION_LAED4
#undef  PRECISION_LAED5
#undef  PRECISION_NRM2
#undef  PRECISION_LASET
#undef  PRECISION_SCAL
#undef  PRECISION_COPY
#undef  PRECISION_AXPY
#undef  PRECISION_GER
#undef  gpublas_PRECISION_GEMM
#undef  gpublas_PRECISION_TRMM
#undef  gpublas_PRECISION_GEMV
#undef  gpublas_PRECISION_SYMV
#undef  gpublas_PRECISION_COPY
#undef  gpublas_PRECISION_TRSM
#undef  gpublas_PRECISION_DOT
#undef  gpublas_PRECISION_SCAL
#undef  gpusolver_PRECISION_TRTRI
#undef  gpusolver_PRECISION_POTRF
#undef  cublas_PRECISION_GEMM
#undef  cublas_PRECISION_TRMM
#undef  cublas_PRECISION_GEMV
#undef  cublas_PRECISION_SYMV
#undef  cublas_PRECISION_COPY
#undef  gpu_copy_PRECISION_a_tmp1
#undef  gpu_copy_PRECISION_tmp1_tmp2
#undef  gpu_copy_PRECISION_a_tmat2
#undef  gpu_copy_PRECISION_tmp2_tmat2
#undef  gpu_copy_PRECISION_a_tmat1
#undef  gpu_copy_PRECISION_a_tmatc
#undef  gpu_copy_PRECISION_a_aux_bc
#undef  gpu_copy_PRECISION_aux_bc_aux_mat
#undef  gpu_copy_PRECISION_tmp2_c
#undef  gpu_copy_and_set_zeros_PRECISION
#undef  gpu_dot_product_PRECISION
#undef  gpu_dot_product_and_assign_PRECISION
#undef  gpu_set_e_vec_scale_set_one_store_v_row_PRECISION
#undef  gpu_store_u_v_in_uv_vu_PRECISION
#undef  gpu_update_matrix_element_add_PRECISION
#undef  gpu_update_array_element_add_PRECISION
#undef  gpu_hh_transform_PRECISION
#undef  gpu_transpose_reduceadd_vectors_copy_block_PRECISION
#undef  mkl_offload_PRECISION_GEMM
#undef  mkl_offload_PRECISION_GEMV
#undef  mkl_offload_PRECISION_TRMM
#undef  scal_PRECISION_GEMM
#undef  scal_PRECISION_NRM2
#undef  scal_PRECISION_LASET
#undef  scal_PRECISION_GEMR2D
#undef  PRECISION_SUFFIX
#undef  ELPA_IMPL_SUFFIX

#undef ELPA_PRECISION_SSMV
#undef ELPA_PRECISION_SSR2

#undef  MPI_REAL_PRECISION
#undef  MPI_MATH_DATATYPE_PRECISION
#undef  MPI_MATH_DATATYPE_PRECISION_C
#undef  MPI_MATH_DATATYPE_PRECISION_EXPL
#undef  C_DATATYPE_KIND

#undef  GPU_ZERO_SKEWSYMMETRIC_Q_PRECISION
#undef  GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION
#undef  GPU_COPY_SKEWSYMMETRIC_FIRST_HALF_Q_PRECISION
#undef  GPU_GET_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION
#undef  GPU_PUT_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION

#if 0
/* General definitions needed in single and double case */
/* the if 0 bracket is just to make the IBM Fortran compiler happy */
#endif

#define  MATH_DATATYPE real
#define  BLAS_TRANS_OR_CONJ 'T'

#ifdef DOUBLE_PRECISION
#define DOUBLE_PRECISION_REAL
#define  PRECISION double
#define  PRECISION_STR 'double'
#define  PRECISION_SUFFIX "_double"
#define  ELPA_IMPL_SUFFIX d
#define  REAL_DATATYPE rk8
#define  C_REAL_DATATYPE c_double
#define  BLAS_CHAR D
#define  BLAS_CHAR_AND_SY_OR_HE DSY
#define  PRECISION_CHAR 'D'
#define  SPECIAL_COMPLEX_DATATYPE ck8

#define  PRECISION_TRTRI DTRTRI
#define  PRECISION_POTRF DPOTRF
#define  PRECISION_TRSM DTRSM
#define  PRECISION_GEMV DGEMV
#define  PRECISION_GERC DGER
#define  PRECISION_TRMV DTRMV
#define  PRECISION_GEMM DGEMM
#define  PRECISION_TRMM DTRMM
#define  PRECISION_HERK DHERK
#define  PRECISION_SYRK DSYRK
#define  PRECISION_SYMV DSYMV
#define  PRECISION_SYMM DSYMM
#define  PRECISION_HEMV DHEMV
#define  PRECISION_HER2 DHER2
#define  PRECISION_SYR2 DSYR2
#define  PRECISION_SYR2K DSYR2K
#define  PRECISION_GEQRF DGEQRF
#define  PRECISION_STEDC DSTEDC
#define  PRECISION_STEQR DSTEQR
#define  PRECISION_LAMRG DLAMRG
#define  PRECISION_LAMCH DLAMCH
#define  PRECISION_LAPY2 DLAPY2
#define  PRECISION_LAED4 DLAED4
#define  PRECISION_LAED5 DLAED5
#define  PRECISION_NRM2 DNRM2
#define  PRECISION_LASET DLASET
#define  PRECISION_GER DGER
#define  PRECISION_SCAL DSCAL
#define  PRECISION_COPY DCOPY
#define  PRECISION_AXPY DAXPY
#define  gpublas_PRECISION_GEMM gpublas_DGEMM
#define  gpublas_PRECISION_TRMM gpublas_DTRMM
#define  gpublas_PRECISION_GEMV gpublas_DGEMV
#define  gpublas_PRECISION_SYMV gpublas_DSYMV
#define  gpublas_PRECISION_COPY gpublas_DCOPY
#define  gpublas_PRECISION_TRSM gpublas_DTRSM
#define  gpublas_PRECISION_DOT  gpublas_DDOT
#define  gpublas_PRECISION_SCAL gpublas_DSCAL
#define  gpusolver_PRECISION_TRTRI gpusolver_DTRTRI
#define  gpusolver_PRECISION_POTRF gpusolver_DPOTRF
#define  cublas_PRECISION_GEMM cublas_DGEMM
#define  cublas_PRECISION_TRMM cublas_DTRMM
#define  cublas_PRECISION_GEMV cublas_DGEMV
#define  cublas_PRECISION_SYMV cublas_DSYMV
#define  cublas_PRECISION_COPY cublas_DCOPY
#define  gpu_copy_PRECISION_a_tmp1 gpu_copy_double_a_tmp1
#define  gpu_copy_PRECISION_tmp1_tmp2 gpu_copy_double_tmp1_tmp2
#define  gpu_copy_PRECISION_a_tmat2 gpu_copy_double_a_tmat2
#define  gpu_copy_PRECISION_tmp2_tmat2 gpu_copy_double_tmp2_tmat2
#define  gpu_copy_PRECISION_a_tmat1 gpu_copy_double_a_tmat1
#define  gpu_copy_PRECISION_a_tmatc gpu_copy_double_a_tmatc
#define  gpu_copy_PRECISION_a_aux_bc gpu_copy_double_a_aux_bc
#define  gpu_copy_PRECISION_aux_bc_aux_mat gpu_copy_double_aux_bc_aux_mat
#define  gpu_copy_PRECISION_tmp2_c gpu_copy_double_tmp2_c
#define  gpu_copy_and_set_zeros_PRECISION gpu_copy_and_set_zeros_double
#define  gpu_dot_product_PRECISION gpu_dot_product_double
#define  gpu_dot_product_and_assign_PRECISION gpu_dot_product_and_assign_double
#define  gpu_set_e_vec_scale_set_one_store_v_row_PRECISION gpu_set_e_vec_scale_set_one_store_v_row_double
#define  gpu_store_u_v_in_uv_vu_PRECISION gpu_store_u_v_in_uv_vu_double
#define  gpu_update_matrix_element_add_PRECISION gpu_update_matrix_element_add_double
#define  gpu_update_array_element_add_PRECISION gpu_update_array_element_add_double
#define  gpu_hh_transform_PRECISION gpu_hh_transform_double
#define  gpu_transpose_reduceadd_vectors_copy_block_PRECISION gpu_transpose_reduceadd_vectors_copy_block_double
#define  mkl_offload_PRECISION_GEMM mkl_offload_DGEMM
#define  mkl_offload_PRECISION_GEMV mkl_offload_DGEMV
#define  mkl_offload_PRECISION_TRMM mkl_offload_DTRMM
#define  scal_PRECISION_GEMM PDGEMM
#define  scal_PRECISION_NRM2 PDNRM2
#define  scal_PRECISION_LASET PDLASET
#define  scal_PRECISION_GEMR2D PDGEMR2D
#define  MPI_REAL_PRECISION MPI_REAL8
#define  MPI_MATH_DATATYPE_PRECISION MPI_REAL8
#define  MPI_MATH_DATATYPE_PRECISION_C MPI_DOUBLE
#define  MPI_MATH_DATATYPE_PRECISION_EXPL MPI_REAL8
#define  C_DATATYPE_KIND c_double

#define ELPA_PRECISION_SSMV elpa_dssmv
#define ELPA_PRECISION_SSR2 elpa_dssr2

#define C_GEMM dgemm_
#define C_LACPY dlacpy_
#define C_PLACPY pdlacpy_
#define C_PTRAN pdtran_

#define GPU_ZERO_SKEWSYMMETRIC_Q_PRECISION gpu_zero_skewsymmetric_q_double
#define GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION gpu_copy_skewsymmetric_second_half_q_double
#define GPU_COPY_SKEWSYMMETRIC_FIRST_HALF_Q_PRECISION gpu_copy_skewsymmetric_first_half_q_double
#define GPU_GET_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION gpu_get_skewsymmetric_second_half_q_double
#define GPU_PUT_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION gpu_put_skewsymmetric_second_half_q_double

#endif /* DOUBLE_PRECISION */

#ifdef SINGLE_PRECISION

#define SINGLE_PRECISION_REAL

#define  PRECISION single
#define  PRECISION_STR 'single'
#define  PRECISION_SUFFIX "_single"
#define  ELPA_IMPL_SUFFIX f
#define  REAL_DATATYPE rk4
#define  C_REAL_DATATYPE c_float
#define  BLAS_CHAR S
#define  BLAS_CHAR_AND_SY_OR_HE SSY
#define  PRECISION_CHAR 'S'
#define  SPECIAL_COMPLEX_DATATYPE ck4

#define  PRECISION_TRTRI STRTRI
#define  PRECISION_POTRF SPOTRF
#define  PRECISION_TRSM STRSM
#define  PRECISION_GEMV SGEMV
#define  PRECISION_GERC SGER
#define  PRECISION_TRMV STRMV
#define  PRECISION_GEMM SGEMM
#define  PRECISION_TRMM STRMM
#define  PRECISION_HERK SHERK
#define  PRECISION_SYRK SSYRK
#define  PRECISION_SYMV SSYMV
#define  PRECISION_SYMM SSYMM
#define  PRECISION_HEMV SHEMV
#define  PRECISION_HER2 SHER2
#define  PRECISION_SYR2 SSYR2
#define  PRECISION_SYR2K SSYR2K
#define  PRECISION_GEQRF SGEQRF
#define  PRECISION_STEDC SSTEDC
#define  PRECISION_STEQR SSTEQR
#define  PRECISION_LAMRG SLAMRG
#define  PRECISION_LAMCH SLAMCH
#define  PRECISION_LAPY2 SLAPY2
#define  PRECISION_LAED4 SLAED4
#define  PRECISION_LAED5 SLAED5
#define  PRECISION_NRM2 SNRM2
#define  PRECISION_LASET SLASET
#define  PRECISION_GER SGER
#define  PRECISION_SCAL SSCAL
#define  PRECISION_COPY SCOPY
#define  PRECISION_AXPY SAXPY
#define  gpublas_PRECISION_GEMM gpublas_SGEMM
#define  gpublas_PRECISION_TRMM gpublas_STRMM
#define  gpublas_PRECISION_GEMV gpublas_SGEMV
#define  gpublas_PRECISION_SYMV gpublas_SSYMV
#define  gpublas_PRECISION_COPY gpublas_SCOPY
#define  gpublas_PRECISION_TRSM gpublas_STRSM
#define  gpublas_PRECISION_DOT  gpublas_SDOT
#define  gpublas_PRECISION_SCAL gpublas_SSCAL
#define  gpusolver_PRECISION_TRTRI gpusolver_STRTRI
#define  gpusolver_PRECISION_POTRF gpusolver_SPOTRF
#define  cublas_PRECISION_GEMM cublas_SGEMM
#define  cublas_PRECISION_TRMM cublas_STRMM
#define  cublas_PRECISION_GEMV cublas_SGEMV
#define  cublas_PRECISION_SYMV cublas_SSYMV
#define  cublas_PRECISION_COPY cublas_SCOPY
#define  gpu_copy_PRECISION_a_tmp1 gpu_copy_float_a_tmp1
#define  gpu_copy_PRECISION_tmp1_tmp2 gpu_copy_float_tmp1_tmp2
#define  gpu_copy_PRECISION_a_tmat2 gpu_copy_float_a_tmat2
#define  gpu_copy_PRECISION_tmp2_tmat2 gpu_copy_float_tmp2_tmat2
#define  gpu_copy_PRECISION_a_tmat1 gpu_copy_float_a_tmat1
#define  gpu_copy_PRECISION_a_tmatc gpu_copy_float_a_tmatc
#define  gpu_copy_PRECISION_a_aux_bc gpu_copy_float_a_aux_bc
#define  gpu_copy_PRECISION_aux_bc_aux_mat gpu_copy_float_aux_bc_aux_mat
#define  gpu_copy_PRECISION_tmp2_c gpu_copy_float_tmp2_c
#define  gpu_copy_and_set_zeros_PRECISION gpu_copy_and_set_zeros_float
#define  gpu_dot_product_PRECISION gpu_dot_product_float
#define  gpu_dot_product_and_assign_PRECISION gpu_dot_product_and_assign_float
#define  gpu_set_e_vec_scale_set_one_store_v_row_PRECISION gpu_set_e_vec_scale_set_one_store_v_row_float
#define  gpu_store_u_v_in_uv_vu_PRECISION gpu_store_u_v_in_uv_vu_float
#define  gpu_update_matrix_element_add_PRECISION gpu_update_matrix_element_add_float
#define  gpu_update_array_element_add_PRECISION gpu_update_array_element_add_float
#define  gpu_hh_transform_PRECISION gpu_hh_transform_float
#define  gpu_transpose_reduceadd_vectors_copy_block_PRECISION gpu_transpose_reduceadd_vectors_copy_block_float
#define  mkl_offload_PRECISION_GEMM mkl_offload_SGEMM
#define  mkl_offload_PRECISION_GEMV mkl_offload_SGEMV
#define  mkl_offload_PRECISION_TRMM mkl_offload_STRMM
#define  scal_PRECISION_GEMM PSGEMM
#define  scal_PRECISION_NRM2 PSNRM2
#define  scal_PRECISION_LASET PSLASET
#define  scal_PRECISION_GEMR2D PSGEMR2D
#define  MPI_REAL_PRECISION MPI_REAL4
#define  MPI_MATH_DATATYPE_PRECISION MPI_REAL4
#define  MPI_MATH_DATATYPE_PRECISION_C MPI_FLOAT
#define  MPI_MATH_DATATYPE_PRECISION_EXPL MPI_REAL4
#define  C_DATATYPE_KIND c_float

#define ELPA_PRECISION_SSMV elpa_sssmv
#define ELPA_PRECISION_SSR2 elpa_sssr2

#define C_GEMM sgemm_
#define C_LACPY slacpy_
#define C_PLACPY pslacpy_
#define C_PTRAN pstran_

#define GPU_ZERO_SKEWSYMMETRIC_Q_PRECISION gpu_zero_skewsymmetric_q_float
#define GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION gpu_copy_skewsymmetric_second_half_q_float
#define GPU_COPY_SKEWSYMMETRIC_FIRST_HALF_Q_PRECISION gpu_copy_skewsymmetric_first_half_q_float
#define GPU_GET_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION gpu_get_skewsymmetric_second_half_q_float
#define GPU_PUT_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION gpu_put_skewsymmetric_second_half_q_float

#endif /* SINGLE_PRECISION */

#endif /* REALCASE */

#ifdef COMPLEXCASE

#undef DOUBLE_PRECISION_COMPLEX
#undef SINGLE_PRECISION_COMPLEX
#undef  MATH_DATATYPE
#undef  BLAS_TRANS_OR_CONJ
#undef  BLAS_CHAR
#undef  BLAS_CHAR_AND_SY_OR_HE
#undef  PRECISION_CHAR
#undef  PRECISION
#undef COMPLEX_DATATYPE

#if 0
/* in the complex case also sometime real valued variables are needed */
/* the if 0 bracket is just to make the IBM Fortran compiler happy */
#endif

#undef REAL_DATATYPE
#undef C_REAL_DATATYPE

#undef C_GEMM
#undef C_LACPY
#undef C_PLACPY
#undef C_PTRAN

#undef  PRECISION_TRTRI
#undef  PRECISION_POTRF
#undef  PRECISION_TRSM
#undef  PRECISION_STR
#undef  PRECISION_GEMV
#undef  PRECISION_GERC
#undef  PRECISION_TRMV
#undef  PRECISION_GEMM
#undef  PRECISION_TRMM
#undef  PRECISION_HERK
#undef  PRECISION_SYRK
#undef  PRECISION_SYMV
#undef  PRECISION_SYMM
#undef  PRECISION_HEMV
#undef  PRECISION_HER2
#undef  PRECISION_SYR2
#undef  PRECISION_SYR2K
#undef  PRECISION_GEQRF
#undef  PRECISION_STEDC
#undef  PRECISION_STEQR
#undef  PRECISION_LAMRG
#undef  PRECISION_LAMCH
#undef  PRECISION_LAPY2
#undef  PRECISION_LAED4
#undef  PRECISION_LAED5
#undef  PRECISION_DOTC
#undef  PRECISION_LASET
#undef  PRECISION_GER
#undef  PRECISION_SCAL
#undef  PRECISION_COPY
#undef  PRECISION_AXPY
#undef  gpublas_PRECISION_GEMM
#undef  gpublas_PRECISION_TRMM
#undef  gpublas_PRECISION_GEMV
#undef  gpublas_PRECISION_SYMV
#undef  gpublas_PRECISION_COPY
#undef  gpublas_PRECISION_TRSM
#undef  gpublas_PRECISION_DCAL
#undef  gpublas_PRECISION_DOT
#undef  gpublas_PRECISION_SCAL
#undef  cublas_PRECISION_GEMM
#undef  cublas_PRECISION_TRMM
#undef  cublas_PRECISION_GEMV
#undef  cublas_PRECISION_SYMV
#undef  cublas_PRECISION_COPY
#undef  gpu_copy_PRECISION_a_tmp1
#undef  gpu_copy_PRECISION_tmp1_tmp2
#undef  gpu_copy_PRECISION_a_tmat2
#undef  gpu_copy_PRECISION_tmp2_tmat2
#undef  gpu_copy_PRECISION_a_tmat1
#undef  gpu_copy_PRECISION_a_tmatc
#undef  gpu_copy_PRECISION_a_aux_bc
#undef  gpu_copy_PRECISION_aux_bc_aux_mat
#undef  gpu_copy_PRECISION_tmp2_c
#undef  gpu_copy_and_set_zeros_PRECISION
#undef  gpu_dot_product_PRECISION
#undef  gpu_dot_product_and_assign_PRECISION
#undef  gpu_set_e_vec_scale_set_one_store_v_row_PRECISION
#undef  gpu_store_u_v_in_uv_vu_PRECISION
#undef  gpu_update_matrix_element_add_PRECISION
#undef  gpu_update_array_element_add_PRECISION
#undef  gpu_hh_transform_PRECISION
#undef  gpu_transpose_reduceadd_vectors_copy_block_PRECISION
#undef  gpusolver_PRECISION_TRTRI
#undef  gpusolver_PRECISION_POTRF
#undef  mkl_offload_PRECISION_GEMM 
#undef  mkl_offload_PRECISION_GEMV 
#undef  mkl_offload_PRECISION_TRMM 
#undef  scal_PRECISION_GEMM
#undef  scal_PRECISION_DOTC
#undef  scal_PRECISION_LASET
#undef  scal_PRECISION_GEMR2D
#undef  PRECISION_SUFFIX
#undef  ELPA_IMPL_SUFFIX
#undef  MPI_COMPLEX_PRECISION
#undef  MPI_MATH_DATATYPE_PRECISION
#undef  MPI_MATH_DATATYPE_PRECISION_C
#undef  MPI_MATH_DATATYPE_PRECISION_EXPL
#undef  MPI_COMPLEX_EXPLICIT_PRECISION
#undef  MPI_REAL_PRECISION
#undef  KIND_PRECISION
#undef  PRECISION_CMPLX
#undef  PRECISION_IMAG
#undef  PRECISION_REAL
#undef  C_DATATYPE_KIND

#undef ELPA_PRECISION_SSMV
#undef ELPA_PRECISION_SSR2

#undef  GPU_SCALE_QMAT_PRECISION
#undef  GPU_COPY_REAL_PART_TO_Q_PRECISION

#if 0
/* General definitions needed in single and double case */
/* the if 0 bracket is just to make the IBM Fortran compiler happy */
#endif

#define  MATH_DATATYPE complex
#define  BLAS_TRANS_OR_CONJ 'C'
#ifdef DOUBLE_PRECISION

#define DOUBLE_PRECISION_COMPLEX
#define  PRECISION double
#define  PRECISION_STR 'double'
#define  PRECISION_SUFFIX "_double"
#define  ELPA_IMPL_SUFFIX dc
#define COMPLEX_DATATYPE CK8
#define BLAS_CHAR Z
#define BLAS_CHAR_AND_SY_OR_HE ZHE
#define PRECISION_CHAR 'Z'
#define REAL_DATATYPE RK8
#define C_REAL_DATATYPE c_double

#define C_GEMM zgemm_
#define C_LACPY zlacpy_
#define C_PLACPY pzlacpy_
#define C_PTRAN pztranc_

#define  PRECISION_TRTRI ZTRTRI
#define  PRECISION_POTRF ZPOTRF
#define  PRECISION_TRSM ZTRSM
#define  PRECISION_GEMV ZGEMV
#define  PRECISION_GERC ZGERC
#define  PRECISION_TRMV ZTRMV
#define  PRECISION_GEMM ZGEMM
#define  PRECISION_TRMM ZTRMM
#define  PRECISION_HERK ZHERK
#define  PRECISION_SYRK ZSYRK
#define  PRECISION_SYMV ZSYMV
#define  PRECISION_SYMM ZSYMM
#define  PRECISION_HEMV ZHEMV
#define  PRECISION_HER2 ZHER2
#define  PRECISION_SYR2 ZSYR2
#define  PRECISION_SYR2K ZSYR2K
#define  PRECISION_GEQRF ZGEQRF
#define  PRECISION_STEDC ZSTEDC
#define  PRECISION_STEQR ZSTEQR
#define  PRECISION_LAMRG ZLAMRG
#define  PRECISION_LAMCH ZLAMCH
#define  PRECISION_LAPY2 ZLAPY2
#define  PRECISION_LAED4 ZLAED4
#define  PRECISION_LAED5 ZLAED5
#define  PRECISION_DOTC ZDOTC
#define  PRECISION_LASET ZLASET
#define  PRECISION_GER ZGER
#define  PRECISION_SCAL ZSCAL
#define  PRECISION_COPY ZCOPY
#define  PRECISION_AXPY ZAXPY
#define  gpublas_PRECISION_GEMM gpublas_ZGEMM
#define  gpublas_PRECISION_TRMM gpublas_ZTRMM
#define  gpublas_PRECISION_GEMV gpublas_ZGEMV
#define  gpublas_PRECISION_SYMV gpublas_ZSYMV
#define  gpublas_PRECISION_COPY gpublas_ZCOPY
#define  gpublas_PRECISION_TRSM gpublas_ZTRSM
#define  gpublas_PRECISION_DOT gpublas_ZDOT
#define  gpublas_PRECISION_SCAL gpublas_ZSCAL
#define  gpusolver_PRECISION_TRTRI gpusolver_ZTRTRI
#define  gpusolver_PRECISION_POTRF gpusolver_ZPOTRF
#define  cublas_PRECISION_GEMM cublas_ZGEMM
#define  cublas_PRECISION_TRMM cublas_ZTRMM
#define  cublas_PRECISION_GEMV cublas_ZGEMV
#define  cublas_PRECISION_SYMV cublas_ZSYMV
#define  cublas_PRECISION_COPY cublas_ZCOPY
#define  gpu_copy_PRECISION_a_tmp1 gpu_copy_double_complex_a_tmp1
#define  gpu_copy_PRECISION_tmp1_tmp2 gpu_copy_double_complex_tmp1_tmp2
#define  gpu_copy_PRECISION_a_tmat2 gpu_copy_double_complex_a_tmat2
#define  gpu_copy_PRECISION_tmp2_tmat2 gpu_copy_double_complex_tmp2_tmat2
#define  gpu_copy_PRECISION_a_tmat1 gpu_copy_double_complex_a_tmat1
#define  gpu_copy_PRECISION_a_tmatc gpu_copy_double_complex_a_tmatc
#define  gpu_copy_PRECISION_a_aux_bc gpu_copy_double_complex_a_aux_bc
#define  gpu_copy_PRECISION_aux_bc_aux_mat gpu_copy_double_complex_aux_bc_aux_mat
#define  gpu_copy_PRECISION_tmp2_c gpu_copy_double_complex_tmp2_c
#define  gpu_copy_and_set_zeros_PRECISION gpu_copy_and_set_zeros_double_complex
#define  gpu_dot_product_PRECISION gpu_dot_product_double_complex
#define  gpu_dot_product_and_assign_PRECISION gpu_dot_product_and_assign_double_complex
#define  gpu_set_e_vec_scale_set_one_store_v_row_PRECISION gpu_set_e_vec_scale_set_one_store_v_row_double_complex
#define  gpu_store_u_v_in_uv_vu_PRECISION gpu_store_u_v_in_uv_vu_double_complex
#define  gpu_update_matrix_element_add_PRECISION gpu_update_matrix_element_add_double_complex
#define  gpu_update_array_element_add_PRECISION gpu_update_array_element_add_double_complex
#define  gpu_hh_transform_PRECISION gpu_hh_transform_double_complex
#define  gpu_transpose_reduceadd_vectors_copy_block_PRECISION gpu_transpose_reduceadd_vectors_copy_block_double_complex
#define  mkl_offload_PRECISION_GEMM mkl_offload_ZGEMM
#define  mkl_offload_PRECISION_GEMV mkl_offload_ZGEMV
#define  mkl_offload_PRECISION_TRMM mkl_offload_ZTRMM
#define  scal_PRECISION_GEMM PZGEMM
#define  scal_PRECISION_DOTC PZDOTC
#define  scal_PRECISION_LASET PZLASET
#define  scal_PRECISION_GEMR2D PZGEMR2D
#define  MPI_COMPLEX_PRECISION MPI_DOUBLE_COMPLEX
#define  MPI_MATH_DATATYPE_PRECISION MPI_DOUBLE_COMPLEX
#define  MPI_MATH_DATATYPE_PRECISION_C MPI_DOUBLE_COMPLEX
#define  MPI_MATH_DATATYPE_PRECISION_EXPL MPI_COMPLEX16
#define  MPI_COMPLEX_EXPLICIT_PRECISION MPI_COMPLEX16
#define  MPI_REAL_PRECISION MPI_REAL8
#define  KIND_PRECISION rk8
#define  PRECISION_CMPLX DCMPLX
#define  PRECISION_IMAG DIMAG
#define  PRECISION_REAL DREAL
#define  C_DATATYPE_KIND c_double

#define GPU_SCALE_QMAT_PRECISION gpu_scale_qmat_double_complex
#define GPU_COPY_REAL_PART_TO_Q_PRECISION gpu_copy_real_part_to_q_double_complex

#define ELPA_PRECISION_SSMV elpa_zssmv
#define ELPA_PRECISION_SSR2 elpa_zssr2

#endif /* DOUBLE PRECISION */

#ifdef SINGLE_PRECISION
#define SINGLE_PRECISION_COMPLEX
#define  PRECISION single
#define  PRECISION_STR 'single'
#define  PRECISION_SUFFIX "_single"
#define  ELPA_IMPL_SUFFIX fc
#define COMPLEX_DATATYPE CK4
#define BLAS_CHAR C
#define BLAS_CHAR_AND_SY_OR_HE CHE
#define PRECISION_CHAR 'C'
#define REAL_DATATYPE RK4
#define C_REAL_DATATYPE c_float

#define C_GEMM cgemm_
#define C_LACPY clacpy_
#define C_PLACPY pclacpy_
#define C_PTRAN pctranc_

#define  PRECISION_TRTRI CTRTRI
#define  PRECISION_POTRF CPOTRF
#define  PRECISION_TRSM CTRSM
#define  PRECISION_GEMV CGEMV
#define  PRECISION_GERC CGERC
#define  PRECISION_TRMV CTRMV
#define  PRECISION_GEMM CGEMM
#define  PRECISION_TRMM CTRMM
#define  PRECISION_HERK CHERK
#define  PRECISION_SYRK CSYRK
#define  PRECISION_SYMV CSYMV
#define  PRECISION_SYMM CSYMM
#define  PRECISION_HEMV CHEMV
#define  PRECISION_HER2 CHER2
#define  PRECISION_SYR2 CSYR2
#define  PRECISION_SYR2K CSYR2K
#define  PRECISION_GEQRF CGEQRF
#define  PRECISION_STEDC CSTEDC
#define  PRECISION_STEQR CSTEQR
#define  PRECISION_LAMRG CLAMRG
#define  PRECISION_LAMCH CLAMCH
#define  PRECISION_LAPY2 CLAPY2
#define  PRECISION_LAED4 CLAED4
#define  PRECISION_LAED5 CLAED5
#define  PRECISION_DOTC CDOTC
#define  PRECISION_LASET CLASET
#define  PRECISION_SCAL CSCAL
#define  PRECISION_COPY CCOPY
#define  PRECISION_AXPY CAXPY
#define  PRECISION_GER CGER
#define  gpublas_PRECISION_GEMM gpublas_CGEMM
#define  gpublas_PRECISION_TRMM gpublas_CTRMM
#define  gpublas_PRECISION_GEMV gpublas_CGEMV
#define  gpublas_PRECISION_SYMV gpublas_CSYMV
#define  gpublas_PRECISION_COPY gpublas_CCOPY
#define  gpublas_PRECISION_TRSM gpublas_CTRSM
#define  gpublas_PRECISION_DOT  gpublas_CDOT
#define  gpublas_PRECISION_SCAL gpublas_CSCAL
#define  gpusolver_PRECISION_TRTRI gpusolver_CTRTRI
#define  gpusolver_PRECISION_POTRF gpusolver_CPOTRF
#define  cublas_PRECISION_GEMM cublas_CGEMM
#define  cublas_PRECISION_TRMM cublas_CTRMM
#define  cublas_PRECISION_GEMV cublas_CGEMV
#define  cublas_PRECISION_SYMV cublas_CSYMV
#define  cublas_PRECISION_COPY cublas_CCOPY
#define  gpu_copy_PRECISION_a_tmp1 gpu_copy_float_complex_a_tmp1
#define  gpu_copy_PRECISION_tmp1_tmp2 gpu_copy_float_complex_tmp1_tmp2
#define  gpu_copy_PRECISION_a_tmat2 gpu_copy_float_complex_a_tmat2
#define  gpu_copy_PRECISION_tmp2_tmat2 gpu_copy_float_complex_tmp2_tmat2
#define  gpu_copy_PRECISION_a_tmat1 gpu_copy_float_complex_a_tmat1
#define  gpu_copy_PRECISION_a_tmatc gpu_copy_float_complex_a_tmatc
#define  gpu_copy_PRECISION_a_aux_bc gpu_copy_float_complex_a_aux_bc
#define  gpu_copy_PRECISION_aux_bc_aux_mat gpu_copy_float_complex_aux_bc_aux_mat
#define  gpu_copy_PRECISION_tmp2_c gpu_copy_float_complex_tmp2_c
#define  gpu_copy_and_set_zeros_PRECISION gpu_copy_and_set_zeros_float_complex
#define  gpu_dot_product_PRECISION gpu_dot_product_float_complex
#define  gpu_dot_product_and_assign_PRECISION gpu_dot_product_and_assign_float_complex
#define  gpu_set_e_vec_scale_set_one_store_v_row_PRECISION gpu_set_e_vec_scale_set_one_store_v_row_float_complex
#define  gpu_store_u_v_in_uv_vu_PRECISION gpu_store_u_v_in_uv_vu_float_complex
#define  gpu_update_matrix_element_add_PRECISION gpu_update_matrix_element_add_float_complex
#define  gpu_update_array_element_add_PRECISION gpu_update_array_element_add_float_complex
#define  gpu_hh_transform_PRECISION gpu_hh_transform_float_complex
#define  gpu_transpose_reduceadd_vectors_copy_block_PRECISION gpu_transpose_reduceadd_vectors_copy_block_float_complex
#define  mkl_offload_PRECISION_GEMM mkl_offload_CGEMM
#define  mkl_offload_PRECISION_GEMV mkl_offload_CGEMV
#define  mkl_offload_PRECISION_TRMM mkl_offload_CTRMM
#define  scal_PRECISION_GEMM PCGEMM
#define  scal_PRECISION_DOTC PCDOTC
#define  scal_PRECISION_LASET PCLASET
#define  scal_PRECISION_GEMR2D PCGEMR2D
#define  MPI_COMPLEX_PRECISION MPI_COMPLEX
#define  MPI_MATH_DATATYPE_PRECISION MPI_COMPLEX
#define  MPI_MATH_DATATYPE_PRECISION_C MPI_COMPLEX
#define  MPI_MATH_DATATYPE_PRECISION_EXPL MPI_COMPLEX8
#define  MPI_COMPLEX_EXPLICIT_PRECISION MPI_COMPLEX8
#define  MPI_REAL_PRECISION MPI_REAL4
#define  KIND_PRECISION rk4
#define  PRECISION_CMPLX CMPLX
#define  PRECISION_IMAG AIMAG
#define  PRECISION_REAL REAL
#define  C_DATATYPE_KIND c_float

#define ELPA_PRECISION_SSMV elpa_cssmv
#define ELPA_PRECISION_SSR2 elpa_cssr2

#define GPU_SCALE_QMAT_PRECISION gpu_scale_qmat_float_complex
#define GPU_COPY_REAL_PART_TO_Q_PRECISION gpu_copy_real_part_to_q_float_complex

#endif /* SINGLE PRECISION */


#endif /* COMPLEXCASE */
