#ifdef DOUBLE_PRECISION_COMPLEX
#define  tridiag_complex_PRECISION tridiag_complex_double
#define  trans_ev_complex_PRECISION trans_ev_complex_double
#define  solve_complex_PRECISION solve_complex_double
#define  hh_transform_complex_PRECISION hh_transform_complex_double
#define  elpa_transpose_vectors_complex_PRECISION elpa_transpose_vectors_complex_double
#define  elpa_reduce_add_vectors_complex_PRECISION elpa_reduce_add_vectors_complex_double
#define  PRECISION_GEMV ZGEMV
#define  PRECISION_TRMV ZTRMV
#define  PRECISION_GEMM ZGEMM
#define  PRECISION_TRMM ZTRMM
#define  PRECISION_HERK ZHERK
#define  PRECISION_SUFFIX "_double"
#define  MPI_COMPLEX_PRECISION MPI_DOUBLE_COMPLEX
#define  MPI_REAL_PRECISION MPI_REAL8
#define  KIND_PRECISION rk8
#define  PRECISION_CMPLX DCMPLX
#define  PRECISION_IMAG DIMAG
#define  CONST_REAL_0_0 0.0_rk8
#define  CONST_REAL_1_0 1.0_rk8
#else
#undef  tridiag_complex_PRECISION
#undef  trans_ev_complex_PRECISION
#undef  solve_complex_PRECISION
#undef  hh_transform_complex_PRECISION
#undef  elpa_transpose_vectors_complex_PRECISION
#undef  elpa_reduce_add_vectors_complex_PRECISION
#undef  PRECISION_GEMV
#undef  PRECISION_TRMV
#undef  PRECISION_GEMM
#undef  PRECISION_TRMM
#undef  PRECISION_HERK
#undef  PRECISION_SUFFIX
#undef  MPI_COMPLEX_PRECISION
#undef  MPI_REAL_PRECISION
#undef  KIND_PRECISION
#undef  PRECISION_CMPLX
#undef  PRECISION_IMAG
#undef  CONST_REAL_0_0
#undef  CONST_REAL_1_0
#define  tridiag_complex_PRECISION tridiag_complex_single
#define  trans_ev_complex_PRECISION trans_ev_complex_single
#define  solve_complex_PRECISION solve_complex_single
#define  hh_transform_complex_PRECISION hh_transform_complex_single
#define  elpa_transpose_vectors_complex_PRECISION elpa_transpose_vectors_complex_single
#define  elpa_reduce_add_vectors_complex_PRECISION elpa_reduce_add_vectors_complex_single
#define  PRECISION_GEMV CGEMV
#define  PRECISION_TRMV CTRMV
#define  PRECISION_GEMM CGEMM
#define  PRECISION_TRMM CTRMM
#define  PRECISION_HERK CHERK
#define  PRECISION_SUFFIX "_single"
#define  MPI_COMPLEX_PRECISION MPI_COMPLEX
#define  MPI_REAL_PRECISION MPI_REAL4
#define  KIND_PRECISION rk4
#define  PRECISION_CMPLX CMPLX
#define  PRECISION_IMAG AIMAG
#define  CONST_REAL_0_0 0.0_rk4
#define  CONST_REAL_1_0 1.0_rk4
#endif
