#undef PRECISION
#undef MATH_DATATYPE
#define MATH_DATATYPE real

#ifdef DOUBLE_PRECISION_REAL
#define PRECISION double
#else
#define PRECISION single
#endif

#ifdef DOUBLE_PRECISION_REAL
#undef  elpa_transpose_vectors_real_PRECISION
#undef  elpa_reduce_add_vectors_real_PRECISION
#undef  bandred_real_PRECISION
#undef  trans_ev_band_to_full_real_PRECISION
#undef  tridiag_band_real_PRECISION
#undef  trans_ev_tridi_to_band_real_PRECISION
#undef  band_band_real_PRECISION
#undef  tridiag_real_PRECISION
#undef  solve_tridi_PRECISION
#undef  solve_tridi_col_PRECISION
#undef  solve_tridi_single_problem_PRECISION
#undef  qr_pdgeqrf_2dcomm_PRECISION
#undef  hh_transform_real_PRECISION
#undef  symm_matrix_allreduce_PRECISION
#undef  redist_band_real_PRECISION
#undef  unpack_row_real_cpu_PRECISION
#undef  unpack_row_real_cpu_openmp_PRECISION
#undef  unpack_and_prepare_row_group_real_gpu_PRECISION
#undef  extract_hh_tau_real_gpu_PRECISION
#undef  compute_hh_dot_products_real_gpu_PRECISION
#undef  compute_hh_trafo_real_cpu_openmp_PRECISION
#undef  compute_hh_trafo_real_cpu_PRECISION
#undef  pack_row_group_real_gpu_PRECISION
#undef  pack_row_real_cpu_openmp_PRECISION
#undef  pack_row_real_cpu_PRECISION
#undef  wy_gen_PRECISION
#undef  wy_right_PRECISION
#undef  wy_left_PRECISION
#undef  wy_symm_PRECISION
#undef  merge_recursive_PRECISION
#undef  merge_systems_PRECISION
#undef  distribute_global_column_PRECISION
#undef  check_monotony_PRECISION
#undef  global_gather_PRECISION
#undef  resort_ev_PRECISION
#undef  transform_columns_PRECISION
#undef  solve_secular_equation_PRECISION
#undef  global_product_PRECISION
#undef  add_tmp_PRECISION
#undef  v_add_s_PRECISION
#undef  PRECISION_GEMV
#undef  PRECISION_TRMV
#undef  PRECISION_GEMM
#undef  PRECISION_TRMM
#undef  PRECISION_HERK
#undef  PRECISION_SYRK
#undef  PRECISION_SYMV
#undef  PRECISION_SYMM
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
#undef  cublas_PRECISION_GEMM
#undef  cublas_PRECISION_TRMM
#undef  cublas_PRECISION_GEMV
#undef  PRECISION_SUFFIX
#undef  CONST_0_0
#undef  CONST_0_5
#undef  CONST_1_0
#undef  CONST_2_0
#undef  CONST_8_0
#undef  size_of_PRECISION_real
#undef  MPI_REAL_PRECISION
#define  elpa_transpose_vectors_real_PRECISION elpa_transpose_vectors_real_double
#define  elpa_reduce_add_vectors_real_PRECISION elpa_reduce_add_vectors_real_double
#define  bandred_real_PRECISION bandred_real_double
#define  trans_ev_band_to_full_real_PRECISION trans_ev_band_to_full_real_double
#define  tridiag_band_real_PRECISION tridiag_band_real_double
#define  trans_ev_tridi_to_band_real_PRECISION trans_ev_tridi_to_band_real_double
#define  band_band_real_PRECISION band_band_real_double
#define  tridiag_real_PRECISION tridiag_real_double
#define  solve_tridi_PRECISION solve_tridi_double
#define  solve_tridi_col_PRECISION solve_tridi_col_double
#define  solve_tridi_single_problem_PRECISION solve_tridi_single_problem_double
#define  qr_pdgeqrf_2dcomm_PRECISION qr_pdgeqrf_2dcomm_double
#define  hh_transform_real_PRECISION hh_transform_real_double
#define  symm_matrix_allreduce_PRECISION symm_matrix_allreduce_double
#define  redist_band_real_PRECISION redist_band_real_double
#define  unpack_row_real_cpu_PRECISION unpack_row_real_cpu_double
#define  unpack_row_real_cpu_openmp_PRECISION unpack_row_real_cpu_openmp_double
#define  unpack_and_prepare_row_group_real_gpu_PRECISION unpack_and_prepare_row_group_real_gpu_double
#define  extract_hh_tau_real_gpu_PRECISION extract_hh_tau_real_gpu_double
#define  compute_hh_dot_products_real_gpu_PRECISION compute_hh_dot_products_real_gpu_double
#define  compute_hh_trafo_real_cpu_openmp_PRECISION compute_hh_trafo_real_cpu_openmp_double
#define  compute_hh_trafo_real_cpu_PRECISION compute_hh_trafo_real_cpu_double
#define  pack_row_group_real_gpu_PRECISION pack_row_group_real_gpu_double
#define  pack_row_real_cpu_openmp_PRECISION pack_row_real_cpu_openmp_double
#define  pack_row_real_cpu_PRECISION pack_row_real_cpu_double
#define  wy_gen_PRECISION wy_gen_double
#define  wy_right_PRECISION wy_right_double
#define  wy_left_PRECISION wy_left_double
#define  wy_symm_PRECISION wy_symm_double
#define  merge_recursive_PRECISION merge_recursive_double
#define  merge_systems_PRECISION merge_systems_double
#define  distribute_global_column_PRECISION distribute_global_column_double
#define  check_monotony_PRECISION check_monotony_double
#define  global_gather_PRECISION global_gather_double
#define  resort_ev_PRECISION resort_ev_double
#define  transform_columns_PRECISION transform_columns_double
#define  solve_secular_equation_PRECISION solve_secular_equation_double
#define  global_product_PRECISION global_product_double
#define  add_tmp_PRECISION add_tmp_double
#define  v_add_s_PRECISION v_add_s_double
#define  PRECISION_GEMV DGEMV
#define  PRECISION_TRMV DTRMV
#define  PRECISION_GEMM DGEMM
#define  PRECISION_TRMM DTRMM
#define  PRECISION_HERK DHERK
#define  PRECISION_SYRK DSYRK
#define  PRECISION_SYMV DSYMV
#define  PRECISION_SYMM DSYMM
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
#define  cublas_PRECISION_GEMM cublas_DGEMM
#define  cublas_PRECISION_TRMM cublas_DTRMM
#define  cublas_PRECISION_GEMV cublas_DGEMV
#define  PRECISION_SUFFIX "_double"
#define  CONST_0_0 0.0_rk8
#define  CONST_0_5 0.5_rk8
#define  CONST_1_0 1.0_rk8
#define  CONST_2_0 2.0_rk8
#define  CONST_8_0 8.0_rk8
#define  size_of_PRECISION_real size_of_double_real_datatype
#define  MPI_REAL_PRECISION MPI_REAL8
#else

#undef  elpa_transpose_vectors_real_PRECISION
#undef  elpa_reduce_add_vectors_real_PRECISION
#undef  bandred_real_PRECISION
#undef  trans_ev_band_to_full_real_PRECISION
#undef  tridiag_band_real_PRECISION
#undef  trans_ev_tridi_to_band_real_PRECISION
#undef  band_band_real_PRECISION
#undef  tridiag_real_PRECISION
#undef  solve_tridi_PRECISION
#undef  solve_tridi_col_PRECISION
#undef  solve_tridi_single_problem_PRECISION
#undef  qr_pdgeqrf_2dcomm_PRECISION
#undef  hh_transform_real_PRECISION
#undef  symm_matrix_allreduce_PRECISION
#undef  redist_band_real_PRECISION
#undef  unpack_row_real_cpu_PRECISION
#undef  unpack_row_real_cpu_openmp_PRECISION
#undef  unpack_and_prepare_row_group_real_gpu_PRECISION
#undef  extract_hh_tau_real_gpu_PRECISION
#undef  compute_hh_dot_products_real_gpu_PRECISION
#undef  compute_hh_trafo_real_cpu_openmp_PRECISION
#undef  compute_hh_trafo_real_cpu_PRECISION
#undef  pack_row_group_real_gpu_PRECISION
#undef  pack_row_real_cpu_openmp_PRECISION
#undef  pack_row_real_cpu_PRECISION
#undef  wy_gen_PRECISION
#undef  wy_right_PRECISION
#undef  wy_left_PRECISION
#undef  wy_symm_PRECISION
#undef  merge_recursive_PRECISION
#undef  merge_systems_PRECISION
#undef  distribute_global_column_PRECISION
#undef  check_monotony_PRECISION
#undef  global_gather_PRECISION
#undef  resort_ev_PRECISION
#undef  transform_columns_PRECISION
#undef  solve_secular_equation_PRECISION
#undef  global_product_PRECISION
#undef  add_tmp_PRECISION
#undef  v_add_s_PRECISION
#undef  PRECISION_GEMV
#undef  PRECISION_TRMV
#undef  PRECISION_GEMM
#undef  PRECISION_TRMM
#undef  PRECISION_HERK
#undef  PRECISION_SYRK
#undef  PRECISION_SYMV
#undef  PRECISION_SYMM
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
#undef  cublas_PRECISION_GEMM
#undef  cublas_PRECISION_TRMM
#undef  cublas_PRECISION_GEMV
#undef  PRECISION_SUFFIX
#undef  CONST_0_0
#undef  CONST_0_5
#undef  CONST_1_0
#undef  CONST_2_0
#undef  CONST_8_0
#undef  size_of_PRECISION_real
#undef  MPI_REAL_PRECISION
#define  elpa_transpose_vectors_real_PRECISION elpa_transpose_vectors_real_single
#define  elpa_reduce_add_vectors_real_PRECISION elpa_reduce_add_vectors_real_single
#define  bandred_real_PRECISION bandred_real_single
#define  trans_ev_band_to_full_real_PRECISION trans_ev_band_to_full_real_single
#define  tridiag_band_real_PRECISION tridiag_band_real_single
#define  trans_ev_tridi_to_band_real_PRECISION trans_ev_tridi_to_band_real_single
#define  band_band_real_PRECISION band_band_real_single
#define  tridiag_real_PRECISION tridiag_real_single
#define  solve_tridi_PRECISION solve_tridi_single
#define  solve_tridi_col_PRECISION solve_tridi_col_single
#define  solve_tridi_single_problem_PRECISION solve_tridi_single_problem_single
#define  qr_pdgeqrf_2dcomm_PRECISION qr_pdgeqrf_2dcomm_single
#define  hh_transform_real_PRECISION hh_transform_real_single
#define  symm_matrix_allreduce_PRECISION symm_matrix_allreduce_single
#define  redist_band_real_PRECISION redist_band_real_single
#define  unpack_row_real_cpu_PRECISION unpack_row_real_cpu_single
#define  unpack_row_real_cpu_openmp_PRECISION unpack_row_real_cpu_openmp_single
#define  unpack_and_prepare_row_group_real_gpu_PRECISION unpack_and_prepare_row_group_real_gpu_single
#define  extract_hh_tau_real_gpu_PRECISION extract_hh_tau_real_gpu_single
#define  compute_hh_dot_products_real_gpu_PRECISION compute_hh_dot_products_real_gpu_single
#define  compute_hh_trafo_real_cpu_openmp_PRECISION compute_hh_trafo_real_cpu_openmp_single
#define  compute_hh_trafo_real_cpu_PRECISION compute_hh_trafo_real_cpu_single
#define  pack_row_group_real_gpu_PRECISION pack_row_group_real_gpu_single
#define  pack_row_real_cpu_openmp_PRECISION pack_row_real_cpu_openmp_single
#define  pack_row_real_cpu_PRECISION pack_row_real_cpu_single
#define  wy_gen_PRECISION wy_gen_single
#define  wy_right_PRECISION wy_right_single
#define  wy_left_PRECISION wy_left_single
#define  wy_symm_PRECISION wy_symm_single
#define  merge_recursive_PRECISION merge_recursive_single
#define  merge_systems_PRECISION merge_systems_single
#define  distribute_global_column_PRECISION distribute_global_column_single
#define  check_monotony_PRECISION check_monotony_single
#define  global_gather_PRECISION global_gather_single
#define  resort_ev_PRECISION resort_ev_single
#define  transform_columns_PRECISION transform_columns_single
#define  solve_secular_equation_PRECISION solve_secular_equation_single
#define  global_product_PRECISION global_product_single
#define  add_tmp_PRECISION add_tmp_single
#define  v_add_s_PRECISION v_add_s_single
#define  PRECISION_GEMV SGEMV
#define  PRECISION_TRMV STRMV
#define  PRECISION_GEMM SGEMM
#define  PRECISION_TRMM STRMM
#define  PRECISION_HERK SHERK
#define  PRECISION_SYRK SSYRK
#define  PRECISION_SYMV SSYMV
#define  PRECISION_SYMM SSYMM
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
#define  cublas_PRECISION_GEMM cublas_SGEMM
#define  cublas_PRECISION_TRMM cublas_STRMM
#define  cublas_PRECISION_GEMV cublas_SGEMV
#define  PRECISION_SUFFIX "_single"
#define  CONST_0_0 0.0_rk4
#define  CONST_0_5 0.5_rk4
#define  CONST_1_0 1.0_rk4
#define  CONST_2_0 2.0_rk4
#define  CONST_8_0 8.0_rk4
#define  size_of_PRECISION_real size_of_single_real_datatype
#define  MPI_REAL_PRECISION MPI_REAL4
#endif
