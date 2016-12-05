#ifdef DOUBLE_PRECISION_COMPLEX
#undef  elpa_transpose_vectors_complex_PRECISION
#undef  elpa_reduce_add_vectors_complex_PRECISION
#undef  bandred_complex_PRECISION
#undef  trans_ev_band_to_full_complex_PRECISION
#undef  tridiag_band_complex_PRECISION
#undef  trans_ev_tridi_to_band_complex_PRECISION
#undef  band_band_complex_PRECISION
#undef  tridiag_complex_PRECISION
#undef  trans_ev_complex_PRECISION
#undef  solve_tridi_PRECISION
#undef  solve_tridi_col_PRECISION
#undef  solve_tridi_single_problem_PRECISION
#undef  qr_pdgeqrf_2dcomm_PRECISION
#undef  hh_transform_complex_PRECISION
#undef  symm_matrix_allreduce_PRECISION
#undef  redist_band_complex_PRECISION
#undef  unpack_row_complex_cpu_PRECISION
#undef  unpack_row_complex_cpu_openmp_PRECISION
#undef  unpack_and_prepare_row_group_complex_gpu_PRECISION
#undef  extract_hh_tau_complex_gpu_PRECISION
#undef  compute_hh_dot_products_complex_gpu_PRECISION
#undef  compute_hh_trafo_complex_cpu_openmp_PRECISION
#undef  compute_hh_trafo_complex_cpu_PRECISION
#undef  pack_row_group_complex_gpu_PRECISION
#undef  pack_row_complex_cpu_openmp_PRECISION
#undef  pack_row_complex_cpu_PRECISION
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
#undef  launch_compute_hh_trafo_c_kernel_complex_PRECISION
#undef  compute_hh_trafo_complex_gpu_PRECISION
#undef  launch_my_pack_c_kernel_complex_PRECISION
#undef  launch_my_unpack_c_kernel_complex_PRECISION
#undef  launch_compute_hh_dotp_c_kernel_complex_PRECISION
#undef  launch_extract_hh_tau_c_kernel_complex_PRECISION
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
#undef  MPI_COMPLEX_PRECISION
#undef  MPI_COMPLEX_EXPLICIT_PRECISION
#undef  MPI_REAL_PRECISION
#undef  KIND_PRECISION
#undef  PRECISION_CMPLX
#undef  PRECISION_IMAG
#undef  PRECISION_REAL
#undef  CONST_REAL_0_0
#undef  CONST_REAL_1_0
#undef  CONST_COMPLEX_0_0
#undef  size_of_PRECISION_complex
#define  elpa_transpose_vectors_complex_PRECISION elpa_transpose_vectors_complex_double
#define  elpa_reduce_add_vectors_complex_PRECISION elpa_reduce_add_vectors_complex_double
#define  bandred_complex_PRECISION bandred_complex_double
#define  trans_ev_band_to_full_complex_PRECISION trans_ev_band_to_full_complex_double
#define  tridiag_band_complex_PRECISION tridiag_band_complex_double
#define  trans_ev_tridi_to_band_complex_PRECISION trans_ev_tridi_to_band_complex_double
#define  band_band_complex_PRECISION band_band_complex_double
#define  tridiag_complex_PRECISION tridiag_complex_double
#define  trans_ev_complex_PRECISION trans_ev_complex_double
#define  solve_tridi_PRECISION solve_tridi_double
#define  solve_tridi_col_PRECISION solve_tridi_col_double
#define  solve_tridi_single_problem_PRECISION solve_tridi_single_problem_double
#define  qr_pdgeqrf_2dcomm_PRECISION qr_pdgeqrf_2dcomm_double
#define  hh_transform_complex_PRECISION hh_transform_complex_double
#define  symm_matrix_allreduce_PRECISION symm_matrix_allreduce_double
#define  redist_band_complex_PRECISION redist_band_complex_double
#define  unpack_row_complex_cpu_PRECISION unpack_row_complex_cpu_double
#define  unpack_row_complex_cpu_openmp_PRECISION unpack_row_complex_cpu_openmp_double
#define  unpack_and_prepare_row_group_complex_gpu_PRECISION unpack_and_prepare_row_group_complex_gpu_double
#define  extract_hh_tau_complex_gpu_PRECISION extract_hh_tau_complex_gpu_double
#define  compute_hh_dot_products_complex_gpu_PRECISION compute_hh_dot_products_complex_gpu_double
#define  compute_hh_trafo_complex_cpu_openmp_PRECISION compute_hh_trafo_complex_cpu_openmp_double
#define  compute_hh_trafo_complex_cpu_PRECISION compute_hh_trafo_complex_cpu_double
#define  pack_row_group_complex_gpu_PRECISION pack_row_group_complex_gpu_double
#define  pack_row_complex_cpu_openmp_PRECISION pack_row_complex_cpu_openmp_double
#define  pack_row_complex_cpu_PRECISION pack_row_complex_cpu_double
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
#define  launch_compute_hh_trafo_c_kernel_complex_PRECISION launch_compute_hh_trafo_c_kernel_complex_double
#define  compute_hh_trafo_complex_gpu_PRECISION compute_hh_trafo_complex_gpu_double
#define  launch_my_pack_c_kernel_complex_PRECISION launch_my_pack_c_kernel_complex_double
#define  launch_my_unpack_c_kernel_complex_PRECISION launch_my_unpack_c_kernel_complex_double
#define  launch_compute_hh_dotp_c_kernel_complex_PRECISION launch_compute_hh_dotp_c_kernel_complex_double
#define  launch_extract_hh_tau_c_kernel_complex_PRECISION launch_extract_hh_tau_c_kernel_complex_double
#define  PRECISION_GEMV ZGEMV
#define  PRECISION_TRMV ZTRMV
#define  PRECISION_GEMM ZGEMM
#define  PRECISION_TRMM ZTRMM
#define  PRECISION_HERK ZHERK
#define  PRECISION_SYRK ZSYRK
#define  PRECISION_SYMV ZSYMV
#define  PRECISION_SYMM ZSYMM
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
#define  cublas_PRECISION_GEMM cublas_ZGEMM
#define  cublas_PRECISION_TRMM cublas_ZTRMM
#define  cublas_PRECISION_GEMV cublas_ZGEMV
#define  PRECISION_SUFFIX "_double"
#define  MPI_COMPLEX_PRECISION MPI_DOUBLE_COMPLEX
#define  MPI_COMPLEX_EXPLICIT_PRECISION MPI_COMPLEX16
#define  MPI_REAL_PRECISION MPI_REAL8
#define  KIND_PRECISION rk8
#define  PRECISION_CMPLX DCMPLX
#define  PRECISION_IMAG DIMAG
#define  PRECISION_REAL DREAL
#define  CONST_REAL_0_0 0.0_rk8
#define  CONST_REAL_1_0 1.0_rk8
#define  CONST_COMPLEX_0_0 0.0_ck8
#define  size_of_PRECISION_complex size_of_double_complex_datatype
#else
#undef  elpa_transpose_vectors_complex_PRECISION
#undef  elpa_reduce_add_vectors_complex_PRECISION
#undef  bandred_complex_PRECISION
#undef  trans_ev_band_to_full_complex_PRECISION
#undef  tridiag_band_complex_PRECISION
#undef  trans_ev_tridi_to_band_complex_PRECISION
#undef  band_band_complex_PRECISION
#undef  tridiag_complex_PRECISION
#undef  trans_ev_complex_PRECISION
#undef  solve_tridi_PRECISION
#undef  solve_tridi_col_PRECISION
#undef  solve_tridi_single_problem_PRECISION
#undef  qr_pdgeqrf_2dcomm_PRECISION
#undef  hh_transform_complex_PRECISION
#undef  symm_matrix_allreduce_PRECISION
#undef  redist_band_complex_PRECISION
#undef  unpack_row_complex_cpu_PRECISION
#undef  unpack_row_complex_cpu_openmp_PRECISION
#undef  unpack_and_prepare_row_group_complex_gpu_PRECISION
#undef  extract_hh_tau_complex_gpu_PRECISION
#undef  compute_hh_dot_products_complex_gpu_PRECISION
#undef  compute_hh_trafo_complex_cpu_openmp_PRECISION
#undef  compute_hh_trafo_complex_cpu_PRECISION
#undef  pack_row_group_complex_gpu_PRECISION
#undef  pack_row_complex_cpu_openmp_PRECISION
#undef  pack_row_complex_cpu_PRECISION
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
#undef  launch_compute_hh_trafo_c_kernel_complex_PRECISION
#undef  compute_hh_trafo_complex_gpu_PRECISION
#undef  launch_my_pack_c_kernel_complex_PRECISION
#undef  launch_my_unpack_c_kernel_complex_PRECISION
#undef  launch_compute_hh_dotp_c_kernel_complex_PRECISION
#undef  launch_extract_hh_tau_c_kernel_complex_PRECISION
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
#undef  MPI_COMPLEX_PRECISION
#undef  MPI_COMPLEX_EXPLICIT_PRECISION
#undef  MPI_REAL_PRECISION
#undef  KIND_PRECISION
#undef  PRECISION_CMPLX
#undef  PRECISION_IMAG
#undef  PRECISION_REAL
#undef  CONST_REAL_0_0
#undef  CONST_REAL_1_0
#undef  CONST_COMPLEX_0_0
#undef  size_of_PRECISION_complex
#define  elpa_transpose_vectors_complex_PRECISION elpa_transpose_vectors_complex_single
#define  elpa_reduce_add_vectors_complex_PRECISION elpa_reduce_add_vectors_complex_single
#define  bandred_complex_PRECISION bandred_complex_single
#define  trans_ev_band_to_full_complex_PRECISION trans_ev_band_to_full_complex_single
#define  tridiag_band_complex_PRECISION tridiag_band_complex_single
#define  trans_ev_tridi_to_band_complex_PRECISION trans_ev_tridi_to_band_complex_single
#define  band_band_complex_PRECISION band_band_complex_single
#define  tridiag_complex_PRECISION tridiag_complex_single
#define  trans_ev_complex_PRECISION trans_ev_complex_single
#define  solve_tridi_PRECISION solve_tridi_single
#define  solve_tridi_col_PRECISION solve_tridi_col_single
#define  solve_tridi_single_problem_PRECISION solve_tridi_single_problem_single
#define  qr_pdgeqrf_2dcomm_PRECISION qr_pdgeqrf_2dcomm_single
#define  hh_transform_complex_PRECISION hh_transform_complex_single
#define  symm_matrix_allreduce_PRECISION symm_matrix_allreduce_single
#define  redist_band_complex_PRECISION redist_band_complex_single
#define  unpack_row_complex_cpu_PRECISION unpack_row_complex_cpu_single
#define  unpack_row_complex_cpu_openmp_PRECISION unpack_row_complex_cpu_openmp_single
#define  unpack_and_prepare_row_group_complex_gpu_PRECISION unpack_and_prepare_row_group_complex_gpu_single
#define  extract_hh_tau_complex_gpu_PRECISION extract_hh_tau_complex_gpu_single
#define  compute_hh_dot_products_complex_gpu_PRECISION compute_hh_dot_products_complex_gpu_single
#define  compute_hh_trafo_complex_cpu_openmp_PRECISION compute_hh_trafo_complex_cpu_openmp_single
#define  compute_hh_trafo_complex_cpu_PRECISION compute_hh_trafo_complex_cpu_single
#define  pack_row_group_complex_gpu_PRECISION pack_row_group_complex_gpu_single
#define  pack_row_complex_cpu_openmp_PRECISION pack_row_complex_cpu_openmp_single
#define  pack_row_complex_cpu_PRECISION pack_row_complex_cpu_single
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
#define  launch_compute_hh_trafo_c_kernel_complex_PRECISION launch_compute_hh_trafo_c_kernel_complex_single
#define  compute_hh_trafo_complex_gpu_PRECISION compute_hh_trafo_complex_gpu_single
#define  launch_my_pack_c_kernel_complex_PRECISION launch_my_pack_c_kernel_complex_single
#define  launch_my_unpack_c_kernel_complex_PRECISION launch_my_unpack_c_kernel_complex_single
#define  launch_compute_hh_dotp_c_kernel_complex_PRECISION launch_compute_hh_dotp_c_kernel_complex_single
#define  launch_extract_hh_tau_c_kernel_complex_PRECISION launch_extract_hh_tau_c_kernel_complex_single
#define  PRECISION_GEMV CGEMV
#define  PRECISION_TRMV CTRMV
#define  PRECISION_GEMM CGEMM
#define  PRECISION_TRMM CTRMM
#define  PRECISION_HERK CHERK
#define  PRECISION_SYRK CSYRK
#define  PRECISION_SYMV CSYMV
#define  PRECISION_SYMM CSYMM
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
#define  cublas_PRECISION_GEMM cublas_CGEMM
#define  cublas_PRECISION_TRMM cublas_CTRMM
#define  cublas_PRECISION_GEMV cublas_CGEMV
#define  PRECISION_SUFFIX "_single"
#define  MPI_COMPLEX_PRECISION MPI_COMPLEX
#define  MPI_COMPLEX_EXPLICIT_PRECISION MPI_COMPLEX8
#define  MPI_REAL_PRECISION MPI_REAL4
#define  KIND_PRECISION rk4
#define  PRECISION_CMPLX CMPLX
#define  PRECISION_IMAG AIMAG
#define  PRECISION_REAL REAL
#define  CONST_REAL_0_0 0.0_rk4
#define  CONST_REAL_1_0 1.0_rk4
#define  CONST_COMPLEX_0_0 0.0_ck4
#define  size_of_PRECISION_complex size_of_single_complex_datatype
#endif
