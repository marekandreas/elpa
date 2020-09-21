#!/usr/bin/env python3

simple_tokens = [
    "PRECISION",
    "elpa_transpose_vectors_NUMBER_PRECISION",
    "elpa_transpose_vectors_ssNUMBER_PRECISION",
    "elpa_reduce_add_vectors_NUMBER_PRECISION",
    "bandred_NUMBER_PRECISION",
    "trans_ev_band_to_full_NUMBER_PRECISION",
    "tridiag_band_NUMBER_PRECISION",
    "trans_ev_tridi_to_band_NUMBER_PRECISION",
    "band_band_NUMBER_PRECISION",
    "tridiag_NUMBER_PRECISION",
    "trans_ev_NUMBER_PRECISION",
    "solve_tridi_PRECISION",
    "solve_tridi_col_PRECISION",
    "solve_tridi_single_problem_PRECISION",
    "solve_evp_NUMBER_2stage_PRECISION",
    "qr_pdgeqrf_2dcomm_PRECISION",
    "hh_transform_NUMBER_PRECISION",
    "symm_matrix_allreduce_PRECISION",
    "ssymm_matrix_allreduce_PRECISION",
    "herm_matrix_allreduce_PRECISION",
    "redist_band_NUMBER_PRECISION",
    "unpack_row_NUMBER_cpu_PRECISION",
    "unpack_row_NUMBER_cpu_openmp_PRECISION",
    "unpack_and_prepare_row_group_NUMBER_gpu_PRECISION",
    "extract_hh_tau_NUMBER_gpu_PRECISION",
    "compute_hh_dot_products_NUMBER_gpu_PRECISION",
    "compute_hh_trafo_NUMBER_cpu_openmp_PRECISION",
    "compute_hh_trafo_NUMBER_cpu_PRECISION",
    "pack_row_group_NUMBER_gpu_PRECISION",
    "pack_row_NUMBER_cpu_openmp_PRECISION",
    "pack_row_NUMBER_cpu_PRECISION",
    "wy_gen_PRECISION",
    "wy_right_PRECISION",
    "wy_left_PRECISION",
    "wy_symm_PRECISION",
    "merge_recursive_PRECISION",
    "merge_systems_PRECISION",
    "distribute_global_column_PRECISION",
    "check_monotony_PRECISION",
    "global_gather_PRECISION",
    "resort_ev_PRECISION",
    "transform_columns_PRECISION",
    "solve_secular_equation_PRECISION",
    "global_product_PRECISION",
    "add_tmp_PRECISION",
    "v_add_s_PRECISION",
    "launch_compute_hh_trafo_c_kernel_NUMBER_PRECISION",
    "compute_hh_trafo_NUMBER_gpu_PRECISION", 
    "launch_my_pack_c_kernel_NUMBER_PRECISION",
    "launch_my_unpack_c_kernel_NUMBER_PRECISION",
    "launch_compute_hh_dotp_c_kernel_NUMBER_PRECISION",    
    "launch_extract_hh_tau_c_kernel_NUMBER_PRECISION",
    "AVAILABLE_UPCASENUMBER_ELPA_KERNELS",
    "UPCASENUMBER_ELPA_KERNEL_GENERIC",
    "DEFAULT_UPCASENUMBER_ELPA_KERNEL",
    "UPCASENUMBER_ELPA_KERNEL_NAMES",
    "UPCASENUMBER_ELPA_KERNEL_GPU",
]

blas_tokens = [
    "PRECISION_GEMV",
    "PRECISION_TRMV",
    "PRECISION_GEMM",
    "PRECISION_TRMM",
    "PRECISION_HERK",
    "PRECISION_SYRK",
    "PRECISION_SYMV",
    "PRECISION_SYMM",
    "PRECISION_HEMV",
    "PRECISION_HER2",
    "PRECISION_SYR2",
    "PRECISION_SYR2K",
    "PRECISION_GEQRF",
    "PRECISION_STEDC",
    "PRECISION_STEQR",
    "PRECISION_LAMRG",
    "PRECISION_LAMCH",
    "PRECISION_LAPY2",
    "PRECISION_LAED4",
    "PRECISION_LAED5",
    "cublas_PRECISION_GEMM",
    "cublas_PRECISION_TRMM",
    "cublas_PRECISION_GEMV",
]

explicit_tokens_complex = [
    ("PRECISION_SUFFIX", "\"_double\"", "\"_single\""),
    ("MPI_COMPLEX_PRECISION", "MPI_DOUBLE_COMPLEX", "MPI_COMPLEX"),
    ("MPI_COMPLEX_EXPLICIT_PRECISION", "MPI_COMPLEX16", "MPI_COMPLEX8"),
    ("MPI_REAL_PRECISION", "MPI_REAL8", "MPI_REAL4"),
    ("KIND_PRECISION", "rk8", "rk4"),
    ("PRECISION_CMPLX", "DCMPLX", "CMPLX"),
    ("PRECISION_IMAG", "DIMAG", "AIMAG"),
    ("PRECISION_REAL", "DREAL", "REAL"),
    ("size_of_PRECISION_complex", "size_of_double_complex_datatype", "size_of_single_complex_datatype"),
    ("C_DATATYPE_KIND", "c_double", "c_float"),
]

explicit_tokens_real = [
    ("PRECISION_SUFFIX", "\"_double\"", "\"_single\""),
    ("size_of_PRECISION_real",  "size_of_double_real_datatype",  "size_of_single_real_datatype"),
    ("MPI_REAL_PRECISION", "MPI_REAL8", "MPI_REAL4"),
    ("C_DATATYPE_KIND", "c_double", "c_float"),
]


explicit_order = {"single":2, "double":1}
blas_prefixes = {("real","single") : "S", ("real","double") : "D", ("complex","single") : "C", ("complex","double") : "Z"}

def print_variant(number, precision, explicit):
    for token in simple_tokens:
        print("#define ", token, token.replace("PRECISION", precision).replace("UPCASENUMBER", number.upper()).replace("NUMBER", number))
        print("#define ", token + "_STR", "'" + token.replace("PRECISION", precision).replace("UPCASENUMBER", number.upper()).replace("NUMBER", number) + "'")
        if("NUMBER" in token):
            print("#define ", token.replace("NUMBER", number), token.replace("PRECISION", precision).replace("NUMBER", number))
    for token in blas_tokens:
        print("#define ", token, token.replace("PRECISION_", blas_prefixes[(number, precision)]))
    for token in explicit:
        print("#define ", token[0], token[explicit_order[precision]])


def print_undefs(number, explicit):
    for token in simple_tokens:
        print("#undef ", token)
        print("#undef ", token + "_STR")
        if("NUMBER" in token):
            print("#undef ", token.replace("NUMBER", number))
    for token in blas_tokens:
        print("#undef ", token)
    for token in explicit:
        print("#undef ", token[0])


print("#ifdef REALCASE")
print("#undef  MATH_DATATYPE")
print("#define  MATH_DATATYPE real")
print_undefs("real", explicit_tokens_real)
# print_undefs("complex", explicit_tokens_complex)
print("#ifdef DOUBLE_PRECISION")
print_variant("real", "double", explicit_tokens_real)
print("#endif")
print("#ifdef SINGLE_PRECISION")
print_variant("real", "single", explicit_tokens_real)
print("#endif")
print("#endif")

print("#ifdef COMPLEXCASE")
print("#undef  MATH_DATATYPE")
print("#define  MATH_DATATYPE complex")
# print_undefs("real", explicit_tokens_real)
print_undefs("complex", explicit_tokens_complex)
print("#ifdef DOUBLE_PRECISION")
print_variant("complex", "double", explicit_tokens_complex)
print("#endif")
print("#ifdef SINGLE_PRECISION")
print_variant("complex", "single", explicit_tokens_complex)
print("#endif")
print("#endif")

# print("#elif MACROS_TYPE == COMPLEX_DOUBLE")
# print("#undef  NUMBER")
# print_undefs("complex", explicit_tokens_complex)
# print("#define  NUMBER complex")
# print_variant("complex", "double", explicit_tokens_complex)

# print("#elif MACROS_TYPE == COMPLEX_SINGLE")
# print("#undef  NUMBER")
# print_undefs("complex", explicit_tokens_complex)
# print("#define  NUMBER complex")
# print_variant("complex", "single", explicit_tokens_complex)
# print("#endif")
