#!/usr/bin/python
simple_tokens = ["tridiag_complex_PRECISION",
                 "trans_ev_complex_PRECISION",
                 "solve_complex_PRECISION",
                 "hh_transform_complex_PRECISION",
                 "elpa_transpose_vectors_complex_PRECISION",
                 "elpa_reduce_add_vectors_complex_PRECISION",
                 ]

blas_tokens = ["PRECISION_GEMV",
               "PRECISION_TRMV",
               "PRECISION_GEMM",
               "PRECISION_TRMM",
               "PRECISION_HERK",
               "cublas_PRECISION_gemm",
               "cublas_PRECISION_trmm",
               "cublas_PRECISION_gemv",
               ]

explicit_tokens = [("PRECISION_SUFFIX", "\"_double\"", "\"_single\""),
                   ("MPI_COMPLEX_PRECISION", "MPI_DOUBLE_COMPLEX", "MPI_COMPLEX"),
                   ("MPI_REAL_PRECISION", "MPI_REAL8", "MPI_REAL4"),
                   ("KIND_PRECISION", "rk8", "rk4"),
                   ("PRECISION_CMPLX", "DCMPLX", "CMPLX"),
                   ("PRECISION_IMAG", "DIMAG", "AIMAG"),
                   ("CONST_REAL_0_0", "0.0_rk8", "0.0_rk4"),
                   ("CONST_REAL_1_0", "1.0_rk8", "1.0_rk4"),
                   ("size_of_PRECISION_complex", "size_of_double_complex_datatype", "size_of_single_complex_datatype"),
                   ]

print "#ifdef DOUBLE_PRECISION_COMPLEX"

for token in simple_tokens:
    print "#define ", token, token.replace("PRECISION", "double")
for token in blas_tokens:
    print "#define ", token, token.replace("PRECISION_", "Z")    
for token in explicit_tokens:
    print "#define ", token[0], token[1]
    
print "#else"

for token in simple_tokens:
    print "#undef ", token
for token in blas_tokens:
    print "#undef ", token
for token in explicit_tokens:
    print "#undef ", token[0]

for token in simple_tokens:
    print "#define ", token, token.replace("PRECISION", "single")
for token in blas_tokens:
    print "#define ", token, token.replace("PRECISION_", "C")
for token in explicit_tokens:
    print "#define ", token[0], token[2]

print "#endif"
    
