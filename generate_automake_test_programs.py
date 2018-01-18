#!/usr/bin/env python
from __future__ import print_function
from itertools import product

language_flag = {
    "Fortran": "",
    "C": "_c_version",
}

domain_flag = {
    "real":    "-DTEST_REAL",
    "complex": "-DTEST_COMPLEX",
}
prec_flag = {
    "double": "-DTEST_DOUBLE",
    "single": "-DTEST_SINGLE",
}
solver_flag = {
    "1stage":         "-DTEST_SOLVER_1STAGE",
    "2stage":         "-DTEST_SOLVER_2STAGE",
    "scalapack_all":  "-DTEST_SCALAPACK_ALL",
    "scalapack_part": "-DTEST_SCALAPACK_PART",
}
gpu_flag = {
    0: "-DTEST_GPU=0",
    1: "-DTEST_GPU=1",
}

matrix_flag = {
    "random":   "-DTEST_MATRIX_RANDOM",
    "analytic": "-DTEST_MATRIX_ANALYTIC",
    "toeplitz": "-DTEST_MATRIX_TOEPLITZ",
    "frank":    "-DTEST_MATRIX_FRANK",
}

qr_flag = {
    0: "-DTEST_QR_DECOMPOSITION=0",
    1: "-DTEST_QR_DECOMPOSITION=1",
}

test_type_flag = {
    "eigenvectors":       "-DTEST_EIGENVECTORS",
    "eigenvalues":        "-DTEST_EIGENVALUES",
    "solve_tridiagonal":  "-DTEST_SOLVE_TRIDIAGONAL",
    "cholesky":           "-DTEST_CHOLESKY",
    "hermitian_multiply": "-DTEST_HERMITIAN_MULTIPLY",
}

layout_flag = {
    "all_layouts": "-DTEST_ALL_LAYOUTS",
    "square": ""
}

for lang, m, g, q, t, p, d, s, lay in product(sorted(language_flag.keys()),
                                              sorted(matrix_flag.keys()),
                                              sorted(gpu_flag.keys()),
                                              sorted(qr_flag.keys()),
                                              sorted(test_type_flag.keys()),
                                              sorted(prec_flag.keys()),
                                              sorted(domain_flag.keys()),
                                              sorted(solver_flag.keys()),
                                              sorted(layout_flag.keys())):

    if lang == "C" and (m == "analytic" or m == "toeplitz" or m == "frank" or lay == "all_layouts"):
        continue

    # exclude some test combinations

    # analytic tests only for "eigenvectors" and not on GPU
    if(m == "analytic" and (g == 1 or t != "eigenvectors")):
        continue

    # Frank tests only for "eigenvectors" and eigenvalues and real double precision case
    if(m == "frank" and ((t != "eigenvectors" or t != "eigenvalues") and (d != "real" or p != "double"))):
        continue

    if(s in ["scalapack_all", "scalapack_part"] and (g == 1 or t != "eigenvectors" or m != "analytic")):
        continue

    # do not test single-precision scalapack
    if(s in ["scalapack_all", "scalapack_part"] and ( p == "single")):
        continue

    # solve tridiagonal only for real toeplitz matrix in 1stage
    if (t == "solve_tridiagonal" and (s != "1stage" or d != "real" or m != "toeplitz")):
        continue

    # cholesky tests only 1stage and teoplitz matrix
    if (t == "cholesky" and (m != "toeplitz" or s == "2stage")):
        continue

    if (t == "eigenvalues" and (m == "random")):
        continue

    if (t == "hermitian_multiply" and (s == "2stage")):
        continue

    if (t == "hermitian_multiply" and (m == "toeplitz")):
        continue

    # qr only for 2stage real
    if (q == 1 and (s != "2stage" or d != "real" or t != "eigenvectors" or g == 1 or m != "random")):
        continue

    for kernel in ["all_kernels", "default_kernel"] if s == "2stage" else ["nokernel"]:
        endifs = 0
        extra_flags = []

        if (t == "eigenvalues" and kernel == "all_kernels"):
            continue

        if (lang == "C" and kernel == "all_kernels"):
            continue

        if (g == 1):
            print("if WITH_GPU_VERSION")
            endifs += 1

        if (lay == "all_layouts"):
            print("if WITH_MPI")
            endifs += 1

        if (s in ["scalapack_all", "scalapack_part"]):
            print("if WITH_SCALAPACK_TESTS")
            endifs += 1

        if kernel == "default_kernel":
            extra_flags.append("-DTEST_KERNEL=ELPA_2STAGE_{0}_DEFAULT".format(d.upper()))
        elif kernel == "all_kernels":
            extra_flags.append("-DTEST_ALL_KERNELS")

        if layout_flag[lay]:
            extra_flags.append(layout_flag[lay])

        if (p == "single"):
            if (d == "real"):
                print("if WANT_SINGLE_PRECISION_REAL")
            elif (d == "complex"):
                print("if WANT_SINGLE_PRECISION_COMPLEX")
            else:
                raise Exception("Oh no!")
            endifs += 1

        name = "test{langsuffix}_{d}_{p}_{t}_{s}{kernelsuffix}_{gpusuffix}{qrsuffix}{m}{layoutsuffix}".format(
            langsuffix=language_flag[lang],
            d=d, p=p, t=t, s=s,
            kernelsuffix="" if kernel == "nokernel" else "_" + kernel,
            gpusuffix="gpu_" if g else "",
            qrsuffix="qr_" if q else "",
            m=m,
            layoutsuffix="_all_layouts" if lay == "all_layouts" else "")

        print("if BUILD_KCOMPUTER")
        print("bin_PROGRAMS += " + name)
        print("else")
        print("noinst_PROGRAMS += " + name)
        print("endif")

        if lay == "square":
            print("check_SCRIPTS += " + name + "_default.sh")
        elif lay == "all_layouts":
            print("check_SCRIPTS += " + name + "_extended.sh")
        else:
            raise Exception("Unknown layout {0}".format(lay))

        if lang == "Fortran":
            print(name + "_SOURCES = test/Fortran/test.F90")
            print(name + "_LDADD = $(test_program_ldadd)")
            print(name + "_FCFLAGS = $(test_program_fcflags) \\")

        elif lang == "C":
            print(name + "_SOURCES = test/C/test.c")
            print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
            print(name + "_CFLAGS = $(test_program_cflags) \\")

        print("  -DTEST_CASE=\\\"{0}\\\" \\".format(name))
        print("  " + " \\\n  ".join([
            domain_flag[d],
            prec_flag[p],
            test_type_flag[t],
            solver_flag[s],
            gpu_flag[g],
            qr_flag[q],
            matrix_flag[m]] + extra_flags))

        print("endif\n" * endifs)

for lang, p, d in product(sorted(language_flag.keys()), sorted(prec_flag.keys()), sorted(domain_flag.keys())):
    endifs = 0
    if (p == "single"):
        if (d == "real"):
            print("if WANT_SINGLE_PRECISION_REAL")
        elif (d == "complex"):
            print("if WANT_SINGLE_PRECISION_COMPLEX")
        else:
            raise Exception("Oh no!")
        endifs += 1

    name = "test_autotune{langsuffix}_{d}_{p}".format(langsuffix=language_flag[lang], d=d, p=p)

    print("check_SCRIPTS += " + name + "_extended.sh")
    print("noinst_PROGRAMS += " + name)
    if lang == "Fortran":    
        print(name + "_SOURCES = test/Fortran/test_autotune.F90")
        print(name + "_LDADD = $(test_program_ldadd)")
        print(name + "_FCFLAGS = $(test_program_fcflags) \\")

    elif lang == "C":
        print(name + "_SOURCES = test/C/test_autotune.c")
        print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
        print(name + "_CFLAGS = $(test_program_cflags) \\")

    print("  " + " \\\n  ".join([
        domain_flag[d],
        prec_flag[p]]))
    print("endif\n" * endifs)
