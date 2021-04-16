#!/usr/bin/env python3
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
    "GPU_OFF": "-DTEST_NVIDIA_GPU=0 -DTEST_INTEL_GPU=0 -DTEST_AMD_GPU=0",
    "NVIDIA_GPU_ON": "-DTEST_NVIDIA_GPU=1",
    "INTEL_GPU_ON": "-DTEST_INTEL_GPU=1",
    "AMD_GPU_ON": "-DTEST_AMD_GPU=1",
}
gpu_id_flag = {
    0: "-DTEST_GPU_SET_ID=0",
    1: "-DTEST_GPU_SET_ID=1",
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
    "generalized":        "-DTEST_GENERALIZED_EIGENPROBLEM",
    "generalized_decomp": "-DTEST_GENERALIZED_DECOMP_EIGENPROBLEM",
}

layout_flag = {
    "all_layouts": "-DTEST_ALL_LAYOUTS",
    "square": ""
}

split_comm_flag = {
    "myself": "-DSPLIT_COMM_MYSELF",
    "by_elpa": ""
}

for lang, m, g, gid, q, t, p, d, s, lay, spl in product(sorted(language_flag.keys()),
                                                   sorted(matrix_flag.keys()),
                                                   sorted(gpu_flag.keys()),
                                                   sorted(gpu_id_flag.keys()),
                                                   sorted(qr_flag.keys()),
                                                   sorted(test_type_flag.keys()),
                                                   sorted(prec_flag.keys()),
                                                   sorted(domain_flag.keys()),
                                                   sorted(solver_flag.keys()),
                                                   sorted(layout_flag.keys()),
                                                   sorted(split_comm_flag.keys())):

    if gid == 1 and (g == 0 ):
        continue

    if lang == "C" and (m == "analytic" or m == "toeplitz" or m == "frank" or lay == "all_layouts"):
        continue

    # not implemented in the test.c file yet
    if lang == "C" and (t == "cholesky" or t == "hermitian_multiply" or q == 1):
        continue

    # exclude some test combinations

    # analytic tests only for "eigenvectors" and not on GPU
    if(m == "analytic" and ( g == "NVIDIA_GPU_ON" or g == "INTEL_GPU_ON" or g == "AMD_GPU_ON" or t != "eigenvectors")):
        continue

    # Frank tests only for "eigenvectors" and eigenvalues and real double precision case
    if(m == "frank" and ((t != "eigenvectors" or t != "eigenvalues") and (d != "real" or p != "double"))):
        continue

    if(s in ["scalapack_all", "scalapack_part"] and (g == "NVIDIA_GPU_ON" or g == "INTEL_GPU_ON" or g == "AMD_GPU_ON" or t != "eigenvectors" or m != "analytic")):
        continue

    # do not test single-precision scalapack
    if(s in ["scalapack_all", "scalapack_part"] and (p == "single")):
        continue

    # solve tridiagonal only for real toeplitz matrix in 1stage
    if (t == "solve_tridiagonal" and (s != "1stage" or d != "real" or m != "toeplitz")):
        continue

    # solve generalized only for random matrix in 1stage
    if (t == "generalized" and (m != "random" or s == "2stage")):
        continue

    # solve generalized already decomposed only for random matrix in 1stage
    # maybe this test should be further restricted, maybe not so important...
    if (t == "generalized_decomp" and (m != "random" or s == "2stage")):
        continue

    # cholesky tests only 1stage and teoplitz or random matrix
    if (t == "cholesky" and ((not (m == "toeplitz" or m == "random")) or s == "2stage")):
        continue

    if (t == "eigenvalues" and (m == "random")):
        continue

    if (t == "hermitian_multiply" and (s == "2stage")):
        continue

    if (t == "hermitian_multiply" and (m == "toeplitz")):
        continue

    # qr only for 2stage real
    if (q == 1 and (s != "2stage" or d != "real" or t != "eigenvectors" or g == "NVIDIA_GPU_ON" or "INTEL_GPU_ON"  or g == "AMD_GPU_ON" or m != "random")):
        continue

    if(spl == "myself" and (d != "real" or p != "double" or q != 0 or m != "random" or (t != "eigenvectors" and t != "cholesky")  or lang != "Fortran" or lay != "square")):
        continue

    for kernel in ["all_kernels", "default_kernel"] if s == "2stage" else ["nokernel"]:
        endifs = 0
        extra_flags = []

        if(spl == "myself" and kernel == "all_kernels"):
            continue

        if(spl == "myself"):
            print("if WITH_MPI")
            endifs += 1

        if (t == "eigenvalues" and kernel == "all_kernels"):
            continue

        if (lang == "C" and kernel == "all_kernels"):
            continue

        if (lang == "C"):
            print("if ENABLE_C_TESTS")
            endifs += 1

        if (g == "NVIDIA_GPU_ON"):
            print("if WITH_NVIDIA_GPU_VERSION")
            endifs += 1

        if (g == "INTEL_GPU_ON"):
            print("if WITH_INTEL_GPU_VERSION")
            endifs += 1

        if (g == "AMD_GPU_ON"):
            print("if WITH_AMD_GPU_VERSION")
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

        if split_comm_flag[spl]:
            extra_flags.append(split_comm_flag[spl])

        if (p == "single"):
            if (d == "real"):
                print("if WANT_SINGLE_PRECISION_REAL")
            elif (d == "complex"):
                print("if WANT_SINGLE_PRECISION_COMPLEX")
            else:
                raise Exception("Oh no!")
            endifs += 1

        name = "validate{langsuffix}_{d}_{p}_{t}_{s}{kernelsuffix}_{gpusuffix}{gpuidsuffix}{qrsuffix}{m}{layoutsuffix}{spl}".format(
            langsuffix=language_flag[lang],
            d=d, p=p, t=t, s=s,
            kernelsuffix="" if kernel == "nokernel" else "_" + kernel,
            gpusuffix="gpu_" if  (g == "NVIDIA_GPU_ON" or g == "INTEL_GPU_ON" or g == "AMD_GPU_ON") else "",
            gpuidsuffix="set_gpu_id_" if gid else "",
            qrsuffix="qr_" if q else "",
            m=m,
            layoutsuffix="_all_layouts" if lay == "all_layouts" else "",
            spl="_split_comm_myself" if spl == "myself" else "")

        print("if BUILD_KCOMPUTER")
        print("bin_PROGRAMS += " + name)
        print("else")
        print("noinst_PROGRAMS += " + name)
        print("endif")

        if lay == "square" or t == "generalized":
            if kernel == "all_kernels":
                print("check_SCRIPTS += " + name + "_extended.sh")
            else:
                print("check_SCRIPTS += " + name + "_default.sh")
        elif lay == "all_layouts":
            if kernel == "all_kernels":
                print("check_SCRIPTS += " + name + "_extended.sh")
            else:
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

        else:
            raise Exception("Unknown language")

        print("  -DTEST_CASE=\\\"{0}\\\" \\".format(name))
        print("  " + " \\\n  ".join([
            domain_flag[d],
            prec_flag[p],
            test_type_flag[t],
            solver_flag[s],
            gpu_flag[g],
            gpu_id_flag[gid],
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

    name = "validate_autotune{langsuffix}_{d}_{p}".format(langsuffix=language_flag[lang], d=d, p=p)

    print("if ENABLE_AUTOTUNING")
    if lang == "C":
        print("if ENABLE_C_TESTS")
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

    else:
        raise Exception("Unknown language")

    print("  " + " \\\n  ".join([
        domain_flag[d],
        prec_flag[p]]))
    print("endif\n" * endifs)
    if lang == "C":
        print("endif")
    print("endif")

name = "validate_multiple_objs_real_double"
print("if ENABLE_AUTOTUNING")
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/Fortran/test_multiple_objs.F90")
print(name + "_LDADD = $(test_program_ldadd)")
print(name + "_FCFLAGS = $(test_program_fcflags) \\")
print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['double']]))
print("endif")

name = "test_skewsymmetric_real_double"
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/Fortran/test_skewsymmetric.F90")
print(name + "_LDADD = $(test_program_ldadd)")
print(name + "_FCFLAGS = $(test_program_fcflags) \\")
print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['double']]))

name = "test_skewsymmetric_real_single"
print("if WANT_SINGLE_PRECISION_REAL")
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/Fortran/test_skewsymmetric.F90")
print(name + "_LDADD = $(test_program_ldadd)")
print(name + "_FCFLAGS = $(test_program_fcflags) \\")
print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['single']]))
print("endif")



name = "validate_multiple_objs_real_double_c_version"
print("if ENABLE_C_TESTS")
print("if ENABLE_AUTOTUNING")
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/C/test_multiple_objs.c")
print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
print(name + "_CFLAGS = $(test_program_cflags) \\")
print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['double']]))
print("endif")
print("endif")


name = "validate_split_comm_real_double"
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/Fortran/test_split_comm.F90")
print(name + "_LDADD = $(test_program_ldadd)")
print(name + "_FCFLAGS = $(test_program_fcflags) \\")
print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['double']]))
