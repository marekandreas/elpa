#!/usr/bin/env python3
from itertools import product

language_flag = {
    "Fortran": "",
    "C": "_c_version",
    "C++": "_cpp_version",
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
    "GPU_OFF": "-DTEST_NVIDIA_GPU=0 -DTEST_INTEL_GPU=0 -DTEST_AMD_GPU=0 -DTEST_OPENMP_OFFLOAD_GPU=0 -DTEST_INTEL_GPU_OPENMP=0 -DTEST_INTEL_GPU_SYCL=0",
    "NVIDIA_GPU_ON": "-DTEST_NVIDIA_GPU=1",
    "AMD_GPU_ON": "-DTEST_AMD_GPU=1",
    "OPENMP_OFFLOAD_GPU_ON": "-DTEST_INTEL_GPU_OPENMP=1",
    "SYCL_GPU_ON": "-DTEST_INTEL_GPU_SYCL=1",
}
#"INTEL_GPU_ON": "-DTEST_INTEL_GPU=1"
gpu_id_flag = {
    0: "-DTEST_GPU_SET_ID=0",
    1: "-DTEST_GPU_SET_ID=1",
}

device_pointer_flag = {
    0: "-DTEST_GPU_DEVICE_POINTER_API=0",
    1: "-DTEST_GPU_DEVICE_POINTER_API=1",
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
    "hermitian_multiply_full": "-DTEST_HERMITIAN_MULTIPLY_FULL",
    "hermitian_multiply_upper": "-DTEST_HERMITIAN_MULTIPLY_UPPER",
    "hermitian_multiply_lower": "-DTEST_HERMITIAN_MULTIPLY_LOWER",
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

explicit_name_flag = {
        "explicit": "-DTEST_EXPLICIT_NAME",
        "implicit": ""
}

for lang, m, g, gid, deviceptr, q, t, p, d, s, lay, spl, api_name in product(sorted(language_flag.keys()),
                                                   sorted(matrix_flag.keys()),
                                                   sorted(gpu_flag.keys()),
                                                   sorted(gpu_id_flag.keys()),
                                                   sorted(device_pointer_flag.keys()),
                                                   sorted(qr_flag.keys()),
                                                   sorted(test_type_flag.keys()),
                                                   sorted(prec_flag.keys()),
                                                   sorted(domain_flag.keys()),
                                                   sorted(solver_flag.keys()),
                                                   sorted(layout_flag.keys()),
                                                   sorted(split_comm_flag.keys()),
                                                   sorted(explicit_name_flag.keys())):
    
    # for debugging the loop
    #if (lang=="Fortran" and m=="toeplitz" and g=="NVIDIA_GPU_ON" and gid==1 and q==0 and t=="eigenvalues" and p=="double" and d=="real" and s=="1stage" and lay=="square" and spl=="by_elpa" and api_name=="explicit"):
    #     print("Here")
    
    # begin: exclude some test combinations
         
    if gid == 1 and (g == "GPU_OFF" ):
        continue

    if deviceptr == 1 and (gid == 0 ):
        continue

    if deviceptr == 1 and (api_name != "explicit"):
        continue

    if lay == "all_layouts" and (api_name == "explicit"):
        continue

    if api_name == "explicit" and not(m  == "random" 
                                      or (lang=="Fortran" and t=="eigenvalues" and m=="toeplitz" and s=="1stage" and lay=="square")
                                      or (lang=="C" and t=="eigenvalues" and m=="analytic" and s=="1stage" and gid==deviceptr)):
        continue

    if gid == 1 and not(m  == "random" 
                        or (lang=="Fortran" and t=="eigenvalues" and m=="toeplitz" and s=="1stage" and lay=="square")
                        or (lang!="Fortran" and t=="eigenvalues" and m=="analytic" and s=="1stage" and gid==deviceptr)):
         continue

    if deviceptr == 1 and not(m  == "random" 
                              or (lang=="Fortran" and t=="eigenvalues" and m=="toeplitz" and s=="1stage" and lay=="square")
                              or (lang!="Fortran" and t=="eigenvalues" and m=="analytic" and s=="1stage" and gid==deviceptr)):
      continue
	
    # C/C++-tests only for "random", "analytic" or "toeplitz" matrix and "square" layout
    if lang!="Fortran" and (m == "frank" or lay == "all_layouts"):
        continue

    if lang!="Fortran" and (api_name == "explicit") and (gid != deviceptr) and (t != "eigenvectors") and (t != "eigenvalues") and (t != "cholesky") and (t != "hermitian_multiply_full" and t != "hermitian_multiply_upper" and t != "hermitian_multiply_lower"):
        continue
       
    if api_name == "explicit" and ((t != "eigenvectors") and  (t != "eigenvalues") and (t != "cholesky") and (t != "hermitian_multiply_full" and t != "hermitian_multiply_upper" and t != "hermitian_multiply_lower")):
        continue

    if lang !="Fortran" and (t == "hermitian_multiply_upper" or t == "hermitian_multiply_lower"):
        continue

    # not implemented in the test.c file yet
    if lang!="Fortran" and q == 1:
        continue

    # analytic tests only for "eigenvectors" (Fortan) and "eigenvalues" (C,C++)
    #if(m == "analytic" and ( g == "NVIDIA_GPU_ON" or g == "INTEL_GPU_ON" or g == "AMD_GPU_ON" or g == "OPENMP_OFFLOAD_GPU_ON" or g == "SYCL_GPU_ON" or t != "eigenvectors")):
    if(lang == "Fortran" and m == "analytic" and t != "eigenvectors"):
        continue
    if(lang != "Fortran" and m == "analytic" and t != "eigenvalues"):
        continue
       
    # Frank tests only for "eigenvectors" and eigenvalues and real double precision case
    if(m == "frank" and ((t != "eigenvectors" or t != "eigenvalues") and (d != "real" or p != "double"))):
        continue

    if(s in ["scalapack_all", "scalapack_part"] and (g == "NVIDIA_GPU_ON" or g == "INTEL_GPU_ON" or g == "AMD_GPU_ON" or g == "OPENMP_OFFLOAD_GPU_ON" or g == "SYCL_GPU_ON" or t != "eigenvectors" or m != "analytic")):
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
    
    # "eigenvalues" in C/C++ are tested only for analytic matrix
    if (lang != "Fortran" and t == "eigenvalues" and m != "analytic"):
        continue
    
    # "solve_tridiagonal" in C/C++ are tested only for toeplitz matrix
    # validate_c_version_real_[double/single]_solve_tridiagonal_1stage_toeplitz_default
    # validate_c_version_real_[double/single]_solve_tridiagonal_1stage_gpu_toeplitz_default
    if (lang != "Fortran" and ((t=="solve_tridiagonal" and m!="toeplitz") or (t!="solve_tridiagonal" and m=="toeplitz"))): 
        continue
        
    if ((t == "hermitian_multiply_full" or t == "hermitian_multiply_upper" or t == "hermitian_multiply_lower") and (s == "2stage")):
        continue

    if ((t == "hermitian_multiply_full" or t == "hermitian_multiply_upper" or t == "hermitian_multiply_lower") and (m == "toeplitz")):
        continue

    # qr only for 2stage real
    if (q == 1 and (s != "2stage" or d != "real" or t != "eigenvectors" or g == "NVIDIA_GPU_ON" or "INTEL_GPU_ON" or g == "OPENMP_OFFLOAD_GPU_ON" or g == "SYCL_GPU_ON" or g == "AMD_GPU_ON" or m != "random")):
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

        if (lang != "Fortran" and kernel == "all_kernels"):
            continue
        
        # end: exclude some test combinations
        #if (lang == "Fortran" and ((t != "eigenvectors") and (m !="random") and (lay !="square"))):
        if (lang == "Fortran"): 
            print("if ENABLE_FORTRAN_TESTS")
            endifs += 1
      
        if (lang == "C"):
            print("if ENABLE_C_TESTS")
            endifs += 1

        if (lang == "C++"):
            print("if ENABLE_CPP_TESTS")
            endifs += 1

        if (g == "NVIDIA_GPU_ON"):
            print("if WITH_NVIDIA_GPU_VERSION")
            endifs += 1

        #if (g == "INTEL_GPU_ON"):
        #    print("if WITH_INTEL_GPU_VERSION")
        #    endifs += 1

        if (g == "OPENMP_OFFLOAD_GPU_ON"):
            print("if WITH_OPENMP_OFFLOAD_GPU_VERSION")
            endifs += 1

        if (g == "SYCL_GPU_ON"):
            print("if WITH_SYCL_GPU_VERSION")
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


        if (g == "NVIDIA_GPU_ON" or g == "INTEL_GPU_ON" or g == "AMD_GPU_ON" or g == "OPENMP_OFFLOAD_GPU_ON" or g == "SYCL_GPU_ON"):
          combined_suffix="gpu_"
          if (gid):
            combined_suffix="gpu_id_"
            if (deviceptr):
              combined_suffix="gpu_api_"
        else:
          combined_suffix=""

        if (s == "1stage" or s == "2stage" or s == "scalapack_all" or s == "scalapack_part"):
          solver = s
        else:
          solver =""

        name = "validate{langsuffix}_{d}_{p}_{t}_{solver}{kernelsuffix}_{appended_suffix}{qrsuffix}{m}{layoutsuffix}{spl}{api_name}".format(
            langsuffix=language_flag[lang],
            d=d, p=p, t=t, solver=solver,
            kernelsuffix="" if kernel == "nokernel" else "_" + kernel,
            appended_suffix=combined_suffix,
            qrsuffix="qr_" if q else "",
            m=m,
            layoutsuffix="_all_layouts" if lay == "all_layouts" else "",
            spl="_split_comm_myself" if spl == "myself" else "", 
            api_name="_explicit" if api_name == "explicit" else "")


        if (g == "GPU_OFF"):
          print("if BUILD_CPU_TESTS")
        else:
          print("if BUILD_GPU_TESTS")


        if (m == "analytic"):
          print("if BUILD_FUGAKU")
          print("else")
          print("if BUILD_KCOMPUTER")
          print("bin_PROGRAMS += " + name)
          print("else")
          print("noinst_PROGRAMS += " + name)
          print("endif")
          print("endif")
        else:
          print("if BUILD_KCOMPUTER")
          print("bin_PROGRAMS += " + name)
          print("else")
          print("noinst_PROGRAMS += " + name)
          print("endif")

        if (m == "analytic"):
          print("if BUILD_FUGAKU")
          print("else")
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
          
          elif lang == "C++":
              print(name + "_SOURCES = test/C++/test.cpp")
              print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
              print(name + "_CXXFLAGS = $(test_program_cxxflags) \\")
            
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
              device_pointer_flag[deviceptr],
              qr_flag[q],
              explicit_name_flag[api_name],
              matrix_flag[m]] + extra_flags))

          print("endif\n" * endifs)
          print("")
          print("endif")
          print("")

        else:
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

          elif lang == "C++":
              print(name + "_SOURCES = test/C++/test.cpp")
              print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
              print(name + "_CXXFLAGS = $(test_program_cxxflags) \\")
            
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
              device_pointer_flag[deviceptr],
              qr_flag[q],
              explicit_name_flag[api_name],
              matrix_flag[m]] + extra_flags))

          print("endif\n" * endifs)
          print("")
        #CPU / GPU tests
        print("endif")

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

    print("if BUILD_CPU_TESTS")
    print("if ENABLE_AUTOTUNING")
    if lang == "C":
        print("if ENABLE_C_TESTS")
    if lang == "C++":
        print("if ENABLE_CPP_TESTS")
         
    print("check_SCRIPTS += " + name + "_autotune.sh")
    print("noinst_PROGRAMS += " + name)
    if lang == "Fortran":
        print(name + "_SOURCES = test/Fortran/test_autotune.F90")
        print(name + "_LDADD = $(test_program_ldadd)")
        print(name + "_FCFLAGS = $(test_program_fcflags) \\")

    elif lang == "C":
        print(name + "_SOURCES = test/C/test_autotune.c")
        print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
        print(name + "_CFLAGS = $(test_program_cflags) \\")

    elif lang == "C++":
        print(name + "_SOURCES = test/C++/test_autotune.cpp")
        print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
        print(name + "_CXXFLAGS = $(test_program_cxxflags) \\")
      
    else:
        raise Exception("Unknown language")

    print("  " + " \\\n  ".join([
        domain_flag[d],
        prec_flag[p]]))
    print("endif\n" * endifs)
    if lang == "C":
        print("endif")
    if lang == "C++":
        print("endif")
    print("endif")
    print("endif")




#invert_triangular with GPU
language_flag = {
    "Fortran": "",
}

gpu_flag = {
    "GPU_OFF": "-DTEST_NVIDIA_GPU=0 -DTEST_INTEL_GPU=0 -DTEST_AMD_GPU=0 -DTEST_OPENMP_OFFLOAD_GPU=0 -DTEST_INTEL_GPU_OPENMP=0 -DTEST_INTEL_GPU_SYCL=0",
    "NVIDIA_GPU_ON": "-DTEST_NVIDIA_GPU=1",
    "AMD_GPU_ON": "-DTEST_AMD_GPU=1",
    "OPENMP_OFFLOAD_GPU_ON": "-DTEST_INTEL_GPU_OPENMP=1",
    "SYCL_GPU_ON": "-DTEST_INTEL_GPU_SYCL=1",
}
#"INTEL_GPU_ON": "-DTEST_INTEL_GPU=1"
gpu_id_flag = {
    0: "-DTEST_GPU_SET_ID=0",
    1: "-DTEST_GPU_SET_ID=1",
}

device_pointer_flag = {
    0: "-DTEST_GPU_DEVICE_POINTER_API=0",
    1: "-DTEST_GPU_DEVICE_POINTER_API=1",
}

explicit_name_flag = {
        "explicit": "-DTEST_EXPLICIT_NAME",
        "implicit": ""
}
for lang, g, gid, deviceptr, p, d, api_name in product(sorted(language_flag.keys()),
                                                   sorted(gpu_flag.keys()),
                                                   sorted(gpu_id_flag.keys()),
                                                   sorted(device_pointer_flag.keys()),
                                                   sorted(prec_flag.keys()),
                                                   sorted(domain_flag.keys()),
                                                   sorted(explicit_name_flag.keys())):

    endifs = 0
    
         
    if gid == 1 and (g == "GPU_OFF" ):
        continue

    if deviceptr == 1 and (gid == 0 ):
        continue

    if deviceptr == 1 and (api_name != "explicit"):
        continue

    # conditional cases
    
    if (g == "NVIDIA_GPU_ON"):
        print("if WITH_NVIDIA_GPU_VERSION")
        endifs += 1

    #if (g == "INTEL_GPU_ON"):
    #    print("if WITH_INTEL_GPU_VERSION")
    #    endifs += 1

    print("if BUILD_GPU_TESTS")
    print("if ENABLE_FORTRAN_TESTS")
    endifs += 1
    if (g == "OPENMP_OFFLOAD_GPU_ON"):
        print("if WITH_OPENMP_OFFLOAD_GPU_VERSION")
        endifs += 1

    if (g == "SYCL_GPU_ON"):
        print("if WITH_SYCL_GPU_VERSION")
        endifs += 1

    if (g == "AMD_GPU_ON"):
        print("if WITH_AMD_GPU_VERSION")
        endifs += 1
        
    if (p == "single"):
        if (d == "real"):
            print("if WANT_SINGLE_PRECISION_REAL")
        elif (d == "complex"):
            print("if WANT_SINGLE_PRECISION_COMPLEX")
        else:
            raise Exception("Oh no!")
        endifs += 1
    
    if (g == "NVIDIA_GPU_ON" or g == "INTEL_GPU_ON" or g == "AMD_GPU_ON" or g == "OPENMP_OFFLOAD_GPU_ON" or g == "SYCL_GPU_ON"):
      gpu_suffix="gpu_"
      if (gid):
        gpu_suffix="gpu_id_"
      if (deviceptr):
        gpu_suffix="gpu_api_"
    else:
      gpu_suffix=""

    name = "validate{langsuffix}_{d}_{p}_{gpu_suffix}{api_name}invert_triangular".format(
        langsuffix=language_flag[lang], 
        d=d, p=p, gpu_suffix=gpu_suffix,
        api_name="explicit_" if api_name == "explicit" else "")

    if (lang == "C"):
        print("if ENABLE_C_TESTS")
        endifs += 1
    if (lang == "C++"):
        print("if ENABLE_CPP_TESTS")
        endifs += 1
         
    print("check_SCRIPTS += " + name + "_default.sh")
    print("noinst_PROGRAMS += " + name)
    if lang == "Fortran":
        print(name + "_SOURCES = test/Fortran/test_invert_triangular.F90")
        print(name + "_LDADD = $(test_program_ldadd)")
        print(name + "_FCFLAGS = $(test_program_fcflags) \\")

    elif lang == "C":
        print(name + "_SOURCES = test/C/test_invert_triangular.c")
        print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
        print(name + "_CFLAGS = $(test_program_cflags) \\")
    
    elif lang == "C++":
        print(name + "_SOURCES = test/C++/test_invert_triangular.cpp")
        print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
        print(name + "_CXXFLAGS = $(test_program_cxxflags) \\")
         
    else:
        raise Exception("Unknown language")

    if (explicit_name_flag[api_name] == "-DTEST_EXPLICIT_NAME"):
      print("  " + " \\\n  ".join([
        domain_flag[d],
        prec_flag[p],
        gpu_flag[g], 
        gpu_id_flag[gid],
        device_pointer_flag[deviceptr],
        explicit_name_flag[api_name]]))
    else:
      print("  " + " \\\n  ".join([
        domain_flag[d],
        prec_flag[p],
        gpu_flag[g], 
        gpu_id_flag[gid],
        device_pointer_flag[deviceptr]]))
    print("endif\n" * endifs)
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
print("endif\n")

name = "validate_multiple_objs_real_double_c_version"
print("if BUILD_CPU_TESTS")
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
print("endif\n")
print("endif")

name = "validate_multiple_objs_real_double_cpp_version"
print("if BUILD_CPU_TESTS")
print("if ENABLE_CPP_TESTS")
print("if ENABLE_AUTOTUNING")
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/C++/test_multiple_objs.cpp")
print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
print(name + "_CXXFLAGS = $(test_program_cxxflags) \\")
print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['double']]))
print("endif")
print("endif\n")
print("endif")

for g, gid, deviceptr in product(sorted(gpu_flag.keys()),
                                 sorted(gpu_id_flag.keys()),
                                 sorted(device_pointer_flag.keys())):
  endifs = 0

  if (gid == 1):
    continue
  if (deviceptr == 1):
    continue

  if (g == "NVIDIA_GPU_ON"):
    print("if WITH_NVIDIA_GPU_VERSION")
    endifs += 1
  #if (g == "INTEL_GPU_ON"):
  #    print("if WITH_INTEL_GPU_VERSION")
  #    endifs += 1

  if (g == "OPENMP_OFFLOAD_GPU_ON"):
      print("if WITH_OPENMP_OFFLOAD_GPU_VERSION")
      endifs += 1

  if (g == "SYCL_GPU_ON"):
      print("if WITH_SYCL_GPU_VERSION")
      endifs += 1

  if (g == "AMD_GPU_ON"):
      print("if WITH_AMD_GPU_VERSION")
      endifs += 1

  if (g != "GPU_OFF"):
    name = "validate_skewsymmetric_real_double_gpu"
    print("if BUILD_GPU_TESTS")
  else:
    name = "validate_skewsymmetric_real_double"
    print("if BUILD_CPU_TESTS")
  print("if HAVE_SKEWSYMMETRIC")
  print("check_SCRIPTS += " + name + "_extended.sh")
  print("noinst_PROGRAMS += " + name)
  print(name + "_SOURCES = test/Fortran/test_skewsymmetric.F90")
  print(name + "_LDADD = $(test_program_ldadd)")
  print(name + "_FCFLAGS = $(test_program_fcflags) \\")
  print("  " + " \\\n  ".join([
          domain_flag['real'],
          prec_flag['double'],
          gpu_flag[g]]))
  print("endif\n")
  print("endif\n" * endifs)
  print("endif\n")
    
  if (g == "NVIDIA_GPU_ON"):
    print("if WITH_NVIDIA_GPU_VERSION")
  #if (g == "INTEL_GPU_ON"):
  #    print("if WITH_INTEL_GPU_VERSION")
  if (g == "OPENMP_OFFLOAD_GPU_ON"):
      print("if WITH_OPENMP_OFFLOAD_GPU_VERSION")
  if (g == "SYCL_GPU_ON"):
      print("if WITH_SYCL_GPU_VERSION")
  if (g == "AMD_GPU_ON"):
      print("if WITH_AMD_GPU_VERSION")


  if (g != "GPU_OFF"):
    name = "validate_skewsymmetric_real_single_gpu"
    print("if BUILD_GPU_TESTS")
  else:
    name = "validate_skewsymmetric_real_single"
    print("if BUILD_CPU_TESTS")
  print("if HAVE_SKEWSYMMETRIC")
  print("if WANT_SINGLE_PRECISION_REAL")
  print("check_SCRIPTS += " + name + "_extended.sh")
  print("noinst_PROGRAMS += " + name)
  print(name + "_SOURCES = test/Fortran/test_skewsymmetric.F90")
  print(name + "_LDADD = $(test_program_ldadd)")
  print(name + "_FCFLAGS = $(test_program_fcflags) \\")
  print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['single'],
        gpu_flag[g]]))
  print("endif\n")
  print("endif\n")
  print("endif\n" * endifs)
  print("endif\n")

name = "validate_real_skewsymmetric_double_c_version"
print("if BUILD_CPU_TESTS")
print("if ENABLE_C_TESTS")
print("if HAVE_SKEWSYMMETRIC")
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/C/test_skewsymmetric.c")
print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
print(name + "_CFLAGS = $(test_program_cflags) \\")
print("  " + " \\\n  ".join([
         domain_flag['real'],
         prec_flag['double']]))
print("endif\n")
print("endif\n")
print("endif\n")

name = "validate_real_skewsymmetric_double_cpp_version"
print("if BUILD_CPU_TESTS")
print("if ENABLE_CPP_TESTS")
print("if HAVE_SKEWSYMMETRIC")
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/C++/test_skewsymmetric.cpp")
print(name + "_LDADD = $(test_program_ldadd) $(FCLIBS)")
print(name + "_CXXFLAGS = $(test_program_cxxflags) \\")
print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['double']]))
print("endif")
print("endif")
print("endif")


name = "validate_split_comm_real_double"
print("if BUILD_CPU_TESTS")
print("check_SCRIPTS += " + name + "_extended.sh")
print("noinst_PROGRAMS += " + name)
print(name + "_SOURCES = test/Fortran/test_split_comm.F90")
print(name + "_LDADD = $(test_program_ldadd)")
print(name + "_FCFLAGS = $(test_program_fcflags) \\")
print("  " + " \\\n  ".join([
        domain_flag['real'],
        prec_flag['double']]))
print("endif")
