#!/usr/bin/env python

from __future__ import print_function
from itertools import product

def set_number_of_cores(mpi_tasks, o):
    cores = 1
    cores = 1 * mpi_tasks

    if (o == "openmp"):
        cores = cores * 2
    return cores

def set_requested_memory(na):
    memory="None"
    if (na == "150"):
            memory="2Gb"
    elif (na > "150" and na <= "1500"):
        memory="12Gb"
    elif (na > "1500" and na < "5000"):
        memory="12Gb"
    else:
        memory="16Gb"
    return memory

def set_compiler_wrappers(mpi, fc, cc, instr, fortran_compiler, c_compiler):
    fortran_compiler_wrapper="undefined"
    c_compiler_wrapper = "undefined"
    if (instr != "power8"):
        if (m == "mpi" and fc == "intel"):
            fortran_compiler_wrapper="mpiifort"
        if (m == "mpi" and fc == "gnu"):
            fortran_compiler_wrapper="mpif90"
        if (m == "mpi" and cc == "intel"):
            c_compiler_wrapper="mpiicc"
        if (m == "mpi" and cc == "gnu"):
            c_compiler_wrapper="mpicc"

        if (m == "nompi" and fc == "intel"):
            fortran_compiler_wrapper=fortran_compiler[fc]
        if (m == "nompi" and fc == "gnu"):
            fortran_compiler_wrapper=fortran_compiler[fc]
        if (m == "nompi" and cc == "intel"):
            c_compiler_wrapper=c_compiler[cc]
        if (m == "nompi" and cc == "gnu"):
            c_compiler_wrapper=c_compiler[cc]
    else:
        if (m == "mpi" and fc == "pgi"):
            fortran_compiler_wrapper="mpifort"
        if (m == "mpi" and fc == "gnu"):
           fortran_compiler_wrapper="mpifort"
        if (m == "mpi" and cc == "gnu"):
            c_compiler_wrapper="mpicc"
        if (m == "mpi" and cc == "gnu"):
            c_compiler_wrapper="mpicc"

    return (fortran_compiler_wrapper, c_compiler_wrapper)

def set_scalapack_flags(instr, fc, g, m, o):
    scalapackldflags="undefined"
    scalapackfcflags="undefined"
    libs="undefined"
    ldflags="undefined"

    if (instr != "power8"):
        if (fc == "intel"):
            if (m == "mpi"):
                if (o == "openmp"):
                    scalapackldflags="$MKL_INTEL_SCALAPACK_LDFLAGS_MPI_OMP "
                    scalapackfcflags="$MKL_INTEL_SCALAPACK_FCFLAGS_MPI_OMP "
                else:
                    scalapackldflags="$MKL_INTEL_SCALAPACK_LDFLAGS_MPI_NO_OMP "
                    scalapackfcflags="$MKL_INTEL_SCALAPACK_FCFLAGS_MPI_NO_OMP "
            else:
                if (o == "openmp"):
                    scalapackldflags="$MKL_INTEL_SCALAPACK_LDFLAGS_NO_MPI_OMP "
                    scalapackfcflags="$MKL_INTEL_SCALAPACK_FCFLAGS_NO_MPI_OMP "
                else:
                    scalapackldflags="$MKL_INTEL_SCALAPACK_LDFLAGS_NO_MPI_NO_OMP "
                    scalapackfcflags="$MKL_INTEL_SCALAPACK_FCFLAGS_NO_MPI_NO_OMP "

        if (fc == "gnu"):
            if (m == "mpi"):
                if (o == "openmp"):
                    scalapackldflags="$MKL_GFORTRAN_SCALAPACK_LDFLAGS_MPI_OMP "
                    scalapackfcflags="$MKL_GFORTRAN_SCALAPACK_FCFLAGS_MPI_OMP "
                else:
                    scalapackldflags="$MKL_GFORTRAN_SCALAPACK_LDFLAGS_MPI_NO_OMP "
                    scalapackfcflags="$MKL_GFORTRAN_SCALAPACK_FCFLAGS_MPI_NO_OMP "
            else:
                if (o == "openmp"):
                    scalapackldflags="$MKL_GFORTRAN_SCALAPACK_LDFLAGS_NO_MPI_OMP "
                    scalapackfcflags="$MKL_GFORTRAN_SCALAPACK_FCFLAGS_NO_MPI_OMP "
                else:
                    scalapackldflags="$MKL_GFORTRAN_SCALAPACK_LDFLAGS_NO_MPI_NO_OMP "
                    scalapackfcflags="$MKL_GFORTRAN_SCALAPACK_FCFLAGS_NO_MPI_NO_OMP "

        if (g == "with-gpu"):
            scalapackldflags += " -L\\$CUDA_HOME/lib64 -lcublas -I\\$CUDA_HOME/include"
            scalapackfcflags += " -I\\$CUDA_HOME/include"

        if (instr == "sse" or (instr == "avx" and g != "with-gpu")):
            scalapackldflags = " SCALAPACK_LDFLAGS=\\\""+scalapackldflags+"\\\""
            scalapackfcflags = " SCALAPACK_FCFLAGS=\\\""+scalapackfcflags+"\\\""

        if ( instr == "avx2" or instr == "avx512" or instr == "knl" or g == "with-gpu"):
            scalapackldflags = " SCALAPACK_LDFLAGS=\\\""+"\\"+scalapackldflags+"\\\""
            scalapackfcflags = " SCALAPACK_FCFLAGS=\\\""+"\\"+scalapackfcflags+"\\\""

        libs=""
        ldflags=""
    else:
        # on power 8
        scalapackldflags = ""
        scalapackfcflags = ""
        libs = " LIBS=\\\" -lessl -lreflapack -lessl -lcublas -lgfortran \\\""
        ldflags = " LDFLAGS=\\\" -L/home/elpa/libs/scalapack/lib -L\\$CUDA_HOME/lib64 \\\""


    return (scalapackldflags,scalapackfcflags,libs,ldflags)

def set_cflags_fcflags(instr, cc, fc, instruction_set):
    CFLAGS=""
    FCFLAGS=""
    INSTRUCTION_OPTIONS=""

    if (instr == "avx512"):
        INSTRUCTION_OPTIONS = " --enable-avx512"
        if (cc == "gnu"):
            CFLAGS += "-O3 -march=skylake-avx512"
        else:
            CFLAGS += "-O3 -xCORE-AVX512"
        if (fc == "gnu"):
            FCFLAGS += "-O3 -march=skylake-avx512"
        else:
            FCFLAGS += "-O3 -xCORE-AVX512"

    if (instr == "knl"):
        INSTRUCTION_OPTIONS = " --enable-avx512"
        if (cc == "gnu"):
            CFLAGS += "-O3 -march=native"
        else:
            CFLAGS += "-O3 -xMIC-AVX512"
        if (fc == "gnu"):
            FCFLAGS += "-O3 -march=native"
        else:
            FCFLAGS += "-O3 -xMIC-AVX512"

    if (instr == "avx2"):
        INSTRUCTION_OPTIONS = instruction_set[instr]
        if (cc == "gnu"):
            CFLAGS += "-O3 -mavx2 -mfma"
        else:
            CFLAGS += "-O3 -xAVX2"
        if (fc == "gnu"):
            FCFLAGS += "-O3 -mavx2 -mfma"
        else:
            FCFLAGS += "-O3 -xAVX2"

    if (instr == "avx"):
        INSTRUCTION_OPTIONS = instruction_set[instr] + " --disable-avx2"
        if (cc == "gnu"):
           CFLAGS  += "-O3 -mavx"
           FCFLAGS += "-O3 -mavx"
        else:
           CFLAGS  += "-O3 -xAVX"
           FCFLAGS += "-O3 -xAVX"

    if (instr == "sse"):
        INSTRUCTION_OPTIONS = instruction_set[instr] + " --disable-avx --disable-avx2"
        if (cc == "gnu"):
            CFLAGS  +="-O3 -msse4.2"
            FCFLAGS +="-O3 -msse4.2"
        else:
            CFLAGS  +="-O3 -xSSE4.2"
            FCFLAGS +="-O3 -xSSE4.2"

    if (instr == "power8"):
       INSTRUCTION_OPTIONS = instruction_set[instr]
       CFLAGS +="-O2 -I\$CUDA_HOME/include"
       FCFLAGS +="-O2"

    return (CFLAGS, FCFLAGS, INSTRUCTION_OPTIONS)


#define the stages
print("stages:")
print("  - test")
#print("  - coverage")
#print("  - deploy")
print("\n\n")

#define before test actions
print("before_script:")
print("  - git clean -f")
print("  - export LANG=C")
print("  - ulimit -s unlimited")
print("  - ulimit -v unlimited")
print("  - echo \"HOST \" $(hostname)")
print("  - echo $CI_RUNNER_DESCRIPTION")
print("  - export SLURM=yes")
print("  - export INTERACTIVE_RUN=yes")
print("  - if [ \"$(hostname)\" = \"buildtest-rzg\" ]; then module purge && module load git && module list && export INTERACTIVE_RUN=yes && export SLURM=no && source ./ci_test_scripts/.ci-env-vars; fi")
print("  - if [ \"$(hostname)\" = \"amarek-elpa-gitlab-runner-1\" ]; then module purge && module load git && module list && export INTERACTIVE_RUN=yes && export SLURM=no && source ./ci_test_scripts/.ci-env-vars; fi")
print("  - if [ \"$(hostname)\" = \"amarek-elpa-gitlab-runner-2\" ]; then module purge && module load git && module list && export INTERACTIVE_RUN=yes && export SLURM=no && source ./ci_test_scripts/.ci-env-vars; fi")
print("  - if [ \"$(hostname)\" = \"amarek-elpa-gitlab-runner-3\" ]; then module purge && module load git && module list && export INTERACTIVE_RUN=yes && export SLURM=no && source ./ci_test_scripts/.ci-env-vars; fi")
print("  - if [ \"$(hostname)\" = \"amarek-elpa-gitlab-runner-4\" ]; then module purge && module load git && module list && export INTERACTIVE_RUN=yes && export SLURM=no && source ./ci_test_scripts/.ci-env-vars; fi")
print("  - module list")
print("  - source ./ci_test_scripts/.ci-env-vars")

print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"freya01-interactive\" ]; then export INTERACTIVE_RUN=yes && export SLURM=no; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"freya01-interactive-2\" ]; then export INTERACTIVE_RUN=yes && export SLURM=no; fi")

print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-1\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-2\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-3\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-4\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-5\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-6\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-7\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-8\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-9\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-gp02-10\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=gp02 && export SLURMPARTITION=gp && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"skylake\" ; fi")
print("\n")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-1\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-2\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-3\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-4\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-5\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-6\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-7\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-8\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-9\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-10\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-11\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-12\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-13\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-14\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-15\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-16\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-17\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-18\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-19\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl1-20\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl1 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ;  fi")
print("\n")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl2\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl2 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ; fi")
print("\n")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl3\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl3 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ; fi")
print("\n")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-knl4\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=knl4 && export SLURMPARTITION=knl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ; fi")
print("\n")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-maik\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=maik && export SLURMPARTITION=maik && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"knl\" ; fi")

print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-dvl01\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=dvl01 && export SLURMPARTITION=dvl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140  && export CONTSTRAINTS=\"x86_64&gpu0&gpu1\" && export GEOMETRYRESERVATION=\"gpu:2\" ; fi")
print("\n")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-dvl02\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=dvl02 && export SLURMPARTITION=dvl && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"x86_64&gpu0&gpu1\" && export GEOMETRYRESERVATION=\"gpu:2\" ; fi")
print("\n")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-miy01\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=miy01 && export SLURMPARTITION=minsky && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"power8&gpu0&gpu1&gpu2&gpu3\" && export GEOMETRYRESERVATION=\"gpu:4\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-miy02\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=miy02 && export SLURMPARTITION=minsky && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"power8&gpu0&gpu1&gpu2&gpu3\" && export GEOMETRYRESERVATION=\"gpu:4\" ; fi")
print("  - if [ \"$CI_RUNNER_DESCRIPTION\" = \"appdev-miy03\" ]; then export INTERACTIVE_RUN=no && export SLURM=no && export SLURMHOST=miy03 && export SLURMPARTITION=minsky && export CONFIGURETIME=15 && export BUILDTIME=80 && export RUNTIME=140 && export CONTSTRAINTS=\"power8&gpu0&gpu1&gpu2&gpu3\" && export GEOMETRYRESERVATION=\"gpu:4\" ; fi")
print("\n")
print("  - export MATRIX_SIZE=150")
print("  - export NUMBER_OF_EIGENVECTORS=150")
print("  - export BLOCK_SIZE=16")
print("  - if [ \"$MEDIUM_MATRIX\" = \"yes\" ]; then export MATRIX_SIZE=1500 && export NUMBER_OF_EIGENVECTORS=750; fi")
print("  - if [ \"$LARGE_MATRIX\" = \"yes\" ]; then export MATRIX_SIZE=5000 && export NUMBER_OF_EIGENVECTORS=500; fi")
print("  - if [ \"$GPU_BLOCKSIZE\" = \"yes\" ]; then export BLOCK_SIZE=128 ; fi")
print("  - echo \"This test will run with matrix size na = $MATRIX_SIZE, nev= $NUMBER_OF_EIGENVECTORS, on a blacs grid with blocksize nblk= $BLOCK_SIZE \" ")
print("  - export SKIP_STEP=0")
print("  - ./autogen.sh")
print("  - export SKIP_STEP=0")
print("  - if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh ; else source /etc/profile.d/mpcdf_modules.sh; fi  && . ./ci_test_scripts/.ci-env-vars")

print("\n\n")

#define after script
print("# For some reason sometimes not-writable files remain, which cause trouble the")
print("# next time a runner tries to clean-up")
print("after_script:")
print("  - chmod u+w -R .")
print("\n\n")


##define the total coverage phase
#print("# print coverage results")
#print("total_coverage:")
#print("  only:")
#print("    - /.*master.*/")
#print("  stage: coverage")
#print("  tags:")
#print("    - coverage")
#print("  script:")
#print("    - echo \"Generating coverage report\"")
#print("    - ./ci_test_scripts/ci_coverage_summary")
#print("  artifacts:")
#print("    paths:")
#print("      - public")
#print("\n\n")
#
#print("pages:")
#print("  stage: deploy")
#print("  tags:")
#print("    - coverage")
#print("  script:")
#print("    - echo \"Publishing pages\"")
#print("  artifacts:")
#print("    paths:")
#print("      - public")
#print("  only:")
#print("    - master")
#print("\n\n")

print("static-build:")
print("  tags:")
print("    - avx")
print("  script:")
print("    - ./ci_test_scripts/run_ci_tests.sh -c \" CFLAGS=\\\"-O3 -mavx\\\" FCFLAGS=\\\"-O3 -axAVX\\\" SCALAPACK_LDFLAGS=\\\"$MKL_INTEL_SCALAPACK_LDFLAGS_NO_MPI_NO_OMP\\\"  \
        SCALAPACK_FCFLAGS=\\\"$MKL_INTEL_SCALAPACK_FCFLAGS_NO_MPI_NO_OMP\\\" --with-mpi=no FC=ifort --enable-shared=no --enable-static=yes --disable-avx2 || { cat config.log; exit 1; } \" -j 8 \
        -t 2 -m 150 -n 50 -b 16 -s $SKIP_STEP -i $INTERACTIVE_RUN -S $SLURM ")
print("\n\n")

print("# test distcheck")
print("distcheck:")
print("  tags:")
print("    - distcheck")
print("  script:")
print("    - ./ci_test_scripts/run_ci_tests.sh -c \" CC=gcc FC=gfortran SCALAPACK_LDFLAGS=\\\"$MKL_GFORTRAN_SCALAPACK_LDFLAGS_NO_MPI_NO_OMP\\\"  \
        SCALAPACK_FCFLAGS=\\\"$MKL_GFORTRAN_SCALAPACK_FCFLAGS_NO_MPI_NO_OMP\\\" --enable-option-checking=fatal --with-mpi=no --disable-sse-assembly \
        --disable-sse --disable-avx --disable-avx2 || { cat config.log; exit 1; } \" -t 2 -m 150 -n 50 -b 16 -s $SKIP_STEP -i $INTERACTIVE_RUN -S $SLURM ")
print("    - ./ci_test_scripts/run_distcheck_tests.sh  -c \"  CC=gcc FC=gfortran SCALAPACK_LDFLAGS=\\\\\\\"$MKL_GFORTRAN_SCALAPACK_LDFLAGS_NO_MPI_NO_OMP\\\\\\\"  \
        SCALAPACK_FCFLAGS=\\\\\\\"$MKL_GFORTRAN_SCALAPACK_FCFLAGS_NO_MPI_NO_OMP\\\\\\\" --enable-option-checking=fatal --with-mpi=no --disable-sse-assembly \
        --disable-sse --disable-avx --disable-avx2 \" -t 2 -m 150 -n 50 -b 16 -S$SLURM ")
print("\n\n")

print("distcheck-mpi:")
print("  tags:")
print("    - distcheck")
print("  script:")
print("    - ./ci_test_scripts/run_ci_tests.sh -c \" FC=mpiifort FCFLAGS=\\\"-xHost\\\" CFLAGS=\\\"-march=native\\\" \
  SCALAPACK_LDFLAGS=\\\"$MKL_INTEL_SCALAPACK_LDFLAGS_MPI_NO_OMP\\\" \
  SCALAPACK_FCFLAGS=\\\"$MKL_INTEL_SCALAPACK_FCFLAGS_MPI_NO_OMP\\\" \
  --enable-option-checking=fatal --with-mpi=yes \
 --disable-sse-assembly --disable-sse --disable-avx --disable-avx2 || { cat config.log; exit 1; } \" -t 2 -m 150 \
 -n 50 -b 16 -s $SKIP_STEP -i $INTERACTIVE_RUN -S $SLURM ")
print("    - ./ci_test_scripts/run_distcheck_tests.sh  -c \" FC=mpiifort FCFLAGS=\\\\\\\"-xHost\\\\\\\" \
 CFLAGS=\\\\\\\"-march=native\\\\\\\" SCALAPACK_LDFLAGS=\\\\\\\"$MKL_INTEL_SCALAPACK_LDFLAGS_MPI_NO_OMP\\\\\\\"  \
 SCALAPACK_FCFLAGS=\\\\\\\"$MKL_INTEL_SCALAPACK_FCFLAGS_MPI_NO_OMP\\\\\\\" --enable-option-checking=fatal \
 --with-mpi=yes --disable-sse-assembly --disable-sse --disable-avx --disable-avx2 \" -t 2 -m 150 -n 50 -b 16 -S$SLURM ")
#print('    - make distcheck DISTCHECK_CONFIGURE_FLAGS="FC=mpiifort FCFLAGS=\\"-xHost\\" CFLAGS=\\"-march=native\\" SCALAPACK_LDFLAGS=\\"$MKL_INTEL_SCALAPACK_LDFLAGS_MPI_NO_OMP\\" SCALAPACK_FCFLAGS=\\"$MKL_INTEL_SCALAPACK_FCFLAGS_MPI_NO_OMP\\" --with-mpi=yes --disable-sse-assembly --disable-sse --disable-avx --disable-avx2" TASKS=2 TEST_FLAGS="150 50 16" || { chmod u+rwX -R . ; exit 1 ; }')
print("\n\n")

print("distcheck-no-autotune:")
print("  tags:")
print("    - distcheck")
print("  script:")
print("    - ./ci_test_scripts/run_ci_tests.sh -c \" FC=mpiifort FCFLAGS=\\\"-xHost\\\" CFLAGS=\\\"-march=native\\\" \
  SCALAPACK_LDFLAGS=\\\"$MKL_INTEL_SCALAPACK_LDFLAGS_MPI_NO_OMP\\\" \
  SCALAPACK_FCFLAGS=\\\"$MKL_INTEL_SCALAPACK_FCFLAGS_MPI_NO_OMP\\\" \
  --enable-option-checking=fatal --with-mpi=yes \
  --disable-sse-assembly --disable-sse --disable-avx --disable-avx2 --disable-autotuning || { cat config.log; exit 1; } \" -t 2 -m 150 \
  -n 50 -b 16 -s $SKIP_STEP -i $INTERACTIVE_RUN -S $SLURM ")
print("    - ./ci_test_scripts/run_distcheck_tests.sh  -c \" FC=mpiifort FCFLAGS=\\\\\\\"-xHost\\\\\\\" \
 CFLAGS=\\\\\\\"-march=native\\\\\\\" SCALAPACK_LDFLAGS=\\\\\\\"$MKL_INTEL_SCALAPACK_LDFLAGS_MPI_NO_OMP\\\\\\\"  \
 SCALAPACK_FCFLAGS=\\\\\\\"$MKL_INTEL_SCALAPACK_FCFLAGS_MPI_NO_OMP\\\\\\\" --enable-option-checking=fatal \
 --with-mpi=yes --disable-sse-assembly --disable-sse --disable-avx --disable-avx2 --disable-autotuning \" -t 2 -m 150 -n 50 -b 16 -S$SLURM ")
print("\n\n")





# add python tests
python_ci_tests = [
    "# python tests",
    "python-intel-intel-mpi-openmp:",
    "  tags:",
    "    - python",
    "  artifacts:",
    "    when: on_success",
    "    expire_in: 2 month",
    "  script:",
    '   - ./ci_test_scripts/run_ci_tests.sh -c "'
    'CC=\\"mpiicc\\" CFLAGS=\\"-O3 -xAVX\\" '
    'FC=\\"mpiifort\\" FCFLAGS=\\"-O3 -xAVX\\" '
    'SCALAPACK_LDFLAGS=\\"$MKL_ANACONDA_INTEL_SCALAPACK_LDFLAGS_MPI_OMP \\" '
    'SCALAPACK_FCFLAGS=\\"$MKL_ANACONDA_INTEL_SCALAPACK_FCFLAGS_MPI_OMP \\" '
    '--enable-option-checking=fatal --with-mpi=yes --enable-openmp '
    '--disable-gpu --enable-avx --enable-python --enable-python-tests'
    '" -j 8 -t 2 -m $MATRIX_SIZE -n $NUMBER_OF_EIGENVECTORS -b $BLOCK_SIZE '
    '-s $SKIP_STEP -i $INTERACTIVE_RUN -S $SLURM',
    "\n",
    "python-distcheck:",
    "  tags:",
    "    - python",
    "  script:",
    '    - ./configure '
    'CC="mpiicc" CFLAGS="-O3 -xAVX" '
    'FC="mpiifort" FCFLAGS="-O3 -xAVX" '
    'SCALAPACK_LDFLAGS="$MKL_ANACONDA_INTEL_SCALAPACK_LDFLAGS_MPI_OMP" '
    'SCALAPACK_FCFLAGS="$MKL_ANACONDA_INTEL_SCALAPACK_FCFLAGS_MPI_OMP" '
    '--enable-option-checking=fatal --with-mpi=yes --enable-openmp '
    '--disable-gpu --enable-avx --enable-python --enable-python-tests'
    ' || { cat config.log; exit 1; }',
    "    # stupid 'make distcheck' leaves behind write-protected files that "
    "the stupid gitlab runner cannot remove",
    '    - make distcheck DISTCHECK_CONFIGURE_FLAGS="'
    'CC=\\"mpiicc\\" CFLAGS=\\"-O3 -xAVX\\" '
    'FC=\\"mpiifort\\" FCFLAGS=\\"-O3 -xAVX\\" '
    'SCALAPACK_LDFLAGS=\\"$MKL_ANACONDA_INTEL_SCALAPACK_LDFLAGS_MPI_OMP \\" '
    'SCALAPACK_FCFLAGS=\\"$MKL_ANACONDA_INTEL_SCALAPACK_FCFLAGS_MPI_OMP \\" '
    '--enable-option-checking=fatal --with-mpi=yes --enable-openmp '
    '--disable-gpu --enable-avx --enable-python --enable-python-tests'
    '" TASKS=2 TEST_FLAGS="150 50 16" || { chmod u+rwX -R . ; exit 1 ; }',
    "\n",
]
print("\n".join(python_ci_tests))


# construct the builds of the "test_projects"

stage = {
          "1stage" : "1stage",
          "2stage" : "2stage",
}

api = {
        "new_api" : "",
        "legacy_api" : "_legacy_api",
}

compiler = {
             "gnu" : "gnu",
             "intel" : "intel"
}
for comp, s, a in product(
                             sorted(compiler.keys()),
                             sorted(stage.keys()),
                             sorted(api.keys())):

    print("# test_project_"+stage[s]+api[a]+"_"+compiler[comp])
    print("test_project_"+stage[s]+api[a]+"_"+compiler[comp]+":")
    print("  tags:")
    print("    - project_test")
    print("  script:")
    if ( s == "1stage"):
        projectBinary="test_real"
    else:
        projectBinary="test_real2"

    if (comp == "intel"):
        print("    - ./ci_test_scripts/run_project_tests.sh -c \" FC=mpiifort FCFLAGS=\\\"-march=native \\\" CFLAGS=\\\"-march=native\\\" \
                SCALAPACK_LDFLAGS=\\\"$MKL_INTEL_SCALAPACK_LDFLAGS_MPI_NO_OMP\\\" \
                SCALAPACK_FCFLAGS=\\\"$MKL_INTEL_SCALAPACK_FCFLAGS_MPI_NO_OMP\\\" \
                --enable-option-checking=fatal  --disable-avx2 --prefix=$PWD/installdest --disable-avx2 || { cat config.log; exit 1; } \" \
                -t 2 -m 150 -n 50 -b 16 -S $SLURM -p test_project_"+stage[s]+api[a]+" -e "+projectBinary+" \
                -C \" FC=mpiifort PKG_CONFIG_PATH=../../installdest/lib/pkgconfig  \
                 --enable-option-checking=fatal || { cat config.log; exit 1; } \" ")
    if (comp == "gnu"):
        print("    - ./ci_test_scripts/run_project_tests.sh -c \" FC=mpif90 FCFLAGS=\\\"-march=native \\\" CFLAGS=\\\"-march=native\\\" \
                SCALAPACK_LDFLAGS=\\\"$MKL_GFORTRAN_SCALAPACK_LDFLAGS_MPI_NO_OMP\\\" \
                SCALAPACK_FCFLAGS=\\\"$MKL_GFORTRAN_SCALAPACK_FCFLAGS_MPI_NO_OMP\\\" \
                --enable-option-checking=fatal  --disable-avx2 --prefix=$PWD/installdest --disable-avx2 || { cat config.log; exit 1; } \" \
                -t 2 -m 150 -n 50 -b 16 -S $SLURM -p test_project_"+stage[s]+api[a]+" -e "+projectBinary+" \
                -C \" FC=mpif90 PKG_CONFIG_PATH=../../installdest/lib/pkgconfig \
                 --enable-option-checking=fatal || { cat config.log; exit 1; } \" ")
    print("\n\n")

print("#The tests follow here")

c_compiler = {
        "gnu"   : "gcc",
        "intel" : "icc",
}
fortran_compiler = {
        "gnu" : "gfortran",
        "intel" : "ifort",
        "pgi"   : "pgfortran",
}
mpi = {
        "mpi"   : "--with-mpi=yes",
        "nompi" : "--with-mpi=no --disable-mpi-module",
}

openmp = {
        "openmp"   : "--enable-openmp",
        "noopenmp" : "--disable-openmp",
}

precision = {
        "double-precision" : "--disable-single-precision",
        "single-precision" : "--enable-single-precision",
}

assumed_size = {
        "assumed-size" : "--enable-assumed-size",
        "no-assumed-size" : "--disable-assumed-size",
}

band_to_full_blocking = {
        "band-to-full-blocking" : "--enable-band-to-full-blocking",
        "no-band-to-full-blocking" : "--disable-band-to-full-blocking",
}

gpu = {
        "with-gpu" : "--enable-gpu --with-cuda-path=\\$CUDA_HOME/",
        "no-gpu" : "--disable-gpu",
}

coverage = {
        "coverage" : "coverage",
        "no-coverage": "no-coverage",
}

        #"knl" : "--enable-avx512",
instruction_set = {
        "sse" : " --enable-sse --enable-sse-assembly",
        "avx" : " --enable-avx",
        "avx2" : " --enable-avx2",
        "avx512" : "--enable-avx512",
        "power8" : " --enable-vsx --disable-sse --disable-sse-assembly --disable-avx --disable-avx2 --disable-mpi-module --with-GPU-compute-capability=sm_60 ",
}

address_sanitize_flag = {
        "address-sanitize" : "address-sanitize",
        "no-address-sanitize" : "no-address-sanitize",
}

#matrix_size = {
#        "small"   : "150",
#        "medium"  : "1500",
#        "large"   : "5000",
#}

matrix_size = {
        "small"   : "150",
}

MPI_TASKS=2
#                             sorted(coverage.keys()),     
for cc, fc, m, o, p, a, b, g, instr, addr, na in product(
                             sorted(c_compiler.keys()),
                             sorted(fortran_compiler.keys()),
                             sorted(mpi.keys()),
                             sorted(openmp.keys()),
                             sorted(precision.keys()),
                             sorted(assumed_size.keys()),
                             sorted(band_to_full_blocking.keys()),
                             sorted(gpu.keys()),
                             sorted(instruction_set.keys()),
                             sorted(address_sanitize_flag.keys()),
                             sorted(matrix_size.keys())):

    cov = "no-coverage"

    nev = 150
    nblk = 16

    # do not all combinations with all compilers
    # especially - use pgi only on minskys for now
    #            - do not allow to use FC=gfortran but CC=intel
    #            - if pgfortran => use always GPUs
    #            - if gfortran disable MPI module
    #            - on KNL only use intel, do not test openmp
    if (instr == "power8" and (fc !="pgi" and fc !="gnu")):
        continue
    if (instr == "knl" and (fc !="intel" and cc !="intel")):
        continue
    if (instr == "knl" and o == "openmp"):
        continue
    if (fc == "pgi" and instr !="power8"):
        continue
    if ( cc == "intel" and fc == "gnu"):
        continue
    if (fc == "pgi" and g !="with-gpu"):
        continue
    mpi_configure_flag = mpi[m]
    if (fc == "gnu" and  m == "mpi"):
        mpi_configure_flag += " --disable-mpi-module"

    # on power8 only test with mpi and gpu
    if (instr == "power8" and (m == "nompi" or g == "no-gpu")):
        continue

    # set C and FCFLAGS according to instruction set
    (CFLAGS, FCFLAGS, INSTRUCTION_OPTIONS) = set_cflags_fcflags(instr, cc, fc, instruction_set)


    #coverage need O0 and only with gnu and no coverage in combination with address-sanitize
    if (cov == "coverage" and addr == "address-sanitize"):
        continue
    if (cov == "coverage" and (cc != "gnu" or fc != "gnu")):
        continue
    if (cov == "coverage" and instr == "sse"):
        continue
    if (cov == "coverage" and instr == "knl"):
        continue
    if (cov == "coverage" and g == "with-gpu"):
        continue
    if (cov == "coverage"):
        CFLAGS +=" --coverage -O0"
        FCFLAGS +=" --coverage -O0"

    # add tests for scalapack for some specific test cases
    runScalapackTest = False
    if (instr == "avx2" and cov == "coverage" and m == "mpi"):
        runScalapackTest = True


    # address-sanitize only with gnu compiler
    if (addr == "address-sanitize" and (cc != "gnu" and fc != "gnu")):
        continue
    if (instr == "power8" and addr == "address-sanitize"):
        continue
    if (g == "with-gpu" and addr == "address-sanitize"):
        continue
    if (instr == "knl" and addr == "address-sanitize"):
        continue

    if (addr == "address-sanitize" and (cc == "gnu" and fc == "gnu")):
        CFLAGS += " -fsanitize=address"
        FCFLAGS += " -fsanitize=address"

    # make sure that CFLAGS and FCFLAGS are not empty
    if (CFLAGS == ""):
        CFLAGS = "-O3"
    if (FCFLAGS == ""):
        FCFLAGS = "-O3"

    #KNL is slow: only run single-precision (which includes double-precision)
    if (p == "double-precision" and instr == "knl"):
        continue

    #no gpu testing with openmp
    if (g == "with-gpu" and o == "openmp"):
        continue

    #no gpu testing with intel C compiler (gcc needed)
    if (g == "with-gpu" and cc == "intel"):
        continue

    #at the moment gpu testing only on AVX machines or minskys
    if (g == "with-gpu" and (instr !="avx" and instr !="power8")):
        continue

#    #on KNL do only intel tests
#    if (instr == "knl" and (cc == "gnu" or fc == "gnu")):
#        continue


    # do to the number of tests, do some only on branchens master and master-pre-stage
    # default is: always test
    # - knl is really slow in building => turn-arround time is aweful
    # - coverage only on masters
    # - double precision only masters (is tested anyway with single precision)
    # - sanitize-address always for sse and avx, else only on master
    MasterOnly=False
    if ( instr == "knl"):
        MasterOnly=True
    if (cov == "coverage"):
        MasterOnly=True
    if (p == "double-precision"):
        MasterOnly=True
    if (instr != "avx" and instr != "sse" and addr == "address-sanitize"):
        MasterOnly=True

    # make non-master tests even faster
    # kicking out gpu is not good, but at the momemt we have a real problem with gpu runners
    # should be returned when solved
    if (g == "with-gpu"):
        MasterOnly=True
    if (a == "no-assumed-size"):
        MasterOnly=True
    if (instr == "avx2" or instr == "avx512"):
        MasterOnly=True

    print("# " + cc + "-" + fc + "-" + m + "-" + o + "-" + p + "-" + a + "-" + b + "-" +g + "-" + cov + "-" + instr + "-" + addr)
    print(cc + "-" + fc + "-" + m + "-" + o + "-" + p + "-" +a + "-" +b + "-" +g + "-" + cov + "-" + instr + "-" + addr + "-jobs:")
    if (MasterOnly):
        print("  only:")
        print("    - /.*master.*/")
    if (instr == "power8"):
        print("  allow_failure: true")
    print("  tags:")
    if (cov == "coverage"):
        if (instr == "avx"):
            print("    - coverage")
        if (instr == "avx2"):
            print("    - avx2-coverage")
        if (g == "with-gpu"):
            print("    - gpu-coverage")
        if (instr == "avx512"):
            print("    - avx512-coverage")
    else:
        if (g == "with-gpu"):
            if (instr == "power8"):
               print("    - minsky")
            else:
               print("    - gpu")
        else:
            print("    - " + instr)
    print("  artifacts:")
    print("    when: on_success")
    print("    expire_in: 2 month")
    print("  script:")

    (fortran_compiler_wrapper, c_compiler_wrapper) = set_compiler_wrappers(mpi, fc, cc, instr, fortran_compiler, c_compiler)

    (scalapackldflags,scalapackfcflags,libs,ldflags) = set_scalapack_flags(instr, fc, g, m, o)

    memory = set_requested_memory(matrix_size[na])

    # do the configure
    if ( instr == "sse" or (instr == "avx" and g != "with-gpu")):
        if ( instr == "sse"):
            print("   - if [ $MATRIX_SIZE -gt 150 ]; then export SKIP_STEP=1 ; fi # our SSE test machines do not have a lot of memory")
        print("   - ./ci_test_scripts/run_ci_tests.sh -c \" CC=\\\""+c_compiler_wrapper+"\\\"" + " CFLAGS=\\\""+CFLAGS+"\\\"" + " FC=\\\""+fortran_compiler_wrapper+"\\\"" + " FCFLAGS=\\\""+FCFLAGS+"\\\"" \
                + libs + " " + ldflags + " " + " "+ scalapackldflags +" " + scalapackfcflags \
                + " --enable-option-checking=fatal" + " " + mpi_configure_flag + " " + openmp[o] \
+ " " + precision[p] + " " + assumed_size[a] + " " + band_to_full_blocking[b] \
+ " " +gpu[g] + INSTRUCTION_OPTIONS + "\" -j 8 -t " + str(MPI_TASKS) + " -m $MATRIX_SIZE -n $NUMBER_OF_EIGENVECTORS -b $BLOCK_SIZE -s $SKIP_STEP -i $INTERACTIVE_RUN -S $SLURM")

    if ( instr == "avx2" or instr == "avx512" or instr == "knl" or g == "with-gpu"):
        print("    - export REQUESTED_MEMORY="+memory)    
        print("\n")
        if (g == "with-gpu"):
            print("    - echo \"The tasks will be submitted to SLURM PARTITION \" $SLURMPARTITION \" on host \" $SLURMHOST \" with constraints \" $CONTSTRAINTS \" with the geometry \" $GEOMETRYRESERVATION" )
        else:
            print("    - echo \"The tasks will be submitted to SLURM PARTITION \" $SLURMPARTITION \" on host \" $SLURMHOST \"with constraints \" $CONTSTRAINTS ")

        # construct srun command-line
        if (g == "with-gpu"):
            print("    - export SRUN_COMMANDLINE_CONFIGURE=\"--partition=$SLURMPARTITION --nodelist=$SLURMHOST --time=$CONFIGURETIME --constraint=$CONTSTRAINTS --gres=$GEOMETRYRESERVATION \" ")
            print("    - export SRUN_COMMANDLINE_BUILD=\"--partition=$SLURMPARTITION --nodelist=$SLURMHOST --time=$BUILDTIME --constraint=$CONTSTRAINTS --gres=$GEOMETRYRESERVATION \" ")
            print("    - export SRUN_COMMANDLINE_RUN=\"--partition=$SLURMPARTITION --nodelist=$SLURMHOST --time=$RUNTIME --constraint=$CONTSTRAINTS --gres=$GEOMETRYRESERVATION \" ")
        else:
            print("    - export SRUN_COMMANDLINE_CONFIGURE=\"--partition=$SLURMPARTITION --nodelist=$SLURMHOST --time=$CONFIGURETIME --constraint=$CONTSTRAINTS --mem=$REQUESTED_MEMORY\" ")
            print("    - export SRUN_COMMANDLINE_BUILD=\"--partition=$SLURMPARTITION --nodelist=$SLURMHOST --time=$BUILDTIME --constraint=$CONTSTRAINTS --mem=$REQUESTED_MEMORY \" ")
            print("    - export SRUN_COMMANDLINE_RUN=\"--partition=$SLURMPARTITION --nodelist=$SLURMHOST --time=$RUNTIME --constraint=$CONTSTRAINTS --mem=$REQUESTED_MEMORY \" ")
        #print("    - echo \"srun --ntasks=1 --cpus-per-task=1 $SRUN_COMMANDLINE_CONFIGURE\" ")

        if (runScalapackTest):
            print("    - ./ci_test_scripts/run_ci_tests.sh -c \" CC=\\\""+c_compiler_wrapper+"\\\"" + " CFLAGS=\\\""+CFLAGS+"\\\"" + " FC=\\\""+fortran_compiler_wrapper+"\\\"" + " FCFLAGS=\\\""+FCFLAGS+"\\\"" \
                + libs + " " + ldflags + " " + " "+ scalapackldflags +" " + scalapackfcflags \
                + " --enable-option-checking=fatal --enable-scalapack-tests" + " " + mpi_configure_flag + " " + openmp[o] \
                + " " + precision[p] + " " + assumed_size[a] + " " + band_to_full_blocking[b] \
                + " " +gpu[g] + INSTRUCTION_OPTIONS + "\" -j 8 -t " + str(MPI_TASKS) + " -m $MATRIX_SIZE -n $NUMBER_OF_EIGENVECTORS -b $BLOCK_SIZE -s $SKIP_STEP -q \"srun\" -S $SLURM")
            

        else:
            print("    - ./ci_test_scripts/run_ci_tests.sh -c \" CC=\\\""+c_compiler_wrapper+"\\\"" + " CFLAGS=\\\""+CFLAGS+"\\\"" + " FC=\\\""+fortran_compiler_wrapper+"\\\"" + " FCFLAGS=\\\""+FCFLAGS+"\\\"" \
                + libs + " " + ldflags + " " + " "+ scalapackldflags +" " + scalapackfcflags \
                + " --enable-option-checking=fatal" + " " + mpi_configure_flag + " " + openmp[o] \
                + " " + precision[p] + " " + assumed_size[a] + " " + band_to_full_blocking[b] \
                + " " +gpu[g] + INSTRUCTION_OPTIONS + "\" -j 8 -t " + str(MPI_TASKS) + " -m $MATRIX_SIZE -n $NUMBER_OF_EIGENVECTORS -b $BLOCK_SIZE -s $SKIP_STEP -q \"srun\" -i $INTERACTIVE_RUN -S $SLURM")

    # do the test

    if ( instr == "avx2" or instr == "avx512" or instr == "knl"  or g == "with-gpu"):
        if (o == "openmp"):
            if (cov == "no-coverage"):
                openmp_threads=" 2 "
            else:
                openmp_threads=" 1 "
        else:
            openmp_threads=" 1 "
        for na in sorted(matrix_size.keys(),reverse=True):
            cores = set_number_of_cores(MPI_TASKS, o)
            #print("    - echo \" srun  --ntasks=1 --cpus-per-task="+str(cores)+" $SRUN_COMMANDLINE_RUN\" ")
            print("    -  echo \"na= $MATRIX_SIZE, nev= $NUMBER_OF_EIGENVECTORS nblock= $BLOCK_SIZE\" ")
            #print("    - srun --ntasks-per-core=1 --ntasks=1 --cpus-per-task="+str(cores)+" $SRUN_COMMANDLINE_RUN \
            #                             /scratch/elpa/bin/run_elpa.sh "+str(MPI_TASKS) + openmp_threads +" \" TEST_FLAGS=\\\" $MATRIX_SIZE $NUMBER_OF_EIGENVECTORS $BLOCK_SIZE " +"\\\"  || { cat test-suite.log; exit 1; }\"")

        if (cov == "coverage"):
            print("    - ./ci_test_scripts/ci_coverage_collect")
            print("  artifacts:")
            print("    paths:")
            print("      - coverage_data")
    print("\n\n")
