# ELPA — CMake Build Examples

Example configure, build, and test scripts for the CMake build system.
The CMake build is an alternative to the existing autotools setup; `configure.ac`
and the `Makefile.am` files are left untouched.

## Quick start

### Linux

```bash
# 1. Audit toolchain components (shows FOUND / MISSING).
./cmake_build_examples/linux/check-prerequisites.sh

# 2. Edit USER-EDITABLE PATHS in the chosen configure script.
#    For Intel toolchains, source oneAPI first:
source /opt/intel/oneapi/setvars.sh --force

# 3. Configure, build, test.
./cmake_build_examples/linux/configure-gcc-gfortran-openblas-openmpi.sh
cmake --build <BLD> -j$(nproc)
cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all
```

### macOS

```bash
brew install cmake gcc openmpi openblas scalapack libomp pkgconf
./cmake_build_examples/macos/configure-gcc-gfortran-openblas-openmpi-macos-arm64.sh
cmake --build <BLD> -j$(sysctl -n hw.ncpu)
cd <BLD> && ctest -j$(sysctl -n hw.ncpu) --timeout 300 --output-on-failure
```

### Windows

```powershell
.\cmake_build_examples\windows\configure-clang-ifx.ps1 -EnableCUDA
.\cmake_build_examples\windows\cmake_build.ps1 -BuildDir C:\path\to\build
.\cmake_build_examples\windows\test.ps1 -AllIncludingExtended -BuildDir C:\path\to\build
```

## Validated configurations

### Linux x86\_64

| Script | C / C++ | Fortran | OpenMP | MPI | Math libraries | CUDA |
|---|---|---|---|---|---|---|
| `configure-gcc-gfortran-openblas-openmpi.sh` | gcc | gfortran | gomp | OpenMPI | OpenBLAS + ScaLAPACK | ON |
| `configure-gcc-gfortran-mkl-impi.sh` | gcc | gfortran | gomp | Intel MPI | MKL gnu\_thread | ON |
| `configure-gcc-gfortran-mkl-openmpi.sh` | gcc | gfortran | gomp | OpenMPI | MKL gnu\_thread | ON |
| `configure-gcc-ifx-mkl-impi.sh` | gcc | ifx | iomp5 | Intel MPI | MKL intel\_thread | ON |
| `configure-gcc-ifort-mkl-impi.sh` | gcc | ifort | iomp5 | Intel MPI | MKL intel\_thread | ON |
| `configure-clang-flang-openblas-openmpi.sh` | clang | flang-new | libomp | OpenMPI¹ | OpenBLAS + ScaLAPACK | ON |
| `configure-clang-flang-mkl-openmpi.sh` | clang | flang-new | libomp | OpenMPI¹ | MKL intel\_thread | ON |
| `configure-clang-flang-blis-openmpi.sh` | clang | flang-new | libomp | OpenMPI¹ | AOCL BLIS + FLAME + ScaLAPACK | ON |
| `configure-clang-ifx-mkl-impi.sh` | clang | ifx | iomp5 | Intel MPI | MKL intel\_thread | ON |
| `configure-icx-ifx-mkl-impi.sh` | icx / icpx | ifx | iomp5 | Intel MPI | MKL intel\_thread | ON |
| `configure-gcc-gfortran-blis-openmpi.sh` | gcc | gfortran | gomp | OpenMPI | AOCL BLIS + FLAME + ScaLAPACK | ON |
| `configure-aocc-blis-openmpi.sh` | AOCC clang | AOCC flang | libomp | OpenMPI | AOCL BLIS + FLAME + ScaLAPACK | OFF |

¹ Requires a separate OpenMPI built with the same clang/flang-new; system
OpenMPI Fortran modules are gfortran-ABI and incompatible with flang-new.
Set `OMPI_HOME` in the script to the custom installation prefix.

CUDA-enabled configurations use GCC as the `nvcc` host compiler when the
project C compiler is not GCC (clang, icx).

### Linux AArch64

| Script | C / C++ | Fortran | MPI | Math libraries | CUDA |
|---|---|---|---|---|---|
| `configure-gcc-gfortran-openblas-openmpi-aarch64.sh` | gcc | gfortran | OpenMPI | OpenBLAS + ScaLAPACK | OFF |
| `configure-clang-flang-openblas-openmpi-aarch64.sh` | clang | flang-new | OpenMPI¹ | OpenBLAS + ScaLAPACK | OFF |

Validated on OCI A1 (Ampere Altra, 4 oCPUs, Ubuntu 24.04 LTS).
NEON AArch64 kernel families (BLOCK2, BLOCK4, BLOCK6) are selected
automatically; x86 kernel families are excluded on non-x86 hosts.

### macOS (Apple Silicon)

| Script | C / C++ | Fortran | MPI | Math libraries | CUDA |
|---|---|---|---|---|---|
| `configure-gcc-gfortran-openblas-openmpi-macos-arm64.sh` | gcc | gfortran | OpenMPI | OpenBLAS + ScaLAPACK | OFF |
| `configure-gcc-gfortran-accelerate-scalapack-openmpi-macos-arm64.sh` | gcc | gfortran | OpenMPI | Accelerate + ScaLAPACK² | OFF |

² Accelerate is the system BLAS/LAPACK; ScaLAPACK must be built separately
from source against Accelerate (Homebrew ScaLAPACK links OpenBLAS).

Validated on Apple MacBook Air M4 (10 cores, 24 GB), macOS Sequoia 15.5.
All 424 tests passing per configuration.

### Windows

| Script | C / C++ | Fortran | MPI | Math libraries | CUDA |
|---|---|---|---|---|---|
| `configure-clang-ifx.ps1` | clang-cl | ifx | Intel MPI | MKL | ON |
| `configure-clang-ifort.ps1` | clang-cl | ifort | Intel MPI | MKL | ON |

Validated on Intel i7-11700K + NVIDIA RTX 3080 (SM 86), Windows 11.
795 tests passing (607 default + 188 extended).
When CUDA is enabled, `nvcc` uses MSVC `cl.exe` as its host compiler.

### Tool versions

| Component | Linux x86\_64 | Linux AArch64 | macOS | Windows |
|---|---|---|---|---|
| OS | Ubuntu 24.04.3 LTS | Ubuntu 24.04 LTS | macOS Sequoia 15.5 | Windows 11 |
| CMake | 4.3.1 | 4.3.1 | 4.3.1 | 4.2.3 (VS) |
| Ninja | 1.11.1 | 1.11.1 | — | 1.12.1 (VS) |
| GCC / GFortran | 13.3.0 | 13.3.0 | 15.2.0 (Homebrew) | — |
| Clang / Flang (LLVM) | 21.1.8 | 21.1.8 | — | 20.1.8 (clang-cl, VS) |
| Intel oneAPI | 2025.3 | — | — | 2025.3 |
| CUDA Toolkit | 13.2 | — | — | 13.2.78 |
| OpenMPI (system) | 4.1.6 | 4.1.6 | 5.0.9 (Homebrew) | — |
| OpenMPI (clang-built) | 5.0.8 | — | — | — |
| OpenBLAS | 0.3.26 | 0.3.26 | 0.3.32 (Homebrew) | — |
| AOCL | 5.2.0 | — | — | — |
| AOCC | 5.1.0 | — | — | — |

## Design decisions

**Mixed-compiler OpenMP handling.**  Configurations like gcc + ifx combine
`libgomp` (GCC) with `libiomp5` (Intel).  `ElpaOpenMP.cmake` detects mixed
runtimes, compiles C/C++ with `-fopenmp`, and replaces the GCC runtime on the
link line with the explicit `libiomp5` path so only one OpenMP runtime is
loaded.  The `-qopenmp` flag from the Fortran side is stripped from CUDA host
compilation to prevent it leaking to `nvcc`.

**Intel Fortran runtime propagation.**  Both ifx and classic ifort ship
runtime libraries (`libifport`, `libifcoremt`, etc.) that test executables
linked by a non-Intel C compiler need on the search path.
`ElpaCompilerOptions.cmake` injects `-L` via `add_link_options()` (not
`link_directories()`) so the path propagates to all targets regardless of
directory scope.

**MPI C++ bindings suppressed.**  ELPA has no C++ MPI calls.  Linking
`MPI::MPI_CXX` can pull in a system `libmpi_cxx.so` that is ABI-incompatible
with a custom-built MPI.  Test targets instead define `OMPI_SKIP_MPICXX` /
`MPICH_SKIP_MPICXX` to suppress the deprecated bindings.

**MKL SDL not supported.**  `MKLConfig.cmake` in oneAPI 2025+ rejects the SDL
link mode (`libmkl_rt.so`) when cluster libraries (ScaLAPACK/BLACS) are
requested.  Since ELPA requires ScaLAPACK, SDL cannot be used.

**Framework ISA versus default kernels.**  `ELPA_FRAMEWORK_ISA` sets the ISA
baseline for non-kernel library code independently of the `ELPA_ENABLE_*_KERNELS`
options.  This allows a library that is safe to load on AVX2 hosts while still
containing AVX-512 kernels selectable at runtime:

```bash
cmake -S . -B build \
    -DELPA_FRAMEWORK_ISA=avx2 \
    -DELPA_ENABLE_AVX512_KERNELS=ON \
    -DELPA_DEFAULT_REAL_KERNEL=real_avx2_block2 \
    -DELPA_DEFAULT_COMPLEX_KERNEL=complex_avx2_block1
```

**CTest labels.**  Every test is labelled `single_precision` or
`double_precision`, plus a variant label (`default`, `extended`, `autotune`).
This enables selective runs such as `ctest -L double_precision` or
`ctest -LE single_precision`.

## CMake options reference

| Variable | Default | Description |
|---|---|---|
| `ELPA_USE_MKL` | `ON` | ON = Intel MKL; OFF = generic BLAS/LAPACK/ScaLAPACK |
| `ELPA_OPENMP` | `OFF` | Enable OpenMP threading |
| `ELPA_CUDA` | `OFF` | Enable NVIDIA CUDA kernels |
| `ELPA_CUDA_ARCHITECTURES` | `native` | Auto-detect host GPU, or explicit SM list (`75;80;90`) |
| `ELPA_TEST_EXTENDED` | `OFF` | Register extended tests in CTest |
| `ELPA_TEST_AUTOTUNE` | `OFF` | Register autotune tests in CTest |
| `ELPA_FRAMEWORK_ISA` | `native` | Baseline ISA for non-kernel code (`native`, `avx2`, `avx512`) |
| `ELPA_ENABLE_AVX512_KERNELS` | `ON` on x86 | Build AVX-512 kernel families |
| `ELPA_DEFAULT_REAL_KERNEL` | auto | Highest compiled ISA if empty |
| `ELPA_DEFAULT_COMPLEX_KERNEL` | auto | Highest compiled ISA if empty |

## Main CMake files

| Path | Purpose |
|---|---|
| `CMakeLists.txt` | top-level project setup |
| `cmake/ElpaConfig.cmake` | dependency wiring |
| `cmake/dependencies/` | MPI, OpenMP, BLAS/LAPACK/MKL, CUDA detection |
| `cmake/ElpaKernels.cmake` | kernel-family enablement and default-kernel selection |
| `cmake/ElpaCompilerOptions.cmake` | framework ISA flags and compiler-specific options |
| `cmake/ElpaGeneratedFiles.cmake` | generated headers and extracted interfaces |
| `cmake/python/` | helper scripts for configure/build |
| `src/CMakeLists.txt` | library target |
| `test/CMakeLists.txt` | CTest registration |

## Scope and limitations

- Intended as a starting point, not a claim that every autotools workflow is
  replicated.
- Only native builds are supported; `try_run` probes (MPI thread level,
  `!$omp masked`) prevent cross-compilation.
- Windows requires `clang-cl` for C/C++ because ELPA's C sources use C99
  `_Complex`.
- AOCC classic `flang` cannot build the CUDA path; the AOCC example disables
  CUDA.
- Code paths for ROCm, SYCL, and additional non-x86 kernel families are wired
  up from the existing sources but were not validated end-to-end.
- On Linux, exported-symbol behavior matches autotools.  Only Windows uses the
  generated `.def` export list.
