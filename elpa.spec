#
# spec file for package elpa
#
# Copyright (c) 2015 Lorenz Huedepohl
#
# All modifications and additions to the file contributed by third parties
# remain the property of their copyright owners, unless otherwise agreed
# upon. The license for this file, and modifications and additions to the
# file, is the same license as for the pristine package itself (unless the
# license for the pristine package is not an Open Source License, in which
# case the license is the MIT License). An "Open Source License" is a
# license that conforms to the Open Source Definition (Version 1.9)
# published by the Open Source Initiative.

%define so_version 8

# OpenMP support requires an MPI implementation with MPI_THREAD_MULTIPLE support,
# which is currently not provided at OpenSUSE
%define with_openmp 0

Name:           elpa
Version:        2024.05.001
Release:        1
Summary:        A massively parallel eigenvector solver
License:        LGPL-3.0
Group:          System/Libraries
Url:            https://elpa.mpcdf.mpg.de/
Source0:        https://elpa.mpcdf.mpg.de/html/Releases/%{version}/%{name}-%{version}.tar.gz
BuildRequires:  c_compiler
BuildRequires:  gcc-fortran >= 4.8
BuildRequires:  openmpi-devel
Requires:       openmpi
BuildRequires:  blas-devel
BuildRequires:  lapack-devel
BuildRequires:  pkg-config
BuildRequires:  autoconf
BuildRequires:  automake
BuildRequires:  libtool

%if %{defined fedora}
BuildRequires:  scalapack-openmpi-devel
BuildRequires:  blacs-openmpi-devel
BuildRequires:  environment-modules
%endif

%if %{defined suse_version}
BuildRequires:  libscalapack2-openmpi-devel
%endif

# For make check, mpirun of openmpi needs an installed openssh
BuildRequires:  openssh
BuildRoot:      %{_tmppath}/%{name}-%{version}-build

%description
A new efficient distributed parallel direct eigenvalue solver for
symmetric matrices. It contains both an improved one-step ScaLAPACK type solver
(ELPA1) and the two-step solver ELPA2.

ELPA uses the same matrix layout as ScaLAPACK. The actual parallel linear
algebra routines are completely rewritten. ELPA1 implements the same linear
algebra as traditional solutions (reduction to tridiagonal form by Householder
transforms, divide & conquer solution, eigenvector backtransform). In ELPA2,
the reduction to tridiagonal form and the corresponding backtransform are
replaced by a two-step version, giving an additional significant performance
improvement.

ELPA has demonstrated good scalability for large matrices on up to 294.000
cores of a BlueGene/P system.

%package     -n lib%{name}%{so_version}
Summary:        A massively parallel eigenvector solver
Group:          System/Libraries
Provides:       %{name} = %{version}
Requires:       %{name}-tools >= %{version}

%description -n lib%{name}%{so_version}
A new efficient distributed parallel direct eigenvalue solver for
symmetric matrices. It contains both an improved one-step ScaLAPACK type solver
(ELPA1) and the two-step solver ELPA2.

ELPA uses the same matrix layout as ScaLAPACK. The actual parallel linear
algebra routines are completely rewritten. ELPA1 implements the same linear
algebra as traditional solutions (reduction to tridiagonal form by Householder
transforms, divide & conquer solution, eigenvector backtransform). In ELPA2,
the reduction to tridiagonal form and the corresponding backtransform are
replaced by a two-step version, giving an additional significant performance
improvement.

ELPA has demonstrated good scalability for large matrices on up to 294.000
cores of a BlueGene/P system.

%package        tools
Summary:        Utility program for %{name}
Group:          Development/Libraries
Requires:       %{name} = %{version}

%description    tools
A small tool program for %{name}, elpa2_print_kernels, which prints the available and
currently selected numerical kernel for ELPA2.

%package        devel
Summary:        Development files for %{name}
Group:          Development/Libraries
Requires:       %{name} = %{version}
Requires:       openmpi
Requires:       lapack-devel
Requires:       blas-devel
Requires:       libscalapack2-openmpi-devel

%description    devel
The %{name}-devel package contains libraries and header files for
developing applications that use %{name}.

%package        devel-static
Summary:        Development files for %{name} - static libraries
Group:          Development/Libraries
Requires:       %{name}-devel

%description    devel-static
This package provides the static libraries for developing applications
that use %{name}.

%if %{defined with_openmp}

%package     -n lib%{name}_openmp%{so_version}
Requires:       openmpi >= 1.8
BuildRequires:  openmpi-devel >= 1.8
Summary:        A massively parallel eigenvector solver
Group:          System/Libraries
Provides:       %{name}_openmp = %{version}
Requires:       %{name}_openmp-tools >= %{version}

%description -n lib%{name}_openmp%{so_version}
OpenMP parallelized version of %{name}, use with an Open MPI implementation
that was configured and tested with MPI_THREAD_MULTIPLE support.

%package     -n %{name}_openmp-tools
Summary:        Utility program for %{name}_openmp
Group:          Development/Libraries
Provides:       %{name}_openmp = %{version}

%description -n %{name}_openmp-tools
A small tool program for %{name}_openmp, elpa2_print_kernels_openmp, which
prints the available and currently selected numerical kernel for ELPA2.

%package     -n %{name}_openmp-devel
Summary:        Development files for %{name}_openmp
Group:          Development/Libraries
Requires:       %{name}_openmp = %{version}
Requires:       openmpi
Requires:       lapack-devel
Requires:       blas-devel
Requires:       libscalapack2-openmpi-devel

%description -n %{name}_openmp-devel
The %{name}_openmp-devel package contains libraries and header files for
developing applications that use %{name}_openmp.

%package     -n %{name}_openmp-devel-static
Summary:        Development files for %{name} - static libraries
Group:          Development/Libraries
Requires:       %{name}-devel

%description -n %{name}_openmp-devel-static
This package provides the static libraries for developing applications
that use %{name}_openmp.
%endif

%prep
%setup

%build
%if %{defined fedora}
module load mpi/openmpi-%{_arch}
%endif
%if %{defined suse_version}
. %{_libdir}/mpi/gcc/openmpi/bin/mpivars.sh
%endif

if [ ! -e configure ] ; then
        # It is possible to use the Open Build Server to automatically
        # checkout from git directly, extract this spec file and set the
        # version of the extracted spec file to some autogenerated
        # string based on the date and git hash of the checkout commit.
        # This line ensures that the spec file in the tar-ball has
        # that same version, so that autogen will insert that version
        # in the configure script
        perl -pi -e "s/^Version:.*/Version: %{version}/" elpa.spec

        ./autogen.sh
fi

# Normal build
mkdir build
pushd build
%define _configure ../configure

%configure \
%ifarch i386 i486 i586 i686 x86_64
        CFLAGS="$CFLAGS -msse4.2" \
        FCFLAGS="$FFLAGS $FCFLAGS -msse4.2" \
%endif
%ifnarch i386 i486 i586 i686 x86_64
        --disable-sse \
        --disable-sse-assembly \
%endif
        --disable-avx \
        --disable-avx2 \
        --docdir=%{_docdir}/%{name}-%{version} \
        || { cat config.log; exit 1; }

make %{?_smp_mflags} V=1
popd


# OpenMP build
%if %{defined with_openmp}
mkdir build_openmp
pushd build_openmp

%configure \
%ifarch i386 i486 i586 i686 x86_64
        CFLAGS="$CFLAGS -msse4.2" \
        FCFLAGS="$FFLAGS $FCFLAGS -msse4.2" \
%endif
%ifnarch i386 i486 i586 i686 x86_64
        --disable-sse \
        --disable-sse-assembly \
%endif
        --disable-avx \
        --disable-avx2 \
        --docdir=%{_docdir}/%{name}_openmp-%{version} \
        --enable-openmp \
        || { cat config.log; exit 1; }

make %{?_smp_mflags} V=1
popd

%endif # OpenMP


%check
%if %{defined fedora}
module load mpi/openmpi-%{_arch}
%endif
%if %{defined suse_version}
. %{_libdir}/mpi/gcc/openmpi/bin/mpivars.sh
%endif

pushd build
make check TEST_FLAGS="1500 50 16" || { echo "Tests failed: Content of ./test-suite.log:"; cat ./test-suite.log; echo; exit 1; }
popd

%if %{defined with_openmp}
pushd build_openmp
make check TEST_FLAGS="1500 50 16" || { echo "Tests failed: Content of ./test-suite.log:"; cat ./test-suite.log; echo; exit 1; }
popd
%endif


%install
%if %{defined with_openmp}
pushd build_openmp
make V=1 install DESTDIR=%{buildroot}
popd
%endif
pushd build
make V=1 install DESTDIR=%{buildroot}
popd

%post   -n lib%{name}%{so_version} -p /sbin/ldconfig
%postun -n lib%{name}%{so_version} -p /sbin/ldconfig

%if %{defined with_openmp}
%post   -n lib%{name}_openmp%{so_version} -p /sbin/ldconfig
%postun -n lib%{name}_openmp%{so_version} -p /sbin/ldconfig
%endif

%files -n lib%{name}%{so_version}
# See http://en.opensuse.org/openSUSE:Shared_library_packaging_policy
# to explain this package's name
%defattr(-,root,root)
%{_libdir}/lib%{name}.so.*
%{_docdir}/%{name}-%{version}/*
%dir %{_docdir}/%{name}-%{version}

%files tools
%attr(0755,root,root) %{_bindir}/elpa2_print_kernels
%attr(0644,root,root) %{_mandir}/man1/elpa2_print_kernels.1.gz

%files devel
%defattr(-,root,root)
%{_libdir}/pkgconfig/%{name}-%{version}.pc
%{_includedir}/%{name}-%{version}
%{_libdir}/lib%{name}.so
%{_libdir}/lib%{name}.la
%_mandir/man3/*

%files devel-static
%defattr(-,root,root)
%{_libdir}/lib%{name}.a

%if %{defined with_openmp}

%files -n lib%{name}_openmp%{so_version}
%defattr(-,root,root)
%{_libdir}/lib%{name}_openmp.so.*
%{_docdir}/%{name}_openmp-%{version}/*
%dir %{_docdir}/%{name}_openmp-%{version}

%files -n %{name}_openmp-tools
%defattr(-,root,root)
%attr(0755,root,root) %{_bindir}/elpa2_print_kernels_openmp


%files -n %{name}_openmp-devel
%defattr(-,root,root)
%{_libdir}/pkgconfig/%{name}_openmp-%{version}.pc
%{_includedir}/%{name}_openmp-%{version}
%{_libdir}/lib%{name}_openmp.so
%{_libdir}/lib%{name}_openmp.la

%files -n %{name}_openmp-devel-static
%defattr(-,root,root)
%{_libdir}/lib%{name}_openmp.a

%endif

%changelog
