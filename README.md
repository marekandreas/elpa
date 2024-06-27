# [Eigenvalue SoLvers for Petaflop-Applications (ELPA)](http://elpa.mpcdf.mpg.de)

## Current Release ##

The current release is ELPA 2024.05.001. The current supported API version
is 20241105. This release supports the earliest API version 20170403.
The current version for autotuning is also 20241105 and down to version 20170403 ist supported
for autotuning. When the autotune version is set to a value **below** 20211125 the old autotunig
implementation is used, and for 20211125 and higher the new implementation is used.

[![Build 
status](https://gitlab.mpcdf.mpg.de/elpa/elpa/badges/master/build.svg)](https://gitlab.mpcdf.mpg.de/elpa/elpa/commits/master)

[![Code 
coverage](https://gitlab.mpcdf.mpg.de/elpa/badges/master/coverage.svg)](https://elpa.pages.mpcdf.de/elpa/coverage_summary)

![License LGPL v3][license-badge]

[license-badge]: https://img.shields.io/badge/License-LGPL%20v3-blue.svg


## About *ELPA* ##

The computation of selected or all eigenvalues and eigenvectors of a symmetric
(Hermitian) matrix has high relevance for various scientific disciplines.
For the calculation of a significant part of the eigensystem typically direct
eigensolvers are used. For large problems, the eigensystem calculations with
existing solvers can become the computational bottleneck.

As a consequence, the *ELPA* project was initiated with the aim to develop and
implement an efficient eigenvalue solver for petaflop applications, supported
by the German Federal Government, through BMBF Grant 01IH08007, from
Dec 2008 to Nov 2011.

The challenging task has been addressed through a multi-disciplinary consortium
of partners with complementary skills in different areas.

The *ELPA* library was originally created by the *ELPA* consortium,
consisting of the following organizations:

- Max Planck Computing and Data Facility (MPCDF), fomerly known as
  Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
- Bergische Universität Wuppertal, Lehrstuhl für angewandte
  Informatik,
- Technische Universität München, Lehrstuhl für Informatik mit
  Schwerpunkt Wissenschaftliches Rechnen ,
- Fritz-Haber-Institut, Berlin, Abt. Theorie,
- Max-Plack-Institut für Mathematik in den Naturwissenschaften,
  Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
  and
- IBM Deutschland GmbH

*ELPA* is distributed under the terms of version 3 of the license of the
GNU Lesser General Public License as published by the Free Software Foundation.

## Obtaining *ELPA*

There exist several ways to obtain the *ELPA* library either as sources or pre-compiled packages:

- official release tar-gz sources from the *ELPA* [webpage](https://elpa.mpcdf.mpg.de/software/tarball-archive/ELPA_TARBALL_ARCHIVE.html)
- from the *ELPA* [git repository](https://gitlab.mpcdf.mpg.de/elpa/elpa)
- as packaged software for several Linux distributions (e.g. Debian, Fedora, OpenSuse)

## Terms of usage

Your are free to obtain and use the *ELPA* library, as long as you respect the terms
of version 3 of the license of the GNU Lesser General Public License.

No other conditions have to be met.

Nonetheless, we are grateful if you cite the following publications:

  If you use ELPA in general:

  T. Auckenthaler, V. Blum, H.-J. Bungartz, T. Huckle, R. Johanni,
  L. Krämer, B. Lang, H. Lederer, and P. R. Willems,
  "Parallel solution of partial symmetric eigenvalue problems from
  electronic structure calculations",
  Parallel Computing 37, 783-794 (2011).
  doi:10.1016/j.parco.2011.05.002.

  Marek, A.; Blum, V.; Johanni, R.; Havu, V.; Lang, B.; Auckenthaler,
  T.; Heinecke, A.; Bungartz, H.-J.; Lederer, H.
  "The ELPA library: scalable parallel eigenvalue solutions for electronic
  structure theory and computational science",
  Journal of Physics Condensed Matter 26, 213201 (2014)
  doi:10.1088/0953-8984/26/21/213201
  
  If you use the GPU version of ELPA:

  Kus, P; Marek, A.; Lederer, H.
  "GPU Optimization of Large-Scale Eigenvalue Solver",
  In: Radu F., Kumar K., Berre I., Nordbotten J., Pop I. (eds) 
  Numerical Mathematics and Advanced Applications ENUMATH 2017. ENUMATH 2017. 
  Lecture Notes in Computational Science and Engineering, vol 126. Springer, Cham
  
  Yu, V.; Moussa, J.; Kus, P.; Marek, A.; Messmer, P.; Yoon, M.; Lederer, H.; Blum, V.
  "GPU-Acceleration of the ELPA2 Distributed Eigensolver for Dense Symmetric and Hermitian Eigenproblems",
  Computer Physics Communications 262, 107808 (2021)
  doi:10.1016/j.cpc.2020.107808

  If you use the new API and/or autotuning:
 
  Kus, P.; Marek, A.; Koecher, S. S.; Kowalski H.-H.; Carbogno, Ch.; Scheurer, Ch.; Reuter, K.; Scheffler, M.; Lederer, H.
  "Optimizations of the Eigenvaluesolvers in the ELPA Library",
  Parallel Computing 85, 167-177 (2019)

  If you use the new support for skew-symmetric matrices:

  Penke C.; Marek, A.; Vorwerk, C.; Draxl, C.; Benner, P.;
  "High Performance Solution of Skew-symmetric Eigenvalue Problems with Applications in Solving the Bethe-Salpeter Eigenvalue Problem",
  Parallel Computing 96, 102639 (2020)
  doi:10.1016/j.parco.2020.102639

## Installation of the *ELPA* library

*ELPA* is shipped with a standard autotools automake installation infrastructure.
Some other libraries are needed to install *ELPA* (the details depend on how you
configure *ELPA*):

  - Basic Linear Algebra Subroutines (BLAS)
  - Lapack routines
  - Basic Linear Algebra Communication Subroutines (BLACS)
  - Scalapack routines
  - a working MPI library

Please refer to the [INSTALL document](./documentation/INSTALL.md) on details of the installation process and
the possible configure options.

## Using *ELPA*

Please have a look at the [USERS_GUIDE](./documentation/USERS_GUIDE.md) file, to get a documentation or at the [online](https://elpa.mpcdf.mpg.de/documentation/doxygen/ELPA_DOXYGEN_PAGES/ELPA-2024.05.001/html/index.html) doxygen documentation, where you find the definition of the interfaces. You might want to have a look at the [PERFORMANCE tuning document](./documentation/PERFORMANCE_TUNING.md) to avoid some usual pitfalls.

## Contributing to *ELPA*

It has been, and is, a tremendous effort to develop and maintain the
*ELPA* library. A lot of things can still be done, but our man-power is limited.

Thus every effort and help to improve the *ELPA* library is highly appreciated.
For details please see the [CONTRIBUTING](CONTRIBUTING.md) document.


