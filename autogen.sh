#!/bin/sh -e

mkdir -p m4/

test -n "$srcdir" || srcdir=`dirname "$0"`
test -n "$srcdir" || srcdir=.

$srcdir/generate_automake_Fortran_test_programs.py > $srcdir/Fortran_test_programs.am
$srcdir/generate_automake_C_test_programs.py > $srcdir/C_test_programs.am
autoreconf --force --install --verbose "$srcdir"
