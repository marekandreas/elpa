#!/bin/sh -e

mkdir -p m4/

test -n "$srcdir" || srcdir=`dirname "$0"`
test -n "$srcdir" || srcdir=.

$srcdir/generate_automake_test_programs.py > $srcdir/test_programs.am
autoreconf --force --install --verbose "$srcdir"
