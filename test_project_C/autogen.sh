#!/bin/sh -e

mkdir -p m4/

test -n "$srcdir" || srcdir=`dirname "$0"`
test -n "$srcdir" || srcdir=.

autoreconf --force --install --verbose "$srcdir"
