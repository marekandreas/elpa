#!/usr/bin/env python3
"""Process cpp output to produce fortran_constants.F90.

Replaces the autotools awk one-liner:
  awk '/!ELPA_C_DEFINE/ {gsub(/!ELPA_C_DEFINE/, "\\n"); gsub(/NEWLINE/, "\\n"); print;}'

Usage: process_fortran_constants.py <input.F90_> <output.F90>
"""

import sys

with open(sys.argv[1]) as f_in, open(sys.argv[2], "w") as f_out:
    for line in f_in:
        if "!ELPA_C_DEFINE" in line:
            line = line.replace("!ELPA_C_DEFINE", "\n")
            line = line.replace("NEWLINE", "\n")
            f_out.write(line)
