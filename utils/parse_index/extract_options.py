#    Copyright 2022, Soheil Soltani, MPCDF
#
#    This file is part of ELPA.
#
#    The ELPA library was originally created by the ELPA consortium,
#    consisting of the following organizations:
#
#    - Max Planck Computing and Data Facility (MPCDF), formerly known as
#      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
#    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
#      Informatik,
#    - Technische Universität München, Lehrstuhl für Informatik mit
#      Schwerpunkt Wissenschaftliches Rechnen ,
#    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
#    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
#      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
#      and
#    - IBM Deutschland GmbH
#
#    More information can be found here:
#    http://elpa.mpcdf.mpg.de/
#
#    ELPA is free software: you can redistribute it and/or modify
#    it under the terms of the version 3 of the license of the
#    GNU Lesser General Public License as published by the Free
#    Software Foundation.
#
#    ELPA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public 
#    License along with ELPA. If not, see <http://www.gnu.org/licenses/>
#
#    ELPA reflects a substantial effort on the part of the original
#    ELPA consortium, and we ask you to respect the spirit of the
#    license that we chose: i.e., please contribute any changes you
#    may have back to the original ELPA library distribution, and keep
#    any derivatives of ELPA under the same license that we chose for
#    the original distribution, the GNU Lesser General Public License.


import re


pattern = re.compile(r'\s*.+ENTRY\((".+"),\s*(".+")')
pattern_comment = re.compile(r'(.*)(?=//).+ENTRY\((".+"),\s*(".+")')

header_length = 80
max_line_len  = 70

print(header_length *'-')
opt_count = 0

with open('../../src/elpa_index.c', 'r') as source:
    content = source.readlines()
    for line in content:
        comments = pattern_comment.finditer(line)
        if any(comments) is False:   # this line is not out-commented           
            matches = pattern.finditer(line)
            for match in matches:
                opt_count += 1
                print(f'===> Option {opt_count}: ', match.group(1))
                desc = match.group(2).strip('"')
                if len(desc) > max_line_len:
                    desc_1 = desc[:max_line_len]
                    desc_2 = desc[max_line_len:]
                    print('\t', desc_1)
                    print('\t', desc_2, '\n')
                else:
                    print('\t', desc, '\n')

                print(header_length *'-')

