#!/usr/bin/env python3
from itertools import product
from scaling import *

output_dir = "out"
template_file = "run_template_hydra.sh"

#elpa_method = ['elpa1', 'elpa2']
elpa_method = ['elpa1', 'elpa2', 'scalapack_all', 'scalapack_part']
#elpa_method = ['scalapack_part']
math_type = ['real', 'complex']
precision = ['single', 'double']
mat_size = [5000, 20000]
proc_eigen = [10,50,100]
block_size = [16]

num_nodes = [1]
#num_nodes.extend([2**i for i in range(2,11)])
num_nodes.extend([2**i for i in range(2,7)])

#num_nodes = [2048]


#===============================================================================================
#===============================================================================================
# the rest of the script should be changed only if something changed (etc. in elpa)
#===============================================================================================
#===============================================================================================


for em, mt, pr, ms, pe, bs, nn in product(elpa_method, math_type, precision, mat_size, proc_eigen, block_size, num_nodes):
    tokens = {}
    tokens['_BLOCK_SIZE_'] = bs
    tokens['_MAT_SIZE_'] = ms·
    tokens['_NUM_EIGEN_'] = ms * pe // 100
    tokens['_NUM_NODES_'] = nn
    variant(output_dir, template_file, tokens, em, mt, pr)
