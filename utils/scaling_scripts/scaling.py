import os
import sys

def substitute(template_file, tokens):
    with open("run.sh", "w") as fout:
        with open(template_file, "r") as fin:
            for line in fin:
                for token in tokens.keys():
                    line = line.replace(token, str(tokens[token]))
                fout.write(line)

def variant_path(output_dir, tokens, elpa_method, math_type, precision):
    return "/".join([output_dir, math_type, precision,
                                       str(tokens['_MAT_SIZE_']),
                                       str(tokens['_NUM_EIGEN_']),
                                       elpa_method])


def variant(output_dir, template_file, tokens, elpa_method, math_type, precision):
    typeprec = math_type + "_" + precision
    tokens['_PRE_RUN_'] = ''
    if(elpa_method == 'elpa1'):
        tokens['_EXECUTABLE_'] = "test_" + typeprec + "_eigenvectors_1stage_analytic"
    elif(elpa_method == 'elpa2'):
        tokens['_PRE_RUN_'] = 'TEST_KERNEL="ELPA_2STAGE_REAL_AVX_BLOCK2"'
        tokens['_EXECUTABLE_'] = "test_" + typeprec + "_eigenvectors_2stage_default_kernel_analytic"
    elif(elpa_method == 'scalapack_all'):
        tokens['_EXECUTABLE_'] = "test_" + typeprec + "_eigenvectors_scalapack_all_analytic"
    elif(elpa_method == 'scalapack_part'):
        tokens['_EXECUTABLE_'] = "test_" + typeprec + "_eigenvectors_scalapack_part_analytic"
    else:
        assert(0)


    tokens['_OUTPUT_DIR_'] = variant_path(output_dir, tokens, elpa_method, math_type, precision)
    if not os.path.exists(tokens['_OUTPUT_DIR_']):
        os.makedirs(tokens['_OUTPUT_DIR_'])

    substitute(template_file, tokens)
    if(len(sys.argv) == 2 and sys.argv[1] == '--submit'):
        os.system('llsubmit run.sh')
    else:
        os.system('cat run.sh')
