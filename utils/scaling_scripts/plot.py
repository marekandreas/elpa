#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from itertools import product

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
results_filename = "results/2017_08/results_sorted.txt"
cores_per_node = 20
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------


print("PLOTING ...")
group_colors = [['red', 'firebrick', 'indianred', 'tomato', 'maroon', 'salmon'],
          ['green', 'darkgreen', 'springgreen', 'darkseagreen', 'lawngreen', 'yellowgreen'],
          ['blue', 'darkblue', 'cornflowerblue', 'dodgerblue', 'midnightblue', 'lightskyblue'],
          ['magenta', 'darkviolet', 'mediumvioletred', 'orchid', 'deeppink', 'purple'],
          ['orange', 'gold', 'navajowhite', 'darkorange', 'goldenrod', 'sandybrown'],
          ['cyan', 'darkcyan', 'lightseagreen', 'turquoise', 'darkturquoise', 'mediumturquoise']]
group_symbols = ['o', 's', '*', 'D', 'x', 'H']
elpa1_subtimes = ["tridiag", "solve", "trans_ev"]
elpa2_subtimes = ["bandred", "tridiag", "solve", "trans_ev_to_band", "trans_ev_to_full"]

def get_from_sorted(filename, mathtype, precision, na, nev, method):
    if(method == "pdsyevd"):
        method = "scalapack_all"
    if(method == "pdsyevr"):
        method = "scalapack_part"
    what = " ".join([mathtype, precision, str(na), str(nev), method])
    regex = r"(.|\n)*" + what + r"( |\n)*-+\n(?P<section>[a-zA-Z_0-9\.\n ]*)\n-+(.|\n)*"
    p = re.compile(regex)
    with open(filename, "r") as f:
        m = p.match(f.read())
    section = m.groupdict()['section']
    lines = section.split("\n")
    ll = [l.split() for l in lines]
    oposite = [[ll[x][y] for x in range(len(ll))] for y in range(len(ll[0]))]
    return {oposite[i][0]:oposite[i][1:] for i in range(len(oposite))}

def scalapack_name(num_type, pr, all_ev):
    if(num_type == "real"):
        if(pr == "single"):
            name = "pssyev"
        else:
            name = "pdsyev"
    else:
        if(pr == "single"):
            name = "pcheev"
        else:
            name = "pzheev"
    if(all_ev):
        name += "d"
    else:
        name += "r"
    return name


def line(what, num_type, prec, mat_size, proc_evec, method, label, color, style):
    data = get_from_sorted(results_filename, num_type, prec, mat_size, mat_size*proc_evec//100, method)
    nodes = data['nodes']
    cores = [cores_per_node * int(n) for n in nodes]
    data_line = data[what]
    plt.plot(cores,data_line, style, label=label, color=color, linewidth=2)

def plot1(num_type, prec, mat_size):
    line("total", num_type, prec, mat_size, 100, "pdsyevd", "MKL 2017, " + scalapack_name(num_type, prec, True), "black", "x-")

    line("total", num_type, prec, mat_size, 100, "pdsyevr", "MKL 2017, " + scalapack_name(num_type, prec, True) + ", 100% EVs", "blue", "x-")
    line("total", num_type, prec, mat_size, 50, "pdsyevr", "MKL 2017, " + scalapack_name(num_type, prec, True) + ", 50% EVs", "green", "x-")
    line("total", num_type, prec, mat_size, 10, "pdsyevr", "MKL 2017, " + scalapack_name(num_type, prec, True) + ", 10% EVs", "red", "x-")

    line("total", num_type, prec, mat_size, 100, "elpa1", "ELPA 1, 100% EVs", "blue", "*--")
    line("total", num_type, prec, mat_size, 50, "elpa1", "ELPA 1, 50% EVs", "green", "*--")
    line("total", num_type, prec, mat_size, 10, "elpa1", "ELPA 1, 10% EVs", "red", "*--")

    line("total", num_type, prec, mat_size, 100, "elpa2", "ELPA 2, 100% EVs", "blue", "o:")
    line("total", num_type, prec, mat_size, 50, "elpa2", "ELPA 2, 50% EVs", "green", "o:")
    line("total", num_type, prec, mat_size, 10, "elpa2", "ELPA 2, 10% EVs", "red", "o:")

def details(num_type, prec, mat_size, proc_ev):
    for i in range(len(elpa1_subtimes)):
        line(elpa1_subtimes[i], num_type, prec, mat_size, proc_ev, "elpa1", "ELPA1 - " + elpa1_subtimes[i], group_colors[0][i], group_symbols[2*i] + '-') 

    for i in range(len(elpa2_subtimes)):
        line(elpa2_subtimes[i], num_type, prec, mat_size, proc_ev, "elpa2", "ELPA2 - " + elpa2_subtimes[i], group_colors[1][i], group_symbols[i] + '-') 


def plot(num_type, prec, mat_size, filename):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(labelright='on')

    plot1(num_type, prec, mat_size)
    #details(num_type, prec, mat_size, 100)

    #plt.title('Num CPUs ' + str(num_cpus) + ' and ' + str(eigenvectors_percent) + '% eigenvectors, ' + numtype)
    #plt.title('Num CPUs ')
    plt.title("Matrix " + str(mat_size//1000) + "k, " + num_type + ", " + prec)
    plt.grid()
    plt.legend(loc=1)
    plt.xlabel('Number of cores')
    plt.ylabel('Execution time [s]')
    plt.xscale('log')
    plt.yscale('log')
    ax.xaxis.grid(b=True, which='major', color='black', linestyle=':')
    ax.yaxis.grid(b=True, which='major', color='black', linestyle='--')
    ax.yaxis.grid(b=True, which='minor', color='black', linestyle=':')
    ticks = [20* 2**i for i in range(0,12)]
    ax.xaxis.set_ticks(ticks)
    ax.xaxis.set_ticklabels(ticks)

    if(mat_size < 10000):
        y_max = 50
        y_min = 0.3
    else:
        y_max = 500
        y_min = 3

    yticks_major = [1,10,100,1000, y_min, y_max]
    ax.yaxis.set_ticks(yticks_major)
    ax.yaxis.set_ticklabels(yticks_major)
    #    yticks_minor = [2, 5, 20, 50, 200, 500]
    #    ax.yaxis.set_ticks(yticks_minor, minor=True)
    #    ax.yaxis.set_ticklabels(yticks_minor, minor=True)
    plt.ylim([y_min, y_max])
    plt.xlim([20, 41000])
    plt.savefig(filename)
    #if show:
    plt.show()
    #plt.close()

#plot("double", "real", 20000, 'plot.pdf')
for n, p, m in product(['real', 'complex'], ['double', 'single'], [5000, 20000]):
    name = "_".join(["plot", n, p, str(m)]) + ".pdf"
    plot(n, p, m, name)
