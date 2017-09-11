#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os

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

cores_per_node = 20
base_paths = ["results", "results2"]
num_type = "real"
prec = "double"
mat_size = 5000

def scalapack_name(num, pr, all_ev):
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


def line(what, mat_size, proc_evec, method, label, color, style):
    data_line_res = []
    nodes_res = []
    for base_path in base_paths:
        path = "/".join([base_path,num_type,prec,str(mat_size),str(mat_size*proc_evec//100),method,"tab.txt"])
        #print(path)
        if not os.path.isfile(path):
            continue
        data = np.genfromtxt(path, names=True)
        nodes = data['nodes']
        data_line = data[what]
        #print("data_line", data_line, "data_line_res", data_line_res)
        if(nodes_res == []):
            assert(data_line_res == [])
            nodes_res = nodes
            data_line_res = data_line
        else:
            assert(all(nodes == nodes_res))
            data_line_res = np.minimum(data_line_res, data_line)

    cores = cores_per_node * nodes_res
    #print(cores, data_line_res)
    plt.plot(cores,data_line_res, style, label=label, color=color, linewidth=2)

def plot1():
    line("total", mat_size, 100, "pdsyevd", "MKL 2017, " + scalapack_name(num_type, prec, True), "black", "x-")

    line("total", mat_size, 100, "pdsyevr", "MKL 2017, " + scalapack_name(num_type, prec, True) + ", 100% EVs", "blue", "x-")
    line("total", mat_size, 50, "pdsyevr", "MKL 2017, " + scalapack_name(num_type, prec, True) + ", 50% EVs", "green", "x-")
    line("total", mat_size, 10, "pdsyevr", "MKL 2017, " + scalapack_name(num_type, prec, True) + ", 10% EVs", "red", "x-")

    line("total", mat_size, 100, "elpa1", "ELPA 1, 100% EVs", "blue", "*--")
    line("total", mat_size, 50, "elpa1", "ELPA 1, 50% EVs", "green", "*--")
    line("total", mat_size, 10, "elpa1", "ELPA 1, 10% EVs", "red", "*--")

    line("total", mat_size, 100, "elpa2", "ELPA 2, 100% EVs", "blue", "o:")
    line("total", mat_size, 50, "elpa2", "ELPA 2, 50% EVs", "green", "o:")
    line("total", mat_size, 10, "elpa2", "ELPA 2, 10% EVs", "red", "o:")

def details(proc_ev):
    for i in range(len(elpa1_subtimes)):
        line(elpa1_subtimes[i], mat_size, proc_ev, "elpa1", "ELPA1 - " + elpa1_subtimes[i], group_colors[0][i], group_symbols[2*i] + '-') 

    for i in range(len(elpa2_subtimes)):
        line(elpa2_subtimes[i], mat_size, proc_ev, "elpa2", "ELPA2 - " + elpa2_subtimes[i], group_colors[1][i], group_symbols[i] + '-') 



fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
ax.tick_params(labelright='on')

plot1()
#details(100)

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
  y_min = 0.1
  y_max = 50
else:
  y_min = 5
  y_max = 500

yticks_major = [1,10,100,1000, y_min, y_max]
ax.yaxis.set_ticks(yticks_major)
ax.yaxis.set_ticklabels(yticks_major)
#    yticks_minor = [2, 5, 20, 50, 200, 500]
#    ax.yaxis.set_ticks(yticks_minor, minor=True)
#    ax.yaxis.set_ticklabels(yticks_minor, minor=True)
plt.ylim([y_min, y_max])
plt.xlim([20, 41000])
plt.savefig('plot.pdf')
#if show:
plt.show()
#plt.close()

