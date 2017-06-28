# Sample script showing a use case of ELPA matrix output for debugging
# In this case, we wanted to compare matrices for GPU and CPU variant of the code

# This feature is only experimental and will probably change
# The recomanded ussage is the following. Go to the directory of this script and run:
# ipython -i ../plot.py 
# than run within python the command:
# run -i script.py
# Now a window with the plot should be created. Rearange your windows such that you can 
# see both console and plot and iterate using the iterators, change views or create new, etc.
# If you want to modify this file and re-run it without closing the plotting window, comment
# out the two following lines. If you do not want to reload the data, comment out creating of 
# traces. 

# To try this example untar the file data.tgz

# create new matplotlib figure and pass it to the Plotter
fig = plt.figure()
pl = Plotter(fig)

# load two traces, one from a run with GPU enabled, one without
# note that this might take a lot of time, so if we have allready run this script and than only
# changed something in the following, we might choose to comment out the following commands
trace = Trace('data_gpu', "w gpu")
trace_nongpu = Trace('data_nongpu', "nogpu")

# descriptions are "metadata", they describe when and what has been outputed
# we check that they do not differ for the two program runs
assert(trace._descriptions == trace_nongpu._descriptions)

# we can then use one iterator to acces both data sets
it = Iterator(trace._descriptions)

# we want to see two matrices from the run with GPU, we have both matrix on host and on device
a_mat = Snapshot(trace, "a_mat", it)
a_dev = Snapshot(trace, "a_dev", it)

# we are also interested in difference between the two matrices
a_mat_dev_diff = Difference(a_mat, a_dev)

# from the second trace we can extract only matrix on host (no matrix on device)
a_nongpu = Snapshot(trace_nongpu, "a_mat", it)

# we have created 4 views, lets group them into a list
snapshot_set = [a_mat, a_dev, a_mat_dev_diff, a_nongpu]

# and tell the Plotter to show them. Note, that all the plots were created to follow
# the same iterator it, so we can change time point shown in all views by iterating it.
pl.reset(snapshot_set)

# we can also create another iterator it_2 and the same shnapshot set, but binded to this new 
# iterator it_2, to be able to switch between two time instances
it_2 = Iterator(trace._descriptions)
a_mat_2 = Snapshot(trace, "a_mat", it_2)
a_dev_2 = Snapshot(trace, "a_dev", it_2)
a_mat_dev_diff_2 = Difference(a_mat_2, a_dev_2)
a_nongpu_2 = Snapshot(trace_nongpu, "a_mat", it_2)

# we can then show this new set by calling pl.reset(snapshot_set_2)
snapshot_set_2 = [a_mat_2, a_dev_2, a_mat_dev_diff_2, a_nongpu_2]

# we have allready used Difference of two different matrices at the same time instance. We can also
# define differences of the same matrix in different time instances and see what has changed
a_mat_diff = Difference(a_mat, a_mat_2)
a_dev_diff = Difference(a_dev, a_dev_2)
a_nongpu_diff = Difference(a_nongpu, a_nongpu_2)

# again we group them to a set and can show them by pl.reset(diffs_set)
diffs_set = [a_mat_diff, a_dev_diff, a_nongpu_diff, a_mat]
