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

# data traces are too large for this example and are not included

# create new matplotlib figure and pass it to the Plotter
fig = plt.figure()
pl = Plotter(fig)

# load two traces, one from a run with GPU enabled, one without
# note that this might take a lot of time, so if we have allready run this script and than only
# changed something in the following, we might choose to comment out the following commands
trace_ok = Trace('data_ok', "OK")
trace_defect = Trace('data_defect', "defect")

# descriptions are "metadata", they describe when and what has been outputed
# we check that they do not differ for the two program runs
assert(trace_ok._descriptions == trace_defect._descriptions)

# we can then use one iterator to acces both data sets
it = Iterator(trace_ok._descriptions)

# we want to see matrix A both from the run we know was OK and that where was error
a_mat_ok = Snapshot(trace_ok, "a_mat", it)
a_mat_defect = Snapshot(trace_defect, "a_mat", it)

# we also want to see where they differ, so that we can find out where the difference originates
a_mat_diff = Difference(a_mat_ok, a_mat_defect)

# we have created 3 views, lets group them into a list
snapshot_set = [a_mat_ok, a_mat_defect, a_mat_diff]

# and tell the Plotter to show them. Note, that all the plots were created to follow
# the same iterator it, so we can change time point shown in all views by iterating it.
pl.reset(snapshot_set)
