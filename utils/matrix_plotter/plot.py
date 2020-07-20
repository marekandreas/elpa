import numpy as np
import matplotlib.pyplot as plt
import os

class Trace:
  """Holds all outputs collected from 1 run of the program.

  This might include images of different matrices at different time instances.
  If data has been collected only during one run, only one instance of Trace can be created
  If, however, we are interested in comparing 2 runs (e.g. defective commit and commit which
  we know is correct, 2 instances of Trace will be created)
  
  At the current moment, the only think that can be stored and viewed is the matrix stored in 
  block cyclic distribution
  
  Attributes:
    directory: path to the directory with collected data
    name: identifier of the trace (e.g. correct_run, defective_run, etc...), just for printing purposes
  """
  def __init__(self, directory, name):      
    """Initialize"""
    self._name = name
    self.__load_matrices(directory)

  def __get_header(self, filename):
    with open(filename, 'r') as f:
      header = f.readline()
      val = [int(x) for x in header.split()]
      val.append(f.readline()[:-1])
      print(val)
    return  val

  def __load_matrix(self, directory, filenames):
    mat = []
    for (mpi_rank, filename) in filenames:
      #print(filename, mpi_rank, prow, pcol, p_cols)
      (na, nblk, lda, localCols, my_prow, my_pcol, p_rows, p_cols, iteration, name) = self.__get_header(directory + '/' + filename)
      if(mat == []):
          mat = np.zeros((na, na))
      (self._na, self._nblk, self._p_rows, self._p_cols) = (na, nblk, p_rows, p_cols)
      prow = mpi_rank % p_rows
      pcol = mpi_rank / p_rows
      #print(na, nblk, lda, localCols, my_prow, my_pcol, my_nprows, my_npcols)
      assert(my_prow == prow) 
      assert(my_pcol == pcol) 
        
      loc_mat = np.loadtxt(fname = directory + '/' + filename, skiprows = 2)
      #print("lda, localCols ", lda, localCols)
      for row_blk in range((lda-1)/nblk + 1):
        loc_row_beg = row_blk * nblk
        if(loc_row_beg >= lda):
          continue
        
        loc_row_end = min(loc_row_beg + nblk, lda)
              
        glob_row_beg = (row_blk * p_rows + prow) * nblk  
        #print("glob_row_beg = row_blk * nblk * p_rows ", glob_row_beg, row_blk, nblk, p_rows)
        assert(glob_row_beg < na)
        
        glob_row_end = min(glob_row_beg + nblk, na)
        
        for col_blk in range((localCols-1)/nblk + 1):
          loc_col_beg = col_blk * nblk
          if(loc_col_beg >= localCols):
            continue
          
          loc_col_end = min(loc_col_beg + nblk, localCols)
              
          glob_col_beg = (col_blk * p_cols + pcol) * nblk 
          assert(glob_col_beg < na)
          
          glob_col_end = min(glob_col_beg + nblk, na)
          
          #print("local", (loc_row_beg, loc_row_end), (loc_col_beg, loc_col_end))
          #print("global", (glob_row_beg, glob_row_end), (glob_col_beg, glob_col_end))
          
          mat[glob_row_beg:glob_row_end, glob_col_beg:glob_col_end] = loc_mat[loc_row_beg:loc_row_end, loc_col_beg:loc_col_end]
    return ((name, iteration), mat)
    
  def __load_matrices(self, directory):
    filenames_dict = {}

    for filename in os.listdir(directory):
      (mat_name, instant, mpi_rank) = filename.split(".")[0].split("-")
      instant = int(instant)
      mpi_rank = int(mpi_rank)

      if(not instant in filenames_dict.keys()):
        filenames_dict[instant] = {}
      if(not mat_name in filenames_dict[instant].keys()):
        filenames_dict[instant][mat_name] = []
      filenames_dict[instant][mat_name].append((mpi_rank,filename))
          
    self._matrices = {}
    self._descriptions = {}
    for instant in filenames_dict.keys():
      self._matrices[instant] = {}
      for mat_name in filenames_dict[instant].keys():
        (ids, matrix) = self.__load_matrix(directory, filenames_dict[instant][mat_name])
        self._matrices[instant][mat_name] = matrix
        if(not instant in self._descriptions.keys()):
          self._descriptions[instant] = ids
        else:
          assert(self._descriptions[instant] == ids)
          

class Iterator:
  """ Allows us to traverse through the history
  
  Contains different methods for moving forward or backword in outputed time instances
  It allows to have different views of different matrices from (possibly) different runs
  of the code bind to the same iterator, so that by manipulationg it (e.g. going to "next")
  we go to next in all simultaneously active views.
  
  Attributes:
    description: identifier of the iterator, to be shown in the plot
  """
  def __init__(self, description):
    self._description = description
    self._index = 0
    self._registerd_views = []
    
  def __str__(self):
    descr = self._description[self._index]
    return "snapshot " + str(self._index) + ", iteration " + str(descr[1]) + ", " + descr[0]
  
  def set(self, it2):
    """ Sets the iterator to the same state as another iterator it2 """
    self.set_snapshot(it2._index)
    
  def next(self):
    """ Moves the iterator to the next image and _updates all _attached images """
    if(self._index < self._description.keys()[-1]):
      self._index += 1
      self._status_changed()
    
  def prev(self):
    """ Moves the iterator to the previous image and _updates all _attached images """
    if(self._index > 0):
      self._index -= 1
      self._status_changed()
      
  def set_snapshot(self, sn):
    """ Sets iterator to the given snapshot sn. There can be multiple snapshots per iteration """
    if(sn >= 0 and sn <= self._description.keys()[-1]):
      self._index = sn
      self._status_changed()
      
  def set_iteration(self, it):
    """ Sets iterator to the given iteration """
    for k in self._description.keys():
      if(self._description[k][1] == it):
        self.set_snapshot(k)
        return
 
  def next_event(self, event = None):
    """ Sets iterator to the next event given by parameter. 
    If not specified, moves to the next occurence of the same event it points to now"""
    sn = self._index
    if(event == None):
      event = self._description[sn][0]
    while(sn < self._description.keys()[-1]):
      sn += 1
      if((self._description[sn][0]).startswith(event)):
        self.set_snapshot(sn)
        return
    
  def prev_event(self, event = None):
    """ Sets iterator to the previous event given by parameter. 
    If not specified, moves to the previous occurence of the same event it points to now"""
    sn = self._index
    if(event == None):
      event = self._description[sn][0]
    while(sn > 0):
      sn -= 1
      if((self._description[sn][0]).startswith(event)):
        self.set_snapshot(sn)
        return   
  
  def get_num_iterations(self):
    """ Returns maximal number of iteration found in the description """
    return max([self._description[k][1] for k in self._description.keys()])
  
  def get_events(self):
    """ Returns all the evetn strings. They (or their beginnings) can be used to call next(prev)_event """
    return set([self._description[k][0] for k in self._description.keys()])
    
  def current(self):    
    return self._index
     
  def _short_str(self):
    descr = self._description[self._index]
    return "sn " + str(self._index) + ", it " + str(descr[1]) + ", " + descr[0]
  
  def _combined_short_str(self, iter2):
    if (self._index == iter2._index):
      descr = self._description[self._index]
      return "sn " + str(self._index) + ", it " + str(descr[1]) + ", " + descr[0]
    else:
      return "sn " + str(self._index) + "/" + str(iter2._index)
  
  def _register_view(self, view):
    self._registerd_views.append(view)
  
  def _status_changed(self):
    for view in self._registerd_views:
      view._update()

class View:
  """ Abstract class representing 1 view (plot) of matrix-like data
  It can be a matrix itself, difference of 2 matrices, etc... 
  """
  def __init__(self):
    self._ax = None
    self._show_grid = False
  
  def show_grid(self, show=True):
    """ Set whether show grid between different processors """
    self._show_grid = show
    self._update()
    
  def _detach(self):
    self._ax = None
    self._plotter = None
    
  def _attach(self, plotter, ax):
    assert(self._ax == None)
    self._ax = ax
    self._plotter = plotter
    
  def _update(self):
    if self._ax != None:
      self._plot(self._ax)
      self._plotter._show()
          
          
class Snapshot(View):
  """ Represents state of 1 particular matrix from 1 particular program run (trace) at 
  1 particular time instance (represented by state of iterator)
  
  It can be used to view the value of a matrix in a given time or to define difference 
  of two snapshots
  
  Attributes:
    trace: instance of Trace, determines from which run of the code we want to take data
           (we might have saved multiple runs)
    matrix: which matrix we want to look at (we might have saved values of more matrices)
    iterator: determins the time instance of interest. Note that the state of Snapshot 
              is changed by changing the state of the iterator
  """
  def __init__(self, trace, matrix, iterator):
    View.__init__(self)
    self._trace = trace
    self._matrix = matrix
    self._iterator = iterator    

    # this is just a hack to fix the scale, it should be done better
    # the scale is fixed to show nicely the first matrix and is kept for the rest
    # !!! this might be very stupid for certain matrices (etc. zero at the begining )
    self.set_limits_from_first()
    
    iterator._register_view(self)
    
  def set_limits(self, lim_lo, lim_hi):
    """ fix limits for the plot, will be kept until changed again"""
    self._limits = (lim_lo, lim_hi)
    
  def set_limits_from_first(self):
    """fixing min and max based on first snapshot, not optimal! """
    mat = self._trace._matrices[0][self._matrix]
    self.set_limits(np.percentile(mat,1), np.percentile(mat,99))
  
  def write(self):
    print(self._matrix, ", ", it._index)
    
  def _get_matrix(self):
    return self._trace._matrices[self._iterator.current()][self._matrix]
    
  def _plot(self, ax):
    mat = self._get_matrix().copy()

    mat = np.minimum(mat, self._limits[1])
    mat = np.maximum(mat, self._limits[0])
    ax.clear()
    ax.matshow(mat)
    ax.set_title(self._trace._name + ', ' + self._matrix + ', ' + self._iterator._short_str())
    
    if(self._show_grid):
      self._plot_grid(ax)
  
  def _plot_grid(self, ax):
    x = 0
    nblk = self._trace._nblk
    na = self._trace._na
    p_rows = self._trace._p_rows
    p_cols = self._trace._p_cols
    while (x < na):
      if((x/nblk % p_rows == 0) and (x/nblk % p_cols == 0)):
        color = 'magenta'
      else:
        color = 'green'
      ax.plot([-0.5,na-0.5],[x-0.5,x-0.5], color=color, linewidth=5)
      ax.plot([x-0.5,x-0.5],[-0.5,na-0.5], color=color, linewidth=5)
      y = 0
      if(x < na - 1):
        while (y < na-1):
          proc_row  = y/nblk % p_rows
          proc_col  = x/nblk % p_cols
          my_rank = proc_row + p_rows * proc_col
          x_text = x+nblk/2-1
          y_text = y+nblk/2+1
          if(x_text <= na and y_text <= na):
            ax.text(x_text, y_text , str(my_rank), fontsize = 23, color='green')
          y+= nblk
      x += nblk
    
    ax.set_xbound(-0.5,na-0.5)
    ax.set_ybound(-0.5,na-0.5)
    
class Difference(View):
  """ Compares two snapshots and shows where they differ
  
  Shows entries where two matrices defined by two snapshots differ
  Intended for debugging and trying to find out where two calculations
  start to diverge. The two snapshots can represent the same matrix in 
  two different runs (e.g. at different git commits) or matrix on host 
  and device, etc..
  
  In principle, one could define and show any operation on two matrices, 
  the operation could be passed as callback function. At the moment, this
  one seems to be the most reasonable
  
  Attributes:
    sn1, sn2: the two snapshots to be compared
  """
  def __init__(self, sn1, sn2):
    View.__init__(self)
    self._sn1 = sn1
    self._sn2 = sn2
    
    sn1._iterator._register_view(self)
    sn2._iterator._register_view(self)

  def _plot(self, ax):
    diff = (abs(self._sn1._get_matrix() - self._sn2._get_matrix()) > 0.00001)
    ax.clear()
    ax.matshow(diff)
    if(self._sn1._trace._name == self._sn2._trace._name):
      trace_name = self._sn1._trace._name
    else:
      trace_name = self._sn1._trace._name + '/' +self._sn2._trace._name
    if(self._sn1._matrix == self._sn2._matrix):
      matrix_name = self._sn1._matrix
    else:
      matrix_name = self._sn1._matrix + '/' + self._sn2._matrix
      
    if(self._sn1._iterator._index == self._sn2._iterator._index):
      iterator_name = self._sn1._iterator._short_str()
    else:
      iterator_name = self._sn1._iterator._short_str() + '/' + self._sn2._iterator._short_str()
    
    ax.set_title(trace_name + ', ' + matrix_name + '\n' + iterator_name)
    if(self._show_grid):
      self._plot_grid(ax)

  def _plot_grid(self, ax):
    self._sn1._plot_grid(ax)
    self._sn2._plot_grid(ax)


class Plotter:
  """ Encapsulates the matplotlib figure and handles updating of the plot when iterators change
  
  It can be configured with 1 or more views by reset function. If more views are required (limited
  to 4 at the moment), the views are automatically shown in subplots of the matplotlib plot. This 
  can be handy when comparing more matrices at the time, especially when they share the same iterator
  and thus one can see changes of more matrices after calling single command to the iterator.
  
  Attributes:
    fig: matplotlib figure
  """
  def __init__(self, fig):
    plt.ion()
    #plt.close()    
    self._fig = fig
    self._views = []

  def reset(self, views, title = "View"):
    """ Defines what the Plotter should show
    It can show 1 matrix or grid of more matrices, depending on parameter views
    
    Attributes:
      views: 1 instance of View or list of up to 4 instances of View to be shown
      title: name of the Plotter
    """
    if(not isinstance(views, list)):
      self.reset([views], title)
      return
    
    self._fig.clear()
    #self._fig.tight_layout()
    
    self._detach_views()
    self._views = views
    
    self._subplots = []
    assert(len(views) <= 4)

    if(len(views) == 1):
      ax = self._fig.add_subplot(111)
      ax.set_xticklabels([])
      ax.set_yticklabels([])

      self._subplots.append(ax)
      views[0]._attach(self, ax)      
    else:    
      subplot_nr = 1
      for view in views:
        ax = self._fig.add_subplot(220 + subplot_nr)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        self._subplots.append(ax)
        view._attach(self, ax)      
        subplot_nr += 1
    
    self._fig.suptitle(title, fontsize=20)
    self._plot()
    
  def show_grid(self, show = True):
    """ Set whether show grid between different processors, applied to all the views """
    for view in self._views:
      view.show_grid(show)

  def _plot(self):
    for view in self._views:
      view._update()
    self._fig.show()    
    
  def _show(self):
    self._fig.show()
  
  def _detach_views(self):
    for view in self._views:
      view._detach()
    
