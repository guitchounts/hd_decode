"""
Various functions to wrangle data for decoding

"""


from scipy import stats, signal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest


def split_data(X, y, test_size, standardize=True,shuffle=False):
    """
    Split neural and head data into training and testing chunks. 
    Optionally, z-score or shuffle the neural data.

    X : ndarray of shape (n_samples, window, n_tetrodes)
        Neural data

    y : ndarray of shape (n_samples, )
        Head data

    test_size : float
        Fraction of data to put into testing set

    standardize : bool, default = True
        Whether to z-score the neural data

    shuffle : bool, default = False
        Whether to randomly shuffle the neural test set.

    
    Returns
    -------

    X_train, X_test, y_train, y_test

    """

    test_size_idx = int(test_size * X.shape[0])
    X_train, X_test, y_train, y_test = X[:-test_size_idx], X[-test_size_idx:], y[:-test_size_idx], y[-test_size_idx:]
    # X shape = e.g. (719800, 200, 16)
    if standardize:
        #Z-score X inputs. 
        X_train_mean = np.nanmean(X_train, axis=0)
        X_train_std = np.nanstd(X_train, axis=0)
        if 0 in X_train_std:
            print('Zero values encountered in X_train_std. Zero-centering but not Z-scoring.')
            X_train = X_train - X_train_mean
            X_test = X_test - X_train_mean
        else:
            X_train = (X_train - X_train_mean) / X_train_std
            X_test = (X_test - X_train_mean) / X_train_std

        #Zero-center outputs
        #y_train_mean = np.mean(y_train, axis=0)
        #y_train = y_train - y_train_mean
        #y_test = y_test - y_train_mean
    
    if shuffle:
        print('!!!!!!!!!!!!!!!!!! SHUFFLING X_test !!!!!!!!!!!!!!!!!!')
        #X_test = X_test.reshape(-1,shuffle_chunk_size)
        np.random.shuffle(X_test)
        #X_test.flatten()


    return X_train, X_test, y_train, y_test

def make_timeseries_instances(X, y, window_size, offset):
    """
    Take temporal window in the neural data and create time offset between neural data and head data
    
    Parameters
    ----------

    X : ndarray of shape (n_samples, n_tetrodes)
        input dataset

    y : ndarray of shape (n_samples, )
        output dataset

    window_size : int
        samples for temporal window

    offset : int
        samples for offset


    Returns
    -------

    X : ndarray (samples, window, n_tetrodes)

    y:  ndarray (samples, )


    """
    X = np.asarray(X)
    y = np.asarray(y)
    ###  shapes e.g. (617413, 16) (617413, 1)
    assert 0 < window_size <= X.shape[0]
    assert X.shape[0] == y.shape[0]

    print('In make_timeseries_instances, window_size, offset =   ', window_size,offset)

    size = int(X.shape[0])
    window_size = int(window_size)
    
    X = np.roll(X,int(offset),axis=0) ## negative roll = shift first n=offset elements to the rear, i.e. X will now start AT offset. positive roll = X starts at -offset i.e. the first elements of original X are moved to +offset.    
    
    X = np.atleast_3d(np.array([X[start:start + window_size] for start in range(0, size - window_size)]))
    y = np.atleast_3d(np.array([y[start:start + window_size] for start in range(0, size - window_size)])) ## e.g. (126446, 200, 1) or (126446, 200, 2) if phi
    y = y[:,0,:]
    
    return X, y

def timeseries_shuffler(X, y, series_length, padding):
    
    """

    Shuffle time series data, chunk by chunk into even and odd bins, discarding a pad
    between each bin.

    Parameters
    ----------
    
    X : ndarray of shape (n_samples, n_tetrodes)
        input dataset

    y : ndarray of shape (n_samples, )
        output dataset

    series_length : int
        length of chunks to bin by (samples)

    padding : int
        pad to discard between each chunk (samples)
    

    Returns
    -------

    X_shuffled, y_shuffled

    """

    X_even = []
    X_odd = []
    y_even = []
    y_odd = []

    # state variable control which bin to place data into
    odd = False
    
    for i in range(X.shape[0]):
        # after series_length + padding, switch odd to !odd
        if (i%(series_length+padding)) == 0:
            odd = not odd

        # only add to bin during the series period, not the padding period
        if (i%(series_length+padding))<series_length:
            
            # put X[i] and y[i] into even/odd bins
            if odd:
                X_odd.append(X[i])
                y_odd.append(y[i])
            else:
                X_even.append(X[i])
                y_even.append(y[i])

    # concatenate back together
    X_even.extend(X_odd)
    y_even.extend(y_odd)
    
    # put them back into np.arrays
    X_shuffled = np.asarray(X_even)
    y_shuffled = np.asarray(y_even)
    
    return X_shuffled, y_shuffled


def get_spikes_with_history(neural_data,bins_before,bins_after,bins_current=1):
    """
    Function that creates the covariate matrix of neural activity

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding

    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    print('******************************** Getting Spikes with History *************************************')

    num_examples=neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons=neural_data.shape[1] #Number of neurons
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    print('num_examples, num_neurons, surrounding_bins = ', num_examples, num_neurons, surrounding_bins)
    
    X=np.empty([num_examples,surrounding_bins,num_neurons]) #Initialize covariate matrix with NaNs
    #X[:] = np.NaN

    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1;
    return X



def circular_mean(data):
    """
    Function to compute mean of circular data.
    
    Parameters
    ----------
    data : ndarray 
        Array of circular data, in radians
    
    
    Returns
    -------
    
    theta : ndarray
        Circular mean, in radians
    
    """
    
    cos = np.sum(np.cos(data)) / data.size
    sin = np.sum(np.sin(data)) / data.size
    
    return np.arctan2(sin,cos)


def _ss_residual(y_true, y_pred):
    """
    
    Circular sum of squares of the model's predictions
    
    """
    
    mod_square = np.square(np.abs(y_true - y_pred) - 2*np.pi)
    raw_square = np.square(y_true - y_pred)
    better = np.minimum(mod_square,raw_square)
    
    return np.sum(better)



def _ss_total(y_true):
    """
    
    Circular sum of squares around the mean
    
    """
    
    mod_square = np.square(np.abs(y_true - circular_mean(y_true) ) - 2*np.pi)
    raw_square = np.square(y_true - circular_mean(y_true) )
    better = np.minimum(mod_square,raw_square)
    
    return np.sum(better)


def circular_r2_score(y_true, y_pred):
    """
    
    Circular goodness of fit (R^2)
    
    Parameters
    ----------
    y_true : ndarray 
        Array of circular data, in radians
    
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) circular target values. Assumed to be in radians.
    
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated circular target values. Assumed to be in radians.


    GG TO DO:
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    
    Returns
    -------

    Circular r2
        
        
    
    
    """
    
    
    return 1 - (_ss_residual(y_true,y_pred) / _ss_total(y_true) )



def pass_filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):
    """
    Design bandpass Butterworth filter for neural data:

    """

    [b,a] = signal.butter(filt_order,[freq/(fs/2) for freq in freq_range],btype=filt_type)
    
    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace
