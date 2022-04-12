"""
A way to gather parameter data from a config file and start decoding V1 head direction data

"""


import sys,os
import argparse
import time
from functools import reduce

import numpy as np
from scipy import stats,signal,io
import h5py

from hd_decode import start_decoding,decode_cross_file, start_decoding_lstm


if __name__ == "__main__" :

    print('Starting work in run_hd_decode_odyssey.py...')


    parser = argparse.ArgumentParser()

    parser.add_argument('-paramfile', type=argparse.FileType('r'))
    parser.add_argument('-line', type=int)

    #parser.add_argument('-savefile', type=argparse.FileType('w'))
    parser.add_argument('-savefile', type=str) # e.g. %s/%s/expt_%s/data/%s/ % base_dir,save_folder, expt, model_name, ## ADD to this the CHUNK # 
    parser.add_argument('-model_name', type=str) ## e.g. = '%s_%s_%s_window_%d_lag_%d_split_%f' % (rat,fil,exp_name,window_idx,lag_idx )

    parser.add_argument('-rat', type=str, default='GRat')
    parser.add_argument('-fil', type=str, default='636')
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-model_type', type=str, default='ridge')


    parser.add_argument('-window', type=float, default=200) # window size
    parser.add_argument('-lag', type=float, default=0) # window lag

    parser.add_argument('-split', type=float, default=0.5) # train/test split

    parser.add_argument('-decode_mua', type=int, default=1) # whether or not to decode on MUA (True) or SUA (False)

    parser.add_argument('-unit_class', type=int, default=2) # if SUA: which unit class to take. 2 = both 

    parser.add_argument('-moving_resting', type=str, default='movingresting')

    parser.add_argument('-decode', type=int, default=1) ## whether to make a decoding model or do cross-file decoding from previous models




    settings = parser.parse_args(); 


    # Read in parameters from correct line of file
    if settings.paramfile is not None:
        for l, line in enumerate(settings.paramfile):
            if l == settings.line:
                settings = parser.parse_args(line.split())
                break
                


    rat = settings.rat
    fil = settings.fil
    exp_name = settings.exp_name
    window = settings.window
    lag = settings.lag
    split = settings.split
    model_type = settings.model_type ## e.g. ridge
    model_name = settings.model_name ### e.g. '%s_%s_%s_window_%d_lag_%d' % (rat,fil,exp_name,window_idx,lag_idx )

    decode_mua = bool(settings.decode_mua) ## bool to see if we're decoding MUA (TRUE) or SUA (FALSE)
    unit_class = settings.unit_class

    moving_resting = settings.moving_resting

    decode = bool(settings.decode)

    save_folder = settings.savefile ## i.e. ....../expt1/data/
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    savefile = '%s/%s' % (save_folder,model_name) #### e.g. %s/%s/expt%d/data/%s % (base_dir,save_folder, expt, model_name) + model_name 
    # /n/home11/guitchounts/ephys/hd_decoding_results/exptX/data/GRatX_636X_window_X_lag_X



    head_signals,neural_data,total_acc = load_data(rat,fil,decode_mua) 

    print('Neural and head data loaded...')
    print('decode_mua = ', decode_mua)

    if not decode_mua:
        print('****************** Decoding SINGLE UNIT DATA ******************')

        unit_class_info = np.load('/n/holyscratch01/cox_lab/Users/guitchounts/ephys/unit_class_info.npz')['class_array'] ## (482,3) w/ unit num, fil, and class columns 
        
        ## FIND THE units corresponding to this fil:
        unit_classes = unit_class_info[np.where(unit_class_info[:,1] == fil)[0],2].astype('int')

        if unit_class != 2: ## unit_class==2 means decode from both types; otherwise, take either the 0's or 1's:
            
            ## if there aren't any units of the class we're aiming for, then break
            unit_idxs = np.where(unit_classes == unit_class)[0]

            if len(unit_idxs) > 0:

                neural_data = neural_data[:, unit_idxs ]

            else:
                print('No units of class %d were found. Breaking...' % unit_class)
                sys.exit()


    head_names = ['yaw','roll','pitch']
      
    ### set a 2 hour limit for recordings for linear models (XXX for LSTM?), cut out bad signals, and make chunks:
    time_lim = int(2e6) if model_type == 'lstm' else int(100*60*60*2) ## 720000 samples @ 100hz; get 200000 @ 10 hz -> lim= 2e6

    num_chunks = max(1,int(head_signals.shape[0] / time_lim)) ## how many two-hour chunks of decoding can we do using this dataset?
    # split tetrodes and head data into chunks:
    chunk_indexes = [time_lim*i for i in range(num_chunks+1)] ## get indexes like [0, 720000] [720000, 1440000] [1440000, 2160000]
    chunk_indexes = [[v, w] for v, w in zip(chunk_indexes[:-1], chunk_indexes[1:])] # reformat to one list


    print('chunk_indexes = ', chunk_indexes)

    all_tetrodes = [neural_data[chunk_indexes[chunk][0]:chunk_indexes[chunk][1],:] for chunk in range(num_chunks)  ] ## list of 16x720000 chunks
    all_head_signals = [head_signals[chunk_indexes[chunk][0]:chunk_indexes[chunk][1],:] for chunk in range(num_chunks)  ]
    all_total_accs = [total_acc[chunk_indexes[chunk][0]:chunk_indexes[chunk][1]] for chunk in range(num_chunks)  ]
    print('all_head_signals[chunk] shape after chunking.   =', all_head_signals[0].shape)
    print('all_tetrodes[chunk] shape after chunking.   =', all_tetrodes[0].shape)


    #### to GET BEHAVIORAL COVERAGE of all the angles:
    roll_bins = np.arange(-90,90,10)
    pitch_bins = np.arange(-180,180,10)
    yaw_bins = np.arange(0,360,10)
    all_bins = {'yaw' : yaw_bins,'roll': roll_bins,'pitch': pitch_bins}


    # if SUA, there's only data in the first chunk. cut the rest:
    if not decode_mua:
        num_chunks = 1
        print('SUA: setting num_chunks to 1')
       


    ## iterate thru chunks
    for chunk in range(num_chunks):

        tetrode = all_tetrodes[chunk]
        head_chunk = all_head_signals[chunk]
        total_acc_chunk = all_total_accs[chunk]
        
        if model_type != 'lstm':

            for head_signal_idx in range(head_signals.shape[1]): # yaw, roll, pitch

                y_key = head_names[head_signal_idx]
                head_signal = head_chunk[:,head_signal_idx]

                save_dir = '%s_chunk_%d_%s.npz' % (savefile,chunk,y_key) 

                if not os.path.isfile(save_dir):

                    ### e.g. /n/home11/guitchounts/ephys/hd_decoding_results/exptX/data/GRatX_636X_window_X_lag_X_chunk_0_yaw.npz

                    print('Running MUA HD Decoding Odyssey script on %s %s chunk %d, %s,\nsave_dir = %s...' % (rat,fil,chunk,y_key,save_dir)  )

                    if decode:
                        y_true,y_pred = start_decoding(X=tetrode,y=head_signal,window=window,lag=lag,model_type=model_type,
                            head_name=y_key,save_dir=save_dir,split=split,moving_resting=moving_resting,total_acc=total_acc_chunk)

                        ###  Get true / predicted histogramed in bins: (for plotting true vs predicted as heatmaps)
                        H, xedges, yedges = np.histogram2d(x=np.squeeze(y_true),y=np.squeeze(y_pred),bins=all_bins[y_key]) # have to be np.squeezed b/c shape e.g. (1234234,1)

                        ### GET BEHAVIORAL COVERAGE of all the angles:
                        #hist,edges = np.histogram(head_signal,all_bins[y_key])

                        ## save the behavioral coverage or true/pred histograms:
                        np.savez('%s_chunk_%d_%s_histogram.npz' % (savefile,chunk,y_key),hist=H,xedges=xedges,yedges=yedges)

                    else:
                        print('Running cross-file decoding on pre-saved models')


                        decode_cross_file(X=tetrode,y=head_signal,window=window,lag=lag,model_type=model_type,
                            head_name=y_key,save_dir=save_dir,split=split,moving_resting=moving_resting,total_acc=total_acc_chunk,rat=rat,fil=fil)

                else:

                    print('File %s exists. Stopping...' % save_dir)

                ## save y_true and y_pred:
                #np.savez('%s_chunk_%d_%s_truepred.npz' % (savefile,chunk,y_key),y_true=y_true,y_pred=y_pred)
            
        else:

            
            save_dir = '%s_chunk_%d.npz' % (savefile,chunk) 
            total_acc_chunk = all_total_accs[chunk]

            y_true,y_pred = start_decoding_lstm(X=tetrode,y=head_chunk,window=window,lag=lag,model_type=model_type,
                            save_dir=save_dir,split=split,moving_resting=moving_resting,total_acc=total_acc_chunk)






def load_data(rat,fil,decode_mua):

    print('Loading neural and head data.........................')
    
    num_tries = range(5)

    base_dir_head = '/n/holyscratch01/cox_lab/Users/guitchounts/ephys/%s/Analysis/%s/' % (rat,fil)
    head_data_name = 'all_head_data_100hz.hdf5'

    if decode_mua:
        
        base_dir_neural = '/n/holyscratch01/cox_lab/Users/guitchounts/ephys/%s/Analysis/%s/' % (rat,fil)
        
        neural_data_name = 'mua_firing_rates_100hz.hdf5'

    else: ## i.e. SUA
        
        base_dir_neural = '/n/holyscratch01/cox_lab/Users/guitchounts/ephys/%s/%s/' % (rat,fil) ## skipping 'Analysis' for the sua path
        
        neural_data_name = 'better_sua_firing_rates_100hz.hdf5'

    #### LOAD THE NEURAL DATA:
    for t in num_tries:
        try:

            start = time.time()

            neural_file = h5py.File('%s/%s' % (base_dir_neural,neural_data_name), 'r')
            head_signals_h5 = h5py.File('%s/%s' % (base_dir_head,head_data_name), 'r')

            end = time.time()

            print('Loaded neural file and head signals. It took %.2f seconds.' %  (end - start) )

        except (FileNotFoundError,NotADirectoryError,OSError) as error:
            print('Path %s doesnt exist. Waiting.....' % base_dir_neural)
            os.path.exists(base_dir_neural)
            time.sleep(60)
        else:
            break


    neural_data_keys = list(neural_file.keys())[0]
    neural_data = np.asarray(neural_file[neural_data_keys]) #
    fs = 100.
    
    print('neural_data shape: ', neural_data.shape)



    ## LOAD THE HEAD DATA:
    
    idx_start, idx_stop = [0,9]
    head_signals = np.asarray([np.asarray(head_signals_h5[key]) for key in head_signals_h5.keys()][0:9]).T[:,idx_start:idx_stop]
    
    total_acc = filter(np.sqrt(head_signals[:,0]**2 + head_signals[:,1]**2 + head_signals[:,2]**2     ),[1],filt_type='lowpass',fs=fs)

    head_signals = head_signals[:,6:9] ## yaw, roll, pitch
    

    ## lowpass filter:
    for x in range(head_signals.shape[1]):
        
        head_signals[:,x] = filter(head_signals[:,x],[1],filt_type='lowpass',fs=fs)
    print('head_signals shape: ', head_signals.shape)


    head_signals_h5.close()
    neural_file.close()
    
    ## in case the BNO recording failed and recorded a bunch of zeros, cut out those zeros from the end:
    start,stop = 0,get_head_stop(head_signals)
    head_signals = head_signals[start:stop,:]
    neural_data = neural_data[start:stop,:]
    total_acc = total_acc[start:stop]
    print('head_signals shape after start,stop = ', head_signals.shape)
    print('neural_data shape after start,stop = ', neural_data.shape)
    print('total_acc shape after start,stop = ', total_acc.shape)

    return head_signals,neural_data,total_acc



def zero_runs(a): 
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0] #.reshape(-1, 2)
    if len(ranges) > 0:
        
        return ranges[0]
    else:
        return a.shape[0]

def filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):

    # design Elliptic filter:

    [b,a] = signal.butter(filt_order,[freq/(fs/2) for freq in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace

def get_head_stop(head_data): ## head_data.shape = e.g. (1000000, 4)
    all_diffs = []
    head_names = range(head_data.shape[1])  #['ox','oy','oz','ax','ay','az']
    for head_name in head_names:
        diffs =np.where(np.diff(head_data[:,head_name]) == 0 )[0] ##  zero_runs(np.diff(head_data[:,head_name])) ###
        all_diffs.append(diffs)
        print('Getting start/stop coordinates for %s. Shape of diffs = ' % (head_name), diffs.shape)

    all_zeros = reduce(np.intersect1d, (all_diffs))
    #stop = np.min(all_diffs)
    if len(all_zeros) == 0:
        stop = head_data.shape[0] + 1
    else:
        stop = all_zeros[0]

    
    return stop


