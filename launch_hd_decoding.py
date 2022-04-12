"""
A launching pad for slurm batch jobs to decode head direction from V1 neural activity 

"""

import numpy as np
from subprocess import call
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-expt', type=str)

settings = parser.parse_args(); 

param_fn = "params_hd_decode_%s.txt" % settings.expt
fo = open(param_fn, "w")

expt = settings.expt


base_dir =  '/n/holyscratch01/cox_lab/Users/guitchounts/ephys/'  # '/n/home11/guitchounts/ephys'
save_folder = 'hd_decoding_results'

data_dir = "/n/holyscratch01/cox_lab/Users/guitchounts/ephys/hd_decoding_results/expt_%s/data" % settings.expt 
logs_dir = "/n/holyscratch01/cox_lab/Users/guitchounts/ephys/hd_decoding_results/expt_%s/logs" % settings.expt 
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)


rats = ['GRat26','GRat27','GRat31','GRat36','GRat54','GRat47','GRat48','GRat49','GRat50'] #,'GRat47','GRat48','GRat49','GRat50'] # ['GRat59','GRat61','GRat63'] #['GRat47','GRat48','GRat49','GRat50']  #
#rats = ['GRat59','GRat61']
#rats = ['GRat47','GRat48','GRat49','GRat50'] ## 'GRat26','GRat27','GRat31','GRat36','GRat54',

## if window = 200, centered lag = 100. generally, centered lag = int(window/2)
## if you want to vary the lags while keeping the window constant, you should 1) reduce the window size to 1, and 
## change the lags. in this case, negative lags = x is ahead of y (i.e. decoding w/ future neural activity)
## if lag is positive, x is behind y (i.e. decoding w/ past neural activity)

windows = np.array([10,26,50,100,150,200]) # np.array([1],dtype='int16') # #np.array([10,26,50,100,150,200]) #  # 
lags = [0] #np.array(windows / 2,dtype='int16')  ## np.array([-100],dtype='int16')       #    #np.array([100],dtype='int') # np.arange(-100,125,25,dtype='int') #  np.arange(-100,125,25,dtype='int') # in samples...  #
### ^^^ lag should be window / 2 (not -2) to center the window 
### for a moving lag: lag==0 means the X window starts at y and moves forward in time ahead of y; lag==window-1 means the X window is lagging behind y (i.e. X is in the past relative to y)


decode = True


model_type = 'lstm' #'ridge_phi' # ,'ridge_phi']

splits =  [0.5] # np.arange(0.1,1.0,0.1) #

moving_resting_keys = ['moving','resting','movingresting']

i = 1
for rat in rats:

    input_file_path = '/n/coxfs01/guitchounts/ephys/%s/Analysis/' % rat
    rat_fils = get_files(input_file_path)

    for fil in rat_fils:
        exp_name = get_exp_name(rat,fil,input_file_path) ## dark or light or flash...
        
        if (exp_name.lower() == 'dark' or exp_name.lower() == 'light'): ### only add the exp line if it's dark or light:
            
            for window_idx,window in enumerate(windows):
                
                for lag_idx,lag in enumerate(lags):

                    for split_idx,split in enumerate(splits):

                        for moving_resting in moving_resting_keys:
            
                            model_name = '%s_%s_%s_window_%d_lag_%d_split_%d_%s' % (rat,fil,exp_name,window_idx,lag_idx,split_idx, moving_resting )

                            print(rat, fil, exp_name, model_type, window, lag, split,moving_resting, model_name)  # '/n/home11/guitchounts/ephys/hd_decoding_results/expt_test/data/'

                            fo.write("-rat %s -fil %s -exp_name %s -model_type %s -window %f -lag %f -split %f -moving_resting %s -model_name %s -decode %d -savefile %s/%s/expt_%s/data/\n" % (rat,fil,exp_name,model_type,
                                                                                                                            window,lag,split, moving_resting, model_name, decode,
                                                                                                                                base_dir,save_folder, expt))
                            i+=1

                    
fo.close()

call("python run_odyssey_array.py -cmd run_hd_decode_odyssey.py -expt %s -cores 8 -hours 1 -mem 32000 -partition gpu -paramfile %s -env pytorch" % (expt,param_fn), shell=True)
#call("python run_odyssey_array.py -cmd run_hd_decode_odyssey.py -expt %s -cores 8 -hours 24 -mem 184000 -partition serial_requeue -paramfile %s" % (expt,param_fn), shell=True)



def get_files(input_file_path):
    ### collect all of this rat's 636 file names:
    
    all_files = []

    for file in os.listdir(input_file_path):
            if file.startswith("636"):
                if len(file) == 18: ### don't take the concatenated double 636xxx_636xxx files!
                    all_files.append(file)

    return np.asarray(all_files)

def get_exp_name(rat,fil,input_file_path):

    for exp in os.listdir('%s/%s/' % (input_file_path,fil) ): ### e.g. ./636596531772835142/
    
        if exp.startswith('%s_0' % rat.lower()):
            exp_name = exp[exp.find('m_')+2:exp.find('.txt')]

            
        elif exp.startswith('%s_1' % rat.lower()):
            exp_name = exp[exp.find('m_')+2:exp.find('.txt')]

            
    return exp_name
