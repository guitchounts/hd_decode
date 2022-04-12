"""
Create the models for a given session and head variables. Called by run_hd_decode_odyssey.py

"""

import numpy as np
from glob import glob
import joblib
import time
import psutil

from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn import linear_model
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import r2_score
from astropy.stats import circcorrcoef
from scipy import stats
from scipy.signal import medfilt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from data_helpers import split_data, make_timeseries_instances, timeseries_shuffler,circular_r2_score


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, device):
        super(LSTMModel, self).__init__()

        self.device = device

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.to(self.device)
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = c0.to(self.device)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0, c0 ))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

def modified_mse(y_pred, y_true):
    #loss = (y_true - y_pred) ** 2
    
    mod_square = torch.square(torch.abs(y_pred - y_true) - 360)
    raw_square = torch.square(y_pred - y_true)
    better = torch.minimum(mod_square,raw_square)
    
    return torch.mean(better)        

def modified_mse_numpy(y_true, y_pred): #### modified MSE loss function for absolute yaw data (0-360 values wrap around)
    
    # 1. take differences of every element:
    diffs = y_true - y_pred
    
    
    # 2. if the abs difference is over 180, subtract 360 from it? 
    ## e.g. abs(0-359)=359 -> 359-360=1; abs(359-0) same;;; abs(0-181)=181 -> 181-360=-179
    
    big_diff_idx = np.where(np.abs(diffs) > 180)
    diffs[big_diff_idx] = np.abs(diffs[big_diff_idx]) - 360
    
    mse = np.average(diffs**2)
    
    return mse


def modified_mae(y_true, y_pred): #### modified MAE loss function for absolute yaw data (0-360 values wrap around)
    
    # 1. take differences of every element:
    diffs = y_true - y_pred
    
    
    # 2. if the abs difference is over 180, subtract 360 from it? 
    ## e.g. abs(0-359)=359 -> 359-360=1; abs(359-0) same;;; abs(0-181)=181 -> 181-360=-179
    
    big_diff_idx = np.where(np.abs(diffs) > 180)
    diffs[big_diff_idx] = np.abs(diffs[big_diff_idx]) - 360
    
    mae = np.median(np.abs(diffs))
    
    return mae



def evaluate_timeseries_gridsearch(timeseries1, timeseries2, total_acc, 
                                    window,lag,model_type,head_name,split,
                                    moving_resting = 'moving_resting',
                                    moving_resting_threshold = 0.5,
                                    series_length = 3000,
                                    padding = 25):
    """
    Creates linear model with gridsearch of hyperparams


    Parameters
    ----------

    timeseries1 : ndarray of shape (n_samples, n_tetrodes)
        Neural data

    timeseries2 : ndarray of shape (n_samples, )
        Head data (yaw, roll, or pitch)

    total_acc : ndarray of shape (n_samples, )
        The total acceleration signal, for conditioning by movement state (moving vs resting)

    window : int
        Size of the temporal window (in samples)

    lag : int
        Offset between neural and head matrices (in samples)

    model_type : {'ridge_phi', 'sgd', 'gamma'}
        Type of model to create

    head_name : {'yaw','roll','pitch'}
        The variable we're decoding 

    split : float
        How to split the data for training/testing. Fraction for test_size (0-1)

    moving_resting : string
        Whether to model only during moving or resting state, or whole session (default)

    
    moving_resting_threshold : float, default = 0.5
        Threshold of total_acc for moving vs resting condition

    series_length : int, default = 3000
        Number of samples in odd/even chunks for splitting the data

    padding : int, default = 25
        Number of samples to discard between odd and even chunks

    
    Returns
    ------- 

    grid_search : object
        The model after fitting

    X_train, X_test : ndarrays of shape (samples, n_tetrodes)
        Training and testing neural data matrices

    y_train, y_test : ndarrays of shape (samples,)
        Training and testing head data vectors

    """

    ## make sure the head data is at least 2D:
    if timeseries2.ndim == 1:
        timeseries2 = np.atleast_2d(timeseries2).T

    ## z-score the neural data:
    scaler = StandardScaler().fit(timeseries1)
    timeseries1 = scaler.transform(timeseries1)

    

    
    print('THE SHAPES OF timeseries1, timeseries2 ==== ',timeseries1.shape, timeseries2.shape)
    print('Test size = %f' % split)
    
    ### create the temporal window and time offset (lag)
    X,y = make_timeseries_instances(timeseries1,timeseries2,window,lag) # X: neural data, y: head data 

    ## do the same with total acc to find indexes for moving/resting state:
    _,total_acc = make_timeseries_instances(timeseries1,total_acc,window,lag) 


    ## get moving or resting indexes [after taking the window]: 

    if moving_resting == 'moving':
        idxs = np.where(total_acc > moving_resting_threshold)[0]
    elif moving_resting == 'resting':
        idxs = np.where(total_acc <= moving_resting_threshold)[0]
    else:
        idxs = np.arange(y.shape[0])

    X = X[idxs] 
    y = y[idxs]




    #X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size = split, random_state=42 ) ## split the data

    ## take 30-sec series_length odd/even windows, discarding the 0.25-sec padding:
    X, y = timeseries_shuffler(X, y, series_length, padding)
    

    ## split into training/testing chunks: 
    X_train, X_test, y_train, y_test = split_data(X, y, split,standardize=False)
    

    # flatten X from 3D to 2D:
    X_train = X_train.reshape([X_train.shape[0],X_train.shape[1] * X_train.shape[2]])
    X_test = X_test.reshape([X_test.shape[0],X_test.shape[1] * X_test.shape[2]])

    # make pipeline:
    pipeline = Pipeline([('reg',  Ridge()   ) ]) # MultiOutputRegressor( ### ('scl', StandardScaler()),

    # make param grid:
    # parameters = {'reg__estimator__alpha' : 10.0**-np.arange(1,7),
    #               'reg__estimator__loss' : ['squared_loss','huber', 'epsilon_insensitive',  'squared_epsilon_insensitive'],
    #               'reg__estimator__penalty' : ['l2', 'l1', 'elasticnet'],
    #               'reg__estimator__learning_rate' : ['constant','optimal','invscaling','adaptive'] }
    # parameters = {'reg__estimator__alpha' : [0.01],
    #               'reg__estimator__loss' : ['huber'],
    #               'reg__estimator__penalty' : ['elasticnet'],
    #               'reg__estimator__learning_rate' : ['invscaling'] }
    # parameters = {'reg__estimator__alpha' : [0.1, 0.01, 0.001, 0.0001],
    #                'reg__estimator__loss' : ['squared_loss','huber', 'epsilon_insensitive'],
    #                'reg__estimator__penalty' : ['l2', 'l1', 'elasticnet'],
    #                'reg__estimator__learning_rate' : ['optimal','invscaling','adaptive'] }
    
    parameters = { 'reg__alpha' : np.logspace(2, 6, 5) } ## np.logspace(-6, 6, 13)

    # establish the grid search:
    grid_search = GridSearchCV(estimator=pipeline,param_grid=parameters, n_jobs=1, verbose=2)

    print('Fitting Grid Search Model...')

    start = time.time()

    grid_search.fit(X_train,y_train)

    end = time.time()

    print('Fit model. It took %.2f seconds.' %  (end - start) )


    print('Best params:\n', grid_search.best_params_)


    return grid_search, X_train, X_test, y_train, y_test



def evaluate_timeseries_lstm(timeseries1, timeseries2,total_acc, 
                             window,lag,split,
                             moving_resting='moving_resting',
                             moving_resting_threshold = 0.5,
                             series_length = 3000,
                             padding = 25):
    """
    Creates LSTM model 


    Parameters
    ----------

    timeseries1 : ndarray of shape (n_samples, n_tetrodes)
        Neural data

    timeseries2 : ndarray of shape (n_samples, 3)
        Head data (yaw, roll, and pitch)

    total_acc : ndarray of shape (n_samples, )
        The total acceleration signal, for conditioning by movement state (moving vs resting)

    window : int
        Size of the temporal window (in samples)

    lag : int
        Offset between neural and head matrices (in samples)

    split : float
        How to split the data for training/testing. Fraction for test_size (0-1)

    moving_resting : string
        Whether to model only during moving or resting state, or whole session (default)

    
    moving_resting_threshold : float, default = 0.5
        Threshold of total_acc for moving vs resting condition

    series_length : int, default = 3000
        Number of samples in odd/even chunks for splitting the data

    padding : int, default = 25
        Number of samples to discard between odd and even chunks

    
    Returns
    ------- 

    lstm : object
        The model after fitting

    X_train, X_test : ndarrays of shape (samples, n_tetrodes)
        Training and testing neural data matrices

    y_train, y_test : ndarrays of shape (samples, 3)
        Training and testing head data matrixes


    y_predict, y_predict_shuffle : ndarrays of shape (samples, 3)
        Predictions from the model and shuffled data 


    """

    if timeseries2.ndim == 1:
        timeseries2 = np.atleast_2d(timeseries2).T

    ## z-score X:
    scaler = StandardScaler().fit(timeseries1)
    timeseries1 = scaler.transform(timeseries1)


    #### downsample by a factor of N (10):
    N = 10 # averaging every 10 elements
    tmp_len = timeseries1.shape[0] - timeseries1.shape[0] % N
    num_neurons = timeseries1.shape[1]
    num_classes = timeseries2.shape[1]
    timeseries1 = np.atleast_3d(timeseries1[:tmp_len]).reshape(-1,N,num_neurons).mean(axis=1)
    timeseries2 = np.atleast_3d(timeseries2[:tmp_len]).reshape(-1,N,num_classes).mean(axis=1)        
    total_acc = np.atleast_3d(total_acc[:tmp_len]).reshape(-1,N,1).mean(axis=1)        


    ## scale the head data (y) -- and start calling it y_z -> y_filt -> y
    y_scaler = StandardScaler()
    y_z = y_scaler.fit_transform(timeseries2)

    ## median filter on y:
    y_filt = np.empty_like(y_z)
    for i in range(y_filt.shape[1]):
        y_filt[:,i] = medfilt(y_z[:,i],25)




    

    
    print('THE SHAPES OF timeseries1, timeseries2 ==== ',timeseries1.shape, y_filt.shape)
    print('Test size = %f' % split)
    ### alternative:
    X,y = make_timeseries_instances(timeseries1,y_filt,window,lag) # get the window

    _,total_acc = make_timeseries_instances(timeseries1,total_acc,window,lag) # get the window for total acc (so we can take moving/resting idx)

    ## get moving or resting indexes [after taking the window]: 
    if moving_resting == 'moving':
        idxs = np.where(total_acc > moving_resting_threshold)[0]
    elif moving_resting == 'resting':
        idxs = np.where(total_acc <= moving_resting_threshold)[0]
    else:
        idxs = np.arange(y.shape[0])

    y = y[idxs]
    X = X[idxs]



    #X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size = split, random_state=42 ) ## split the data

    X, y = timeseries_shuffler(X, y, series_length, padding)
    
     
    X_train, X_test, y_train, y_test = split_data(X, y, split,standardize=False)
    

    ###### make the model:
    

    print('********************************** Making LSTM Model **********************************')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    X_train = torch.Tensor(X_train)
    #X_train = X_train.to(device)
    y_train = torch.Tensor(y_train) 


    X_test = torch.Tensor(X_test)
    #X_test = X_test.to(device)
    y_test = torch.Tensor(y_test) 



    batch_size = 128 #64 #  64 # 
    dataset = TensorDataset(X_train,y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 20
    learning_rate = 0.001 #0.01

    seq_length = X.shape[1] # 11 

    input_size = X.shape[2] #1
    hidden_size = 100 # 50 # was 20

    num_layers = 2

    num_classes = y.shape[1]


    

    lstm = LSTMModel(input_dim=input_size, hidden_dim=hidden_size, layer_dim=num_layers, 
                     output_dim=num_classes, dropout_prob=0.75, device = device)

    lstm.to(device)

    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)


    # Train the model

    losses = []
    running_loss = 0.0

    for epoch in range(num_epochs):
                    
            
        for i, (x_batch, y_batch) in enumerate(dataloader):

            
            
            y_batch_pred = lstm(x_batch.to(device))

            #loss = criterion(y_batch_pred, y_batch.to(device))
            loss = modified_mse(y_batch_pred, y_batch.to(device))
            
            #loss = mse_velocity(y_batch_pred, y_batch.to(device))
            
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            running_loss += loss.item()
            print_batch = 250
            if i % print_batch == print_batch-1:    # print every 2000 mini-batches
                print(f'{epoch}, {i}. loss: {running_loss / print_batch:.3f}')
                running_loss = 0.0
            
        

    with torch.no_grad():
        lstm.eval()
        y_predict = lstm(X_test.to(device)).to('cpu').detach().numpy()

        y_predict_shuffle = lstm(torch.Tensor(np.random.permutation(X_test)).to(device) ).to('cpu').detach().numpy()


    ## median filter for prediction:
    for i in range(num_classes):
        y_predict[:,i] =  medfilt(y_predict[:,i],25) ## was 101
        y_predict_shuffle[:,i] = medfilt(y_predict_shuffle[:,i],25)

    y_predict = y_scaler.inverse_transform(y_predict)
    y_predict_shuffle = y_scaler.inverse_transform(y_predict_shuffle)

    y_test = y_test.to('cpu').detach().numpy() # [::10]
    y_test = y_scaler.inverse_transform(y_test)



    return lstm, X_train, X_test, y_train, y_test, y_predict, y_predict_shuffle




def evaluate_timeseries_crossfile(timeseries1, timeseries2, window,lag,model_type,head_name,split):
      
    if timeseries2.ndim == 1:
        timeseries2 = np.atleast_2d(timeseries2).T    

    scaler = StandardScaler().fit(timeseries1)
    timeseries1 = scaler.transform(timeseries1)
    
    print('THE SHAPES OF timeseries1, timeseries2 ==== ',timeseries1.shape, timeseries2.shape)    
    X, y = make_timeseries_instances(timeseries1, timeseries2, window, lag)
    print('Shapes of X and y after making timeseries instance:', X.shape,y.shape)

    
    
    print('Reshaping X')
    #### X's are (time, window, channels), e.g. (13085, 200, 16). Reshape for the linear model:
    
    X = X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))    
    


    return X,y


def start_decoding_lstm(X,y,window,lag,model_type,
                        save_dir,split,
                        moving_resting=None,total_acc=None):


    """
    Starts the decoding process for the LSTM model


    Parameters
    ----------

    X : ndarray of shape (n_samples, n_tetrodes)
        Neural data

    y : ndarray of shape (n_samples,)
        Head data (yaw, roll, or pitch)

    window : int
        Size of the temporal window (in samples)

    lag : int
        Offset between neural and head matrices (in samples)

    model_type : {'ridge_phi', 'sgd', 'gamma'}
        Type of model to create

    head_name : {'yaw','roll','pitch'}
        The variable we're decoding 

    save_dir : string
        Path where to save the results

    split : float
        How to split the data for training/testing. Fraction for test_size (0-1)

    moving_resting : string
        Whether to model only during moving or resting state, or whole session (default)

    total_acc : ndarray of shape (n_samples,)
        Vector with overall movement (total acceleration) used to determine moving/resting state indexes


    Returns
    -------

    y_test : ndarray of shape (samples,)
        Test portion of the head variable

    y_predict : ndarray of shape (samples,)
        The model's prediction of the head variable

    """



    print('______________Decoding with an LSTM______________')

    model, X_train, X_test, y_train, y_test, prediction, y_predict_shuffle = evaluate_timeseries_lstm(X,y,total_acc,window,lag,
            model_type=model_type,split=split,moving_resting=moving_resting)



    
    torch.save(model.state_dict(), save_dir[:save_dir.find('.npz')] + '.pt')
    

    #################### plot ################

    

    r2 = []
    r2_shuffle = []
    r = []
    r_shuffle = []

    mse = []
    mse_shuffle = []

    mae = []
    mae_shuffle = []

    r_circ =  []
    r_circ_shuffle =  []

    r2_circ = []
    r2_circ_shuffle = []

    #### to GET BEHAVIORAL COVERAGE of all the angles:
    roll_bins = np.arange(-90,90,10)
    pitch_bins = np.arange(-180,180,10)
    yaw_bins = np.arange(0,360,10)
    all_bins = {'yaw' : yaw_bins,'roll': roll_bins,'pitch': pitch_bins}

    num_classes = 3

    f,axarr = plt.subplots(1,num_classes,dpi=600,figsize=(6,2),sharey=False) 

    for i,y_key in enumerate(all_bins.keys()):

        axarr[i].scatter(y_test[:,i],prediction[:,i],s=2,lw=0,alpha=0.5,c='k')

        _r2 = r2_score(y_test[:,i],prediction[:,i])
        _r2_shuffle = r2_score(y_test[:,i],y_predict_shuffle[:,i])

        _r = stats.pearsonr(y_test[:,i],prediction[:,i])[0]
        _r_shuffle = stats.pearsonr(y_test[:,i],y_predict_shuffle[:,i])[0]

        ## MSE:
        _mse = modified_mse_numpy(y_test[:,i], prediction[:,i])
        _mse_shuffle = modified_mse_numpy(y_test[:,i], y_predict_shuffle[:,i])

        ## MAE
        _mae = modified_mae(y_test[:,i], prediction[:,i])
        _mae_shuffle = modified_mae(y_test[:,i], y_predict_shuffle[:,i])


        _r_circ = circcorrcoef(np.deg2rad(y_test[:,i]),np.deg2rad(prediction[:,i]) ) if i == 0 else np.nan 
        _r_circ_shuffle = circcorrcoef(np.deg2rad(y_test[:,i]),np.deg2rad(y_predict_shuffle[:,i]) ) if i == 0 else np.nan

        _r2_circ = circular_r2_score(np.deg2rad(y_test[:,i]),np.deg2rad(prediction[:,i]) ) if i == 0 else np.nan 
        _r2_circ_shuffle = circular_r2_score(np.deg2rad(y_test[:,i]),np.deg2rad(y_predict_shuffle[:,i]) ) if i == 0 else np.nan

        r2.append(_r2)
        r2_shuffle.append(_r2_shuffle)
        
        r.append(_r)
        r_shuffle.append(_r_shuffle)

        mse.append(_mse)
        mse_shuffle.append(_mse_shuffle)

        mae.append(_mae)
        mae_shuffle.append(_mae_shuffle)

        r_circ.append(_r_circ)
        r_circ_shuffle.append(_r_circ_shuffle)        

        r2_circ.append(_r2_circ)
        r2_circ_shuffle.append(_r2_circ_shuffle)

        axarr[i].set_title('R^2 = %.2f\nPearson r = %.2f' % (_r2,_r ) )
        axarr[i].set_xlabel(y_key)
        axarr[i].set_aspect('equal','datalim')
        lim = axarr[i].get_xlim()[1]
        
        lims = [0,lim]
        axarr[i].plot(lims,lims,color='k',linestyle='dashed',alpha=0.5)

        ## save histogram of true and predicted stuff:
        H, xedges, yedges = np.histogram2d(x=y_test[:,i],y=prediction[:,i],bins=all_bins[y_key])
        np.savez(save_dir[:save_dir.find('.npz')] + '_histogram.npz',
                    hist=H,xedges=xedges,yedges=yedges)

    


    f.savefig(save_dir[:save_dir.find('.npz')] + '.pdf')

    plt.close(f)

    np.savez(save_dir,r=r,r2=r2,r_shuffle=r_shuffle,r_circ=r_circ,r_circ_shuffle=r_circ_shuffle,r2_circ=r2_circ,r2_circ_shuffle=r2_circ_shuffle,
            mse=mse,mse_shuffle=mse_shuffle,mae=mae,mae_shuffle=mae_shuffle,r2_shuffle=r2_shuffle,
        window=window, lag=lag,split=split) #, weights=coefs, ) # y_predict=y_predict,y_test=y_test,y_predict_shuffle=y_predict_shuffle,

    
   

    return y_test,prediction





def start_decoding(X,y,window,lag,model_type,
                    head_name,save_dir,split,
                    moving_resting=None,total_acc=None):
    """
    Starts the decoding process for the linear models


    Parameters
    ----------

    X : ndarray of shape (n_samples, n_tetrodes)
        Neural data

    y : ndarray of shape (n_samples,)
        Head data (yaw, roll, or pitch)

    window : int
        Size of the temporal window (in samples)

    lag : int
        Offset between neural and head matrices (in samples)

    model_type : {'ridge_phi', 'sgd', 'gamma'}
        Type of model to create

    head_name : {'yaw','roll','pitch'}
        The variable we're decoding 

    save_dir : string
        Path where to save the results

    split : float
        How to split the data for training/testing. Fraction for test_size (0-1)

    moving_resting : string
        Whether to model only during moving or resting state, or whole session (default)

    total_acc : ndarray of shape (n_samples,)
        Vector with overall movement (total acceleration) used to determine moving/resting state indexes


    Returns
    -------

    y_test : ndarray of shape (samples,)
        Test portion of the head variable

    y_predict : ndarray of shape (samples,)
        The model's prediction of the head variable

    """

    if head_name == 'yaw':

        if model_type == 'ridge_phi' or model_type == 'sgd' or model_type == 'gamma':
            phi = True
        else:
            phi = False
    else:
        phi = False




    if phi: ## convert y to sin + cos vectors    
        y = np.exp(1j * np.deg2rad(y) )  ### (e.g. 1234100,1)    
        y = np.vstack([np.real(y),np.imag(y)]).T #(e.g. 1234100,2)


    
    
    model, X_train, X_test, y_train, y_test = evaluate_timeseries_gridsearch(X,y,total_acc,window,lag,
            model_type=model_type,head_name=head_name,split=split,moving_resting=moving_resting)
    


    ## clear out for memory issues:
    #X = y = None

    
    y_predict = model.predict(X_test)

    y_predict_shuffle = model.predict(np.random.permutation(X_test)) #shuffle_model.predict(X_test)
    




    if phi: ## convert the complex # to angles for y_test and y_predict and shuffled predictions:
        
        y_predict = np.rad2deg(np.arctan2(y_predict[:,1],y_predict[:,0])) ## arctan2 takes the y position first, then x
        negs = np.where(y_predict<0)[0]
        y_predict[negs] += 360 ## correct for wrong quadrant...

        y_test = np.rad2deg(np.arctan2(y_test[:,1],y_test[:,0]))
        negs = np.where(y_test<0)[0]
        y_test[negs] += 360

        y_predict_shuffle = np.rad2deg(np.arctan2(y_predict_shuffle[:,1],y_predict_shuffle[:,0]))
        negs = np.where(y_predict_shuffle<0)[0]
        y_predict_shuffle[negs] += 360

        r_circ =  circcorrcoef(np.deg2rad(y_test),np.deg2rad(y_predict) )

        r_circ_shuffle =  circcorrcoef(np.deg2rad(y_test),np.deg2rad(y_predict_shuffle) )

    else:
        r_circ = np.nan
        r_circ_shuffle = np.nan    


    ############ evaluate the model w/ pearson's r, r^2, MSE, and MAE: ###################

    ## ravel before doing pearonr: (convert from e.g. (10000,1) to (10000,))
    r = stats.pearsonr(y_test.ravel(),y_predict.ravel())[0]
    print('r =', r)
    r_shuffle = stats.pearsonr(y_test.ravel(),y_predict_shuffle.ravel())[0]
    
    r2 = r2_score(y_test,y_predict)

    ## MSE:
    mse = modified_mse_numpy(y_test.ravel(),y_predict.ravel())
    mse_shuffle = modified_mse_numpy(y_test.ravel(),y_predict_shuffle.ravel())

    ## MAE
    mae = modified_mae(y_test.ravel(),y_predict.ravel())
    mae_shuffle = modified_mae(y_test.ravel(),y_predict_shuffle.ravel())

    print('Saving results from %s. r=%.2f, r2=%.2f, r_shuffle=%.2f, r_circ=%.2f, mse=%.2f. mse_shuffle=%.2f, mae=%.2f. mae_shuffle=%.2f, ' % (save_dir,
                                                        r,r2,r_shuffle,r_circ,mse,mse_shuffle,mae,mae_shuffle) )

    ### extract model coefficients, if you want to save those:
    if model_type == 'sgd':

        estimators = model.best_estimator_.named_steps['reg'].estimators_
        
        coefs = np.asarray([estimator.coef_ for estimator in estimators ]).T ## features x number of estimators

    elif model_type == 'ridge_phi':

        coefs = model.best_estimator_.named_steps['reg'].coef_.T ## features x dims
    

    else: ## i.e. the old models w/o pipelines or grid search

        coefs = np.asarray(model.coef_).T ## features x number of estimators


    np.savez(save_dir,r=r,r2=r2,r_shuffle=r_shuffle,r_circ=r_circ,r_circ_shuffle=r_circ_shuffle,
            mse=mse,mse_shuffle=mse_shuffle,mae=mae,mae_shuffle=mae_shuffle,
        window=window, lag=lag,split=split,coefs=coefs) #, weights=coefs, ) # y_predict=y_predict,y_test=y_test,y_predict_shuffle=y_predict_shuffle,

    
    if model_type != 'lstm':
        joblib.dump(model,save_dir[:save_dir.find('.npz')] + '.p')
    else:
        torch.save(model.state_dict(), save_dir[:save_dir.find('.npz')] + '.pt')

    print(head_name,y_test.shape,y_predict.shape)
    
    return y_test,y_predict
   


def decode_cross_file(X,y,window,lag,model_type,head_name,save_dir,split,moving_resting=None,total_acc=None,rat=None,fil=None):

    if head_name == 'yaw':

        if model_type == 'ridge_phi' or model_type == 'sgd' or model_type == 'gamma':
            phi = True
        else:
            phi = False
    else:
        phi = False


    if phi: ## convert y to sin + cos vectors    
        y = np.exp(1j * np.deg2rad(y) )  ### (e.g. 1234100,1)    
        y = np.vstack([np.real(y),np.imag(y)]).T #(e.g. 1234100,2)


    ## call y y_test b/c we'll compare it to our y_predicted which is predicted from X using the loaded coefs
    X,y_test = evaluate_timeseries_crossfile(X,y,window=window,lag=lag,
            model_type=model_type,head_name=head_name,split=split)

    if phi:
        y_test = np.rad2deg(np.arctan2(y_test[:,1],y_test[:,0]))
        negs = np.where(y_test<0)[0]
        y_test[negs] += 360  
    
    ### iterate over this rat's other files, loading the coefs from the npz results files

    data_dir = '/n/holyscratch01/cox_lab/Users/guitchounts/ephys/hd_decoding_results/expt_ridge_mua_movingresting_012921/data/%s' % rat # /GRat36_636508403748459062_dark_window_0_lag_0_split_0_movingresting_chunk_0_yaw.npz



    test_paths = [f for f in glob('%s_*_*_window_0_lag_0_split_0_movingresting_chunk_*_%s*' % (data_dir,head_name)) if not f.endswith('histogram.npz')]
    
    print(test_paths)

    for test_path in test_paths:

        print('Cross-decoding on test path = ', test_path)

        split_name = test_path.split('_')
    
        fil_test = [f for f in split_name if '636' in f][0]

        chunk_test = [split_name[i+1] for i,f in enumerate(split_name) if f == 'chunk' ][0]

        coefs = np.load(test_path)['coefs']
        print('Loaded coefs.shape = ', coefs.shape)

        ## if 
        y_predict = np.dot(X, coefs)

        

        if phi: ## convert the complex # to angles for y_test and y_predict and shuffled predictions:
            
            y_predict = np.rad2deg(np.arctan2(y_predict[:,1],y_predict[:,0])) ## arctan2 takes the y position first, then x
            negs = np.where(y_predict<0)[0]
            y_predict[negs] += 360 ## correct for wrong quadrant...



            # y_predict_shuffle = np.rad2deg(np.arctan2(y_predict_shuffle[:,1],y_predict_shuffle[:,0]))
            # negs = np.where(y_predict_shuffle<0)[0]
            # y_predict_shuffle[negs] += 360

            r_circ =  circcorrcoef(np.deg2rad(y_test),np.deg2rad(y_predict) )

            # r_circ_shuffle =  circcorrcoef(np.deg2rad(y_test),np.deg2rad(y_predict_shuffle) )

        else:
            r_circ = np.nan
            # r_circ_shuffle = np.nan    


        ############ evaluate the model w/ pearson's r and MSE: ###################

        ## ravel before doing pearsonr: (convert from e.g. (10000,1) to (10000,))
        r = stats.pearsonr(y_test.ravel(),y_predict.ravel())[0]
        print('r =', r)
        
        
        r2 = r2_score(y_test,y_predict)

        ## MSE:
        mse = modified_mse_numpy(y_test.ravel(),y_predict.ravel())
        #mse_shuffle = modified_mse(y_test.ravel(),y_predict_shuffle.ravel())

        print('Saving results from %s. r=%.2f, r2=%.2f, r_circ=%.2f. mse=%.2f,' % (save_dir,
                                                            r,r2,r_circ,mse) )

        

        _save_dir = '_fil_test_%s_chunk_test_%s' % (fil_test,chunk_test)

        final_save_dir = save_dir[:save_dir.find('.npz')] + _save_dir + save_dir[save_dir.find('.npz'):]

        np.savez(final_save_dir,r=r,r2=r2,r_circ=r_circ,mse=mse,
            window=window, lag=lag,split=split,coefs=coefs,rat=rat,fil_test=fil_test,chunk_test=chunk_test)

        print(head_name,y_test.shape,y_predict.shape)
    
    return y_test,y_predict











