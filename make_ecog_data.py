import numpy as np
import math, pdb, glob, os, sys, natsort, h5py, mne
import cv2
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import xarray as xr

np.random.seed(seed=42)

'''
The file loads up the .fif file we prepared and splits the data into train and test sets.
'''


#read in speech fif files
def get_seq_epochs(sbj, lp, crop_val=3, time='20min'):
    # The epoching file was created from the ECoG processing tutorial
    
    # MAY HAVE TO CHANGE FILENAMES TO ACCESS YOUR .FIF FILE
    if time == 'long':
        print(lp+'/'+sbj+'/*_epo-*.fif')
        f_load = natsort.natsorted(glob.glob(lp+'/'+sbj+'/*_epo-*.fif'))
        print(glob.glob(lp+'/'+sbj+'/*_epo-*.fif'))
        print("for sjb",sbj, "f_load is",f_load)
        
        for i, f in enumerate(f_load):
            epstotal=f_load[i].split('/')[-1].split('_')[0]
            print("EPS TOTAL: ", epstotal)
            eps = int(epstotal)/len(f_load) * (i+1)
            print("EXTRACTED NUMBER:")
            print(f_load[i].split('/')[-1].split('_')[0])
            eps_to_file[eps] = f_load[i]
            print("File: ", f, "List: ", eps_to_file)
            print("eps_to_file: ", eps_to_file)
        max_eps, min_eps = max(eps_to_file, key=eps_to_file.get), min(eps_to_file, key=eps_to_file.get)
        print("max eps: ", max_eps)
        print("min eps: ", min_eps)
        if crop_val > 0:
            epochs_train = mne.read_epochs(eps_to_file[max_eps]).crop(tmin=-1*crop_val, tmax=crop_val, include_tmax=True)
            epochs_test = mne.read_epochs(eps_to_file[min_eps]).crop(tmin=-1*crop_val, tmax=crop_val, include_tmax=True)
        else:
            epochs_train = mne.read_epochs(eps_to_file[max_eps])
        epochs_test = mne.read_epochs(eps_to_file[min_eps])
        train_dat = epochs_train.get_data()
        test_dat = epochs_test.get_data()
    else:
        f_load = natsort.natsorted(glob.glob(lp+'/'+sbj+'/*_epo-4.fif'))[0]
        epstotal=f_load.split('/')[-1].split('_')[0]
        if time == 'minimal':
            print("Minimal epochs")
            eps = int(epstotal)/8/3/4/3/5/10
        elif time == '1min':
            print("1 minute")
            eps = int(epstotal)/8/3/4/3/5
        elif time == '5min':
            print("5 minutes")
            f_load = natsort.natsorted(glob.glob(lp+'/'+sbj+'/'+'1198_day4_5min_epo.fif'))[0]
            eps=298
        elif time == '15min':
            print("15 minutes")
            eps = int(epstotal)/8/3/4
        elif time == '20min':
            print("20 minutes")
            f_load = natsort.natsorted(glob.glob(lp+'/'+sbj+'/'+'1198_day4_20min_epo.fif'))[0]
            eps = 1198
        elif time == '1hour':
            print("1 hour")
            eps = int(epstotal)/8/3
        print("for sjb",sbj, "f_load is",f_load)
        print("Eps: ", eps)
        max_eps = int(eps * 0.7)
        min_eps = int(eps * 0.3)
        print("max eps: ", max_eps)
        print("min eps: ", min_eps)
        all_epochs = None # this is taking a very long time because its 3 hours of events (3*60*60)
        if crop_val > 0:
            all_epochs = mne.read_epochs(f_load).crop(tmin=-1*crop_val, tmax=crop_val, include_tmax=True)
        else:
            all_epochs = mne.read_epochs(f_load)
        print("length of all: ",  all_epochs.get_data().shape)
        # start in the middle somewhere I guess
        print("how many total epochs obtained: ", len(all_epochs))
        mid = int(len(all_epochs)/2)
        print("mid: ", mid)
        #train_dat = all_epochs.get_data()[mid:(mid+max_eps)]
        #test_dat = all_epochs.get_data()[(mid+max_eps):(mid+max_eps+min_eps)]
        train_dat = all_epochs.get_data()[:(max_eps)]
        test_dat = all_epochs.get_data()[(max_eps):]
        # we want first 64 channels tho
        train_dat = train_dat[:, :64,:]
        test_dat = test_dat[:, :64,:]
        print("length of train: ",  train_dat.shape)
        print("length of test: ", test_dat.shape)    
    return train_dat, test_dat

