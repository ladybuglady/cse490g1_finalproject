'''
This file prepares the dataset by splitting raw ECoG data into 1-second-long events and ommitting seizure-contaminated events.
'''


import os
os.environ["OMP_NUM_THREADS"] = "1" #avoid multithreading if have Anaconda numpy
import numpy as np
import mne
import sys
sys.path.append('/home/zeynep/ForkedRepos/ECoG_processing_tutorial')
from Steve_libPreprocess import *
from ecog_spectrogram_utils import *
import pandas as pd
import pdb
from datetime import datetime as dt
import argparse
import h5py

#Set parameters
patient_id = 'a0f66459'
day = 4
wrist_side = 'r'
epoch_times = [-5,5]
pad_val = 0.5 #(sec) used for padding each side of epochs and cropped off during power computation (~0.5 sec is enough)
num_events = 200 #number of events to select
min_time_between_events = 1.2 #seconds (used for move event detection)
#events_lp = '/data2/users/satsingh/share/events_trimmed_r2_velocities/'


# Load in ECoG data
# Data can be found here: https://figshare.com/articles/dataset/Naturalistic_ECoG_move_v_rest/13010546
loadpath = '/data1/ecog_project/derived/processed_ecog/'+patient_id+'/full_day_ecog/'+\
            patient_id+'_fullday_'+str(day)+'.h5'

chan_info = pd.read_hdf(loadpath,key='chan_info',mode='r')
chan_inds_keep = np.nonzero(np.logical_not(np.isnan(chan_info.iloc[-1,:].values)))[0]

fin = h5py.File(loadpath,"r")
Fs=int(fin['f_sample'][()]) #Hz
ecog_data,inFrameInds=ecog_loadData(fin,chan_inds_keep,-1)
n_channels=ecog_data.shape[0]
ecog_data[np.isnan(ecog_data)] = 0 #remove NaN values (set to 0)


#Add electrode names (fairly arbitrary) and locations
ch_names=[]
for i in range(n_channels):
    ch_names.append('EEG'+str(i))
elec_locs = chan_info.loc[['X','Y','Z'],:].transpose().values[chan_inds_keep,:]


# TAKING CARE OF BLACKLIST...
listpath = '/nas/ecog_project/human/VideoBlackWhiteList/VideoBlackWhiteList_'
blacklist_df = pd.read_csv(listpath+patient_id+'_latest.tsv', sep='\t')
print(blacklist_df)

# https://www.codegrepper.com/code-examples/python/panda+-+subset+based+on+column+value
isbad = blacklist_df['Blacklist'].notnull()
baddies = blacklist_df[isbad]

print(blacklist_df)
print("ok")
print(baddies)

# now we have a list of the video filenames for all the bad eggs!

vse_lp = '/nas/ecog_project/derived/processed_ecog/'+patient_id+'/full_day_ecog/vid_start_end_merge.csv'
vse_df = pd.read_csv(vse_lp,header=0)
vse_df.query('merge_day=='+str(day)) #show the video start times corresponding to the current merge day

Fs_ecog = 500 #Hz (ECoG sampling rate)
vse_df['year'],vse_df['month'],vse_df['day'] = 2000, 1, day
vse_df['datetime'] =  pd.to_datetime(vse_df[['year','month','day','hour','minute','second','microsecond']])

ecog_samples = []
for i in range(vse_df.shape[0]):
    time_from_midnight = ((vse_df['datetime'].iloc[i] - pd.to_datetime({'year':[2000],'month':[1],
                                                                        'day':[day]}))/ np.timedelta64(1, 's'))[0]
    ecog_samples.append(int(time_from_midnight*Fs_ecog))
vse_df['ecog_samples'] = ecog_samples
vse_df.query('merge_day=='+str(day))

day_vse_df = vse_df.query('merge_day=='+str(day))
bad_times = []
print(day_vse_df)
for i, video in enumerate(day_vse_df['filename']):
    if video in baddies['Video Filename'].values:
        #print(video)
        row = day_vse_df[day_vse_df['filename']==video].index.values[0]
        time = vse_df.iloc[row]['datetime']
        bad_times.append(str(time)[11:])
print("Videos that were blacklisted: ", len(bad_times))

bad_times_df = pd.DataFrame(bad_times)
bad_times_df.columns = ['time']
bad_times_df['time'] = pd.to_timedelta(bad_times_df['time'], unit='ms')
print(bad_times_df)

# for patient_id = 'a0f66459, day = 4

bad_time_ecog_indices = event_ecog_indices(bad_times_df['time'],patient_id,day,loadpath,Fs)
print(bad_time_ecog_indices)

#Create event matrix for MNE (Cols: 0: sample number, 1: value from, 2: value to)
total_trial_num = len(bad_time_ecog_indices)
bad_events=np.zeros([total_trial_num,3])
trial_count=0

for i in range(len(bad_time_ecog_indices)):
    bad_events[trial_count,0]=bad_time_ecog_indices[i]
    bad_events[trial_count,2]=1
    trial_count+=1

#bad_events = np.asarray(bad_time_ecog_indices)
bad_events = bad_events.astype('int')
print(bad_events)

# EPOCHING TIME SEQUENCE
# https://mne.tools/dev/auto_tutorials/preprocessing/20_rejecting_bad_data.html

ch_names=[]
for i in range(n_channels):
    ch_names.append('EEG'+str(i))
elec_locs = chan_info.loc[['X','Y','Z'],:].transpose().values[chan_inds_keep,:]

def epoch_ECoG_sequential_MNE_with_blacklist(ecog_data, ch_names, Fs, bad_events):
    print("In the notebook")
    info = mne.create_info(ch_names=ch_names, sfreq=Fs, ch_types='eeg')
    raw = mne.io.RawArray(ecog_data, info)
    #del ecog_data
    onsets = bad_events[:, 0] / raw.info['sfreq'] - 0.25
    durations = [0.5] * len(bad_events)
    descriptions = ['bad'] * len(bad_events)
    bad_annot = mne.Annotations(onsets, durations, descriptions,
                              orig_time=raw.info['meas_date'])
    raw.set_annotations(bad_annot)
    epoched_data = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True, overlap=0.0, 
                                                id=1, verbose=None, reject_by_annotation=True)
    #del raw
    return epoched_data

epoched_seq_data_bl = epoch_ECoG_sequential_MNE_with_blacklist(ecog_data, ch_names, Fs, bad_events)

epoched_seq_data_bl.drop_bad()

epochstotal = len(epoched_seq_data_bl)

# Save 5 minutes of ECoG Epochs

# Grab from middle
mid = int(epochstotal/2)
# 2.5 minutes from each side:
half = int(epochstotal/24/6/2/2)
fivemindata = epoched_seq_data_bl[mid-half:mid+half]
print(len(fivemindata))

# Save 5 minutes of ECoG Epochs
fivemindata.save("/home/zeynep/ForkedRepos/cse490g1_finalproject/"+patient_id+"_"+str(len(fivemindata))+"_"+"day"+str(day)+"_5min_epo.fif")

print("Saved!")