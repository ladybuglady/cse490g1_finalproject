
import os
import numpy as np
from tqdm import tqdm
import scipy.ndimage
import scipy
import sys
from tensorflow.keras import utils as np_utils
from matplotlib import pyplot as plt
from make_ecog_data import *



class DownstreamHandler(object):
    def __init__(self):
        
        sbj = 'a0f66459'
        lp = '/data1/users/stepeter/cnn_hilbert/ecog_data/xarray/'

        # load in the data
        
        norm_rate = 0.25
        test_day = 'last'
        n_chans_all=64
        tlim=[-1,1]
        n_folds = 1
        folds = 3
        
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, nb_classes = self.load_and_split_data(sbj, lp, n_chans_all, test_day, tlim, n_folds)
        
        '''
        Should be of shape:
        SHAPE OF X:  (n, 1, 64, 501)
        SHAPE OF Y:  (n, 2)
        '''
        
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_val.shape)
        print(self.y_val.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)
        
    def load_dataset(self, pats_ids_in, lp, n_chans_all=64, test_day=None, tlim=[-1,1], event_types=['rest','move']):
        
        if not isinstance(pats_ids_in, list):
            pats_ids_in = [pats_ids_in]
        sbj_order,sbj_order_test = [],[]
        X_test_subj,y_test_subj = [],[] #placeholder vals
        
        #Gather each subjects data, and concatenate all days
        for j in tqdm(range(len(pats_ids_in))):
            pat_curr = pats_ids_in[j]
            ep_data_in = xr.open_dataset(lp+pat_curr[:3]+'_ecog_data.nc')
            ep_times = np.asarray(ep_data_in.time)
            time_inds = np.nonzero(np.logical_and(ep_times>=tlim[0],ep_times<=tlim[1]))[0]
            n_ecog_chans = (len(ep_data_in.channels)-1)

            if test_day == 'last':
                test_day_curr = np.unique(ep_data_in.events)[-1] #select last day
            else:
                test_day_curr = test_day

            if n_chans_all < n_ecog_chans:
                n_chans_curr = n_chans_all
            else:
                n_chans_curr = n_ecog_chans



            days_all_in = np.asarray(ep_data_in.events)

            if test_day is None:
                #No test output here
                days_train = np.unique(days_all_in)
                test_day_curr = None
            else:
                days_train = np.unique(days_all_in)[:-1]
                day_test_curr = np.unique(days_all_in)[-1]
                days_test_inds = np.nonzero(days_all_in==day_test_curr)[0]

            #Compute indices of days_train in xarray dataset
            days_train_inds = []
            for day_tmp in list(days_train):
                days_train_inds.extend(np.nonzero(days_all_in==day_tmp)[0])

            #Extract data and labels
            dat_train = ep_data_in[dict(events=days_train_inds,channels=slice(0,n_chans_curr),
                                        time=time_inds)].to_array().values.squeeze()
            labels_train = ep_data_in[dict(events=days_train_inds,channels=ep_data_in.channels[-1],
                                           time=0)].to_array().values.squeeze()
            sbj_order += [j]*dat_train.shape[0]

            if test_day is not None:
                dat_test = ep_data_in[dict(events=days_test_inds,channels=slice(0,n_chans_curr),
                                           time=time_inds)].to_array().values.squeeze()
                labels_test = ep_data_in[dict(events=days_test_inds,channels=ep_data_in.channels[-1],
                                              time=0)].to_array().values.squeeze()
                sbj_order_test += [j]*dat_test.shape[0]

            #Pad data in electrode dimension if necessary
            if (len(pats_ids_in) > 1) and (n_chans_all > n_ecog_chans):
                dat_sh = list(dat_train.shape)
                dat_sh[1] = n_chans_all
                #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
                X_pad = np.zeros(dat_sh)
                X_pad[:,:n_ecog_chans,...] = dat_train
                dat_train = X_pad.copy()

                if test_day is not None:
                    dat_sh = list(dat_test.shape)
                    dat_sh[1] = n_chans_all
                    #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
                    X_pad = np.zeros(dat_sh)
                    X_pad[:,:n_ecog_chans,...] = dat_test
                    dat_test = X_pad.copy()

            #Concatenate across subjects
            if j==0:
                X_subj = dat_train.copy()
                y_subj = labels_train.copy()
                if test_day is not None:
                    X_test_subj = dat_test.copy()
                    y_test_subj = labels_test.copy()
            else:
                X_subj = np.concatenate((X_subj,dat_train.copy()),axis=0)
                y_subj = np.concatenate((y_subj,labels_train.copy()),axis=0)
                if test_day is not None:
                    X_test_subj = np.concatenate((X_test_subj,dat_test.copy()),axis=0)
                    y_test_subj = np.concatenate((y_test_subj,labels_test.copy()),axis=0)

        sbj_order = np.asarray(sbj_order)
        sbj_order_test = np.asarray(sbj_order_test)
        print('Data loaded!')

        return X_subj,y_subj,X_test_subj,y_test_subj,sbj_order,sbj_order_test

    
    def load_and_split_data(self, sbj, lp, n_chans_all, test_day, tlim, n_folds):
        
        X,y,x_test,y_test,sbj_order_all,sbj_order_test_last = self.load_dataset(sbj, lp,
                                                                  n_chans_all=n_chans_all,
                                                                  test_day=test_day, tlim=tlim)
        #split data for test and val, and convert to tensorflow version
        nb_classes = len(np.unique(y))
        order_inds = np.arange(len(y))
        np.random.shuffle(order_inds)
        X = X[order_inds,...]
        y = y[order_inds]
        order_inds_test = np.arange(len(y_test))
        np.random.shuffle(order_inds_test)
        x_test = x_test[order_inds_test,...]
        y_test = y_test[order_inds_test]
        y2 = np_utils.to_categorical(y-1)
        y_test2 = np_utils.to_categorical(y_test-1)
        X2 = np.expand_dims(X,1)
        X_test2 = np.expand_dims(x_test,1)

        split_len = int(X2.shape[0]*0.2)
        last_epochs = np.zeros([n_folds,2])

        val_inds = np.arange(0,split_len)+(0*split_len)
        #take all events not in val set
        train_inds = np.setdiff1d(np.arange(X2.shape[0]),val_inds) 

        x_train = X2[train_inds,...]
        y_train = y2[train_inds]
        x_val = X2[val_inds,...]
        y_val = y2[val_inds]
    
        return x_train, y_train, x_val, y_val, X_test2, y_test2, nb_classes
    
    def get_batch(self, subset, batch_size, rescale=False):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Random choice of samples
        print()
        #print("GETTING BATCH...")
        #print("original shape of x: ", X.shape)
        #print("original shape of y: ", y.shape)
        idx = np.random.choice(X.shape[0], batch_size)
        batch = X[idx, :, :]
        batch = np.squeeze(batch)
        batch = batch[:, :, :500]

        # Process batch
        #batch = self.process_batch(batch, batch_size, image_size, color, rescale)

        # ECoG label
        labels = y[idx]

        return batch.astype('float32'), labels.astype('int32')
    
    
    
    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'valid':
            y_len = self.y_val.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]

        return y_len

class ECoGGenerator(object):

    # Data generator providing ECoG data 

    def __init__(self, batch_size, subset, rescale=False):

        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.rescale = rescale

        # Initialize ECoG dataset
        self.ecog_handler = DownstreamHandler()
        self.n_samples = self.ecog_handler.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Get data
        x, y = self.ecog_handler.get_batch(self.subset, self.batch_size, self.rescale)

        # Convert y to one-hot
        #y_h = np.eye(2)[y]
        
        print("SHAPE OF X: ", x.shape)
        print("SHAPE OF Y: ", y.shape)


        return x, y


