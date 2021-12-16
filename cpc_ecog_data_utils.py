''' This module contains code to handle data '''

import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
from matplotlib import pyplot as plt
from make_ecog_data import *

class ECoGHandler(object):

    ''' Provides a convenient interface to manipulate ECoG data '''

    def __init__(self):

        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_val.shape)
        print(self.y_val.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)
        
        # we also need to keep track of the indices of each set since each event is unique
        self.trainstart = 0
        self.trainend = len(self.X_train)
        self.valstart = len(self.X_train)
        self.valend = len(self.X_train) + len(self.X_val)
        self.teststart = len(self.X_val)
        self.testend = len(self.X_val) + len(self.X_test)

    def load_dataset(self):
        
        sbj = 'a0f66459'
        lp = '/data1/users/zeynep/sequential_epoch' # change this to route to saved epochs data
        train_data, test_data = get_seq_epochs(sbj, lp); # 20 minutes of data
        print("Train data: ", train_data.shape)

        # The labels are the event number? What else could labels be??
        valcount = int(len(train_data)/3)
        
        X_train = train_data[:-valcount]
        y_train = np.arange(len(X_train))
        
        X_val = train_data[-valcount:]
        y_val = np.arange(len(X_val))
        
        X_test = test_data
        y_test = np.arange(len(test_data))
        #y_test = np.arange(len(train_data), len(train_data) + len(test_data))

        # We reserve the last 1/3 of training examples for validation.
        valcount = int(len(X_train)/3)
        

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_batch_by_labels(self, subset, labels, rescale=True):

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

        # Find samples matching labels
        idxs = []
        #print("THIS IS WHAT Y (event numbers) LOOKS LIKE, SHOULD BE A RANGE OF INDICES: ", y)
        for i, label in enumerate(labels):
            idx = np.where(y == int(label))[0]
            #print("np.where(y == int(label)): ", np.where(y == int(label)))
            #print("idx: ", idx)
            #idx_sel = np.random.choice(idx, 1)[0]
            #print("label: ", label)
            idx_sel = idx[0] # there is only 1 unique instance 
            
            idxs.append(idx_sel)
            #idxs.append(int(label))
        
        #idxs = labels.astype(int)
        # Retrieve events
        
        #print("Original size of X: ", X.shape) # (559, 64, 500)
        #print("idxs length: ", len(idxs))
        #print("idxs: ", idxs)
        #print("Selected size of X: ", X[np.array(idxs), :, :].shape) # (256, 500)
        
        '''
        Set:  valid
        Start:  559
        End:  838
        
        Original size of X:  (279, 64, 500)
        idxs length:  256
        idxs:  [564 565 566 567 568 569 570 571 755 756 757 758 759 760 761 762 691 692
         693 694 695 696 697 698 817 818 819 820 821 822 823 824 581 582 583 584
         585 586 587 588 611 612 613 614 615 616 617 618 723 724 725 726 727 728
         729 730 760 761 762 763 764 765 766 767 641 642 643 644 645 646 647 648
         703 704 705 706 707 708 709 710 643 644 645 646 647 648 649 650 636 637
         638 639 640 641 642 643 559 560 561 562 563 564 565 566 609 610 611 612
         613 614 615 616 731 732 733 734 735 736 737 738 763 764 765 766 767 768
         769 770 818 819 820 821 592 653 630 597 712 713 714 715 809 721 676 818
         826 827 828 829 612 794 780 809 670 671 672 673 814 766 788 624 633 634
         635 636 662 751 811 785 656 657 658 659 808 725 588 806 664 665 666 667
         609 639 692 697 742 743 744 745 832 627 592 821 611 612 613 614 764 685
         794 601 798 799 800 801 669 808 709 741 653 654 655 656 566 743 754 754
         713 714 715 716 651 719 619 680 609 610 611 612 836 563 651 729 580 581
         582 583 559 692 792 570 604 605 606 607 592 637 777 632 807 808 809 810
         605 679 772 797]
         '''
        
        #batch = X[np.array(idxs), 0, :].reshape((len(labels), 28, 28))
        batch = X[np.array(idxs), :, :]

        return batch.astype('float32'), labels.astype('int32')

    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'valid':
            y_len = self.y_val.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]

        return y_len

class SortedECoGGenerator(object):

    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, rescale=False):
        
        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.rescale = rescale

        # Initialize ECoG dataset
        self.ecog_handler = ECoGHandler()
        self.n_samples = self.ecog_handler.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size
        #print("Size of y: ", self.ecog_handler.get_n_samples(subset))
        #print("Samples in total: ", self.n_samples)
        #print("Samples in each batch: ", self.n_batches)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        
        '''
        start, end = self.ecog_handler.getrange(self.subset)
        print("Set: ", self.subset)
        print("Start: ", start)
        print("End: ", end)
        '''
        start = 0
        end = self.ecog_handler.get_n_samples(self.subset)
        
        # Build sentences
        event_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        #print("Size of event_labels: ", event_labels.shape) # (32, 8)
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        positive_samples_n = self.positive_samples
        for b in range(self.batch_size):

            # Set ordered predictions for positive samples
            
            seed = np.random.randint(start, end) # validation: a number between 559 and 838
            sentence = np.mod(np.arange(seed, seed + self.terms + self.predict_terms), end)

            if positive_samples_n <= 0:

                # Set random predictions for negative samples
                # Each predicted term draws a number from a distribution that excludes itself
                numbers = np.arange(start, end)
                predicted_terms = sentence[-self.predict_terms:]
                for i, p in enumerate(predicted_terms):
                    predicted_terms[i] = np.random.choice(numbers[numbers != p], 1)
                sentence[-self.predict_terms:] = np.mod(predicted_terms, end)
                sentence_labels[b, :] = 0

            # Save sentence
            event_labels[b, :] = sentence

            positive_samples_n -= 1

        # Retrieve actual ecog events
        events, _ = self.ecog_handler.get_batch_by_labels(self.subset, event_labels.flatten(), self.rescale)

        # Assemble batch
        
        #print("Dimensions for set:", self.subset) # (256, 64, 500)
        #print("shape: ", events.shape)
        #print("dim1: ", events.shape[1])
        #print("dim2: ", events.shape[2])
        # cannot reshape array of size 8192000 into shape (32,8,1,64)
        events = events.reshape((self.batch_size, self.terms + self.predict_terms, events.shape[1], events.shape[2]))
        #print(events.shape) # (32, 8, 64, 500)
        x_events = events[:, :-self.predict_terms, ...]
        y_events = events[:, -self.predict_terms:, ...]

        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)
        #print("x: ", x_events[idxs, ...].shape) # (32, 4, 64, 500)
        #print("y: ", y_events[idxs, ...].shape) # (32, 4, 64, 500)
        return [x_events[idxs, ...], y_events[idxs, ...]], sentence_labels[idxs, ...]




def plot_sequences(x, y, labels=None, output_path=None):

    ''' Draws a plot where sequences of numbers can be studied conveniently '''

    images = np.concatenate([x, y], axis=1)
    n_batches = images.shape[0]
    n_terms = images.shape[1]
    counter = 1
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            plt.subplot(n_batches, n_terms, counter)
            plt.imshow(images[n_b, n_t, :, :, :])
            plt.axis('off')
            counter += 1
        if labels is not None:
            plt.title(labels[n_b, 0])

    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()


if __name__ == "__main__":

    # Test SortedNumberGenerator
    ag = SortedECoGGenerator(batch_size=8, subset='train', terms=4, positive_samples=4, predict_terms=4, rescale=False)
    for (x, y), labels in ag:
        plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted.png')
        break

