# Project: Using Contrastive Predictive Coding for Self-Supervised Electrocorticography Decoding
This is the final project for CSE490 G1 Deep Learning, Fall 2021

## Abstract
In this project I will be experimenting with a contrastive-predictive-coding (CPC) approach to decode wrist movement from electrocorticography (ECoG) brain data via self-supervised learning (SSL). This project is a part of my research at the UW Brunton Lab for Brain, Behavior, and Data Science. 

[SSL](https://bdtechtalks.com/2020/03/23/yann-lecun-self-supervised-learning/) is a type of machine learning capable of identifying meaningful patterns in large quantities of data without requiring explicit labels. It uses a pretask to learn representations of a dataset such that a downstream task may be trained on without needing to annotate a dataset. My primary intention is to decode wrist movement events from a single anonymous participant's ECoG data. The ECoG data, provided by the Brunton Lab, is electrical activity recorded from a participant's cerebral cortex. I devised a pretask that utilizes [CPC](https://arxiv.org/abs/1807.03748) inspired by Oord et. al.'s successes with the approach on image, text, and speech data in 2018. Solving this task forces the model to pick up representations about ECoG, so that it can (in theory) decode whether a wrist is moving or resting without needing to explicitly provide examples of what wrist moving and wrist resting ECoG looks like. Labels are provided for the testing portion of the downstream in order to assess the model's capability.

After 300 epochs, the pretask accuracy was able to reach 96.09% training accuracy and 67.19% validation accuracy. After 30 epochs, the downstream reached a 71.60% training accuracy and 78.75% validation accuracy.

## Video
## Introduction
Computational neuroscience is a field with an abundance of data, but only a small amount available for brain analysis. This is because in order to do any useful brain analysis, researchers must establish what brain activity looks like during specific human behaviors, or else the brain activity renders meaningless. Hence, there is a widening gap between raw data and processed data.

An example of a behavior that can be decoded from ECoG brain data is wrist movement. In a supervised learning approach, we would feed the model examples of ECoG data when the participant's wrist is moving and when it is resting: X would be ECoG events, and Y would be a move/rest label. The Brunton Lab did successfully devise such a model, but my team and I spent months watching hours of video footage to decipher human behavior, and then trace it back to the ECoG to create a label. We thought there must be a more efficient approach-as, if the model can successfully differentiate these two behaviors in ECoG given our labels, then surely, some distinction in the data must exist naturally between brain activity during rest and brain activity during movement. Hence, we decided to experiment with different self-supervised learning pretasks to get the model to learn the most useful patterns without us telling it what those patterns are. 

I was intrigued by Oord et. al.'s contrastive predictive coding approach because the predictive nature of the task seemed most useful in finding patterns in temporal data like brain signals. 

## Related Work
[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) is the paper that inspired me to adapt the CPC approach to my work. The repository for this paper is at: https://github.com/davidtellez/contrastive-predictive-coding

[Self-supervised representation learning from electroencephalography signals](https://arxiv.org/abs/1911.05419) by Banville et. al. published in 2019 utilizes a relative positioning pretask to decode electroencephalography (EEG) brain signals. The relative positioning pretask generates pairs of EEG signals and then determines if they are closer or farther away from each other in time. Though this paper was useful for its own downstream, which was decoding sleep stages, I found that the relative positioning pretask wasn't ideal for decoding wrist movement since it is a shorter lived event. Thus, whether two points are near or far in time is not very helpful since the event itself is barely fractions of a second long.

[Generalized neural decoders for transfer learning across participants and recording modalities](https://www.biorxiv.org/content/10.1101/2020.10.30.362558v1.full.pdf) describes the supervised learning model to decode ECoG at the Brunton Lab. This model is capable of generalizing to multiple participants, and now my current research works on utilizing this HTNet to generalize successfully with self-supervised learning.

[Contrastive Representation Learning for Electroencephalogram Classification](http://proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf) by Mohsenvand et. al. published in 2020 performs signal transformation tasks on EEG data via a contrastive learning approach. This inspired our prior work to utilize signal transformations for the pretask.

[BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data](https://www.frontiersin.org/articles/10.3389/fnhum.2021.653659/full) published in June of 2021 was an intriguing find due to its use of EEG data. The paper attempts to merge CPC and a Masked Language Model (MLM), but it reports that the overall findings seem to be ineffective relative to other techniques that already exist. This approach can be a pretraining step.

[SEQ-CPC : Sequential Contrastive Predictive Coding for Automatic Speech Recognition](https://ieeexplore.ieee.org/document/9413645) published in June of 2021 will be useful for future work in utilizing the sequential nature of temporal data. This paper reports that "experimental results on WSJ corpus show that SEQ-CPC achieves the best performance than CPC and NCE which is the contrastive objective used in wav2vec", though this doesn't necessarily mean that it will be successful on ECoG data.

## Approach
### Data Preparation
The first thing I had to do for this experiment was prepare the raw data. The ECoG data is obtained from 64 electrodes installed onto a patient's cerebral cortex, so there are 64 channels. The electrical activity is recorded at 500Hz, so each second of data contains 500 values. Thus, the ECoG data is stored inside of an [Epochs](https://mne.tools/dev/auto_tutorials/epochs/10_epochs_overview.html) object of dimensions (n, 64, 500). I used the [MEG + EEG Analysis & Visualization](https://mne.tools/stable/index.html) package to split the raw ECoG data into sequences of continuous 1-second long Events.

In addition, this dataset is obtained from epilepsy patients from a hospital, so some of the ECoG data contains seizure activity. These instances must be eliminated from the Events list, so I traced their occurence time in the ECoG data and marked all events overlapping these times as 'bad' so that they would be ommitted from the final dataset.

All of this is done inside the EpochECoGEvents.py file. However, some imported files and directories containing information regarding seizures are classified and only available to Brunton Lab staff, so you cannot run this file. Luckily though, I have provided the .fif file that contains the final epoched data in this repository. Unfortunately, the largest .fif file I could upload here for you to demo yourself is 5 minutes worth of data, but the results I report below are all from ~20 minutes of data!

### Epoching Experiments
Even though our dataset does contain 'move' or 'rest' labels, it is imperative to be able to blindly select a segment of data from an arbitrary point to stay true to the self-supervised learning philosophy. If we cherry pick our data to be solely snippets of ECoG that we already know are labeled events, then we are not evaluating the most honest effectiveness of the pretask. Thus, I conducted a sequential-epoching mini experiment where I tested out different amounts of continuous ECoG data used in a pretask (5 seconds, 1 minute, 5 minutes, 15 minutes, 1 hour) to see how many data points are really necessary to make a difference in performance. This is important to know because in the future if we want to apply this technology to brain-computer interfaces, we would want to keep data collection to a minimum.

Prior to the CPC approach, I had been investigating the effectiveness of a signal transformation pretask. Since this is a simple baseline pretask, I used it as a means to guage approximately how much data is needed for pretask and downstream accuracies. Observe the signal transformation pretask accuracies for different amounts of data:

![test_accuracies_minimal_to_15](https://user-images.githubusercontent.com/67766355/146273462-4c47d2ce-f4ee-4e2a-ac96-dfdbe0f83654.png)

This chart shows us that when tasked with classifying signal transformations, after 5 minutes of data the accuracy doesn't get much better.
Now, let's take a look at the downstream:

![DownstreamAccsDotPlot](https://user-images.githubusercontent.com/67766355/146274057-631a021f-681b-4fa0-8b99-79116269a961.png)

For specifically subject a0f, more than 5 minutes of data used to train the pretask model yields around the same downstream accuracy (but keep in mind the actual downstream task trains on much more data). After a certain point, more data in the pretask seemed to deter the performance of the downstream, and I think this is because its forcing the pretask to overfit to a very specific segment of time, so for next time it might be a better idea to select events with some space in between them to generalize better.

Each event of data is 1 second long. This means that 5 minutes of data is originally 300 events (but for this particular segment of 5 minutes there were 2 seziure-contaminants, so we had 298 events). The signal transformation pretask takes a second of ECoG data and applies a scaling transformation to one copy, a negation transformation to another copy, a flipping transformation to another copy, and then adding noise to a final copy. So there are, in total, 5 versions of each event. Thus, there are 298 * 5 = 1490 events.

Since the contrastive predictive coding pretask doesn't make copies of the data, to obtain approximately 1490 events, I decided to start off with 20 minutes of data.

### Pretask
Here, I will detail the mechanism of the pretask. This pretask is adapted from the https://github.com/davidtellez/contrastive-predictive-coding repository using MNIST image data. 

The contrastive predictive coding mechanism utilizes contrastive learning and predictive coding. This means that the pretask modifies the data in some way to create contrasts, such that there "positive" and "negative" samples. Then, it aims to distinguish the positive samples from the negative samples. It is predictive, because an autogregressive model continually updates the predictions via the use of predicition errors.

We first create examples of predictions. Let an example of a prediction be called a "sentence". A sentence will be a sequence of events. A sentence is the sum of 4 given terms, and 4 predicted terms. A sentence may take on the form of:

[Event 4, Event 5, Event 6, Event 7, Event 8, Event 9, Event 10, Event 11]


Each event is a second of ECoG data. The first 4 events will always be in a correct predicted order, as the second of ECoG data denoted as 'Event 5' comes after the second of ECoG data denoted as 'Event 4' and so forth.

The last 4 events are "predictions". If the last 4 events are in a correctly predicted order, the sentence is a positive sample. If they are not, the sentence is a negative sample. Here are examples of positive samples:

[Event 5, Event 6, Event 7, Event 8, Event 9, Event 10, Event 11, Event 12]

[Event 37, Event 38, Event 39, Event 40, Event 41, Event 42, Event 43, Event 44]

[Event 1198, Event 1199, Event 1200, Event 0, Event 1, Event 2, Event 3, Event 4]


Notice that in the last example, since we have a total of 1200 events, after the 1200th event, the next event is event 0. Since this is still in order, this is a positive sample. 

Here are some negative samples:
[Event 5, Event 6, Event 7, Event 8, Event 45, Event 109, Event 1, Event 800]

[Event 37, Event 38, Event 39, Event 40, Event 2, Event 999, Event 43, Event 1011]

[Event 1198, Event 1199, Event 1200, Event 0, Event 1012, Event 782, Event 9, Event 940]



### Downstream


## Results
## Discussion
