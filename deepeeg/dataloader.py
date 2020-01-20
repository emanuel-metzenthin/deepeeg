import mne
import os
import numpy as np
import pandas as pd

def load_brain_vis_file(filename,label,train,val):
    raw = mne.io.read_raw_brainvision(filename)
    events, _ = mne.events_from_annotations(raw) 
    data = pd.DataFrame(raw.get_data())
    data_A = data.iloc[:128, :].T 
    BAD_STIMULI = [7]
    N_TIMESTEPS = 1000

    for event in events[1:]: # Delete first entry (as 100sec long)
        if event[2] not in BAD_STIMULI:
            train.append(data_A.iloc[event[0]:event[0] + N_TIMESTEPS, :])
            val.append(label)

    return

def read_brainvis_from_directory(folder):
    train = []
    val = []
    labels = pd.read_csv(os.path.join(folder, 'labels.csv'), sep=';', header=0, decimal=',')
    labels = labels.loc[:, ['Pair', 'coact012_bin1234']]
    labels = labels.rename(columns = {'coact012_bin1234': 'Score'})
    labels.astype({'Score': 'float64'})
    minScore = labels.Score.min()
    maxScore = labels.Score.max()
    labels.Score = (labels.Score - minScore) / (maxScore - minScore)
    labels.Score = np.where(labels.Score > 0.5, 1, 0)
    for filename in os.listdir(folder):
        if filename.endswith("Z.vhdr"): 
            ids = filename.split('_')
            label = labels.loc[labels['Pair'] == 'P'+ ids[1]].at[0,'Score']
            print('Filename:' + filename)
            load_brain_vis_file(filename,label,train,val)
            continue

 
    train = np.dstack(train)
    train = np.moveaxis(train, -1, 0)
    val = np.asarray(val)
    print(train.shape)
    print(val.shape)
    return train,val
