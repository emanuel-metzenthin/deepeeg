import mne
import os
import numpy as np
import pandas as pd

def load_brainvis_file(filename, label, train_x, train_y, drop_stimuli=[7], drop_channels=[], n_timesteps=1000):
    raw = mne.io.read_raw_brainvision(filename)
    events, _ = mne.events_from_annotations(raw) 
    data = pd.DataFrame(raw.get_data())
    data = data.drop(drop_channels, axis=0).T
    print(data.shape)

    for event in events[1:]: # Delete first entry (as 100sec long)
        if event[2] not in drop_stimuli:
            train_x.append(data.iloc[event[0]:event[0] + n_timesteps, :])
            train_y.append(label)

    return

def read_brainvis_from_directory(folder):
    train_x = []
    train_y = []

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
            load_brainvis_file(filename, label, train_x, train_y)
            continue

    train_x = np.dstack(train_x)
    train_x = np.moveaxis(train_x, -1, 0)
    train_y = np.asarray(train_y)
    print(train_x.shape)
    print(train_y.shape)

    return train_x, train_y
