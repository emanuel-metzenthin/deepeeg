import mne
import os
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split


def load_and_transform_brainvis_data(path):  # TODO implement
    path = './'
    drop_channels = [128, 129, 130, 131]
    drop_stimuli = [7]
    n_timesteps = 1000

    for filename in os.listdir(path):
        if filename.endswith("Z.vhdr"):
            ids = filename.split('_')
            pair = labels.loc[labels['Pair'] == 'P' + ids[1]]
            label = pair.at[pair.index[0], 'Score']
            print('Filename:' + filename)

            raw = mne.io.read_raw_brainvision(filename)
            events, _ = mne.events_from_annotations(raw)
            data = pd.DataFrame(raw.get_data())
            data = data.drop(drop_channels, axis=0).T

            data['trial'] = 0
            data['label'] = label

            final_data = pd.DataFrame(columns=data.columns)

            trial_count = 0

            for event in events[1:]:  # Delete first entry (as 100sec long)
                if event[2] not in drop_stimuli:
                    data['trial'].iloc[event[0]:event[0] + n_timesteps] = trial_count
                    final_data = final_data.append(data.iloc[event[0]:event[0] + n_timesteps])
                    trial_count += 1

            final_data.to_csv('./deepeeg-format/{}_deep.csv'.format(filename.split('.')[0]), index=False)

            continue

#
# Expects a directory with .csv files of the format
# rows: measurements
# cols: sensors
#       + 'trial' col (to which trial person the measurement value belongs)
#       + 'label' (class/regression label associated with measurement value)
#
def load_data_from_dir(path):  # TODO? Add file pattern param
    train_x = []
    train_y = []

    for filename in os.listdir(path):
        if(not filename.endswith('.csv')):
            continue
        df = pd.read_csv(os.path.join(path, filename))
        cols = df.columns
        trial_ids = df['trial'].unique()
        for id in trial_ids:
            trial = df[df['trial'] == id]
            train_x.append(trial.loc[:, ~df.columns.isin(['trial', 'label'])])
            train_y.append(trial['label'][trial.index[0]])

    train_x = np.dstack(train_x)
    train_x = np.moveaxis(train_x, -1, 0)
    train_y = np.asarray(train_y)

    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

    logging.info('X_train.shape : {}, X_val.shape : {}, y_train: {}, y_val: {}'.format(X_train.shape, X_val.shape, y_train, y_val))

    return X_train, y_train, X_val, y_val
