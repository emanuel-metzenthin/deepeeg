import os
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

#
# Expects a directory with .csv files of the format
# rows: measurements
# cols: sensors
#       + 'trial' col (to which trial person the measurement value belongs)
#       + 'label' (class/regression label associated with measurement value)
#
def load_data_from_dir(path, test_size=0.33):  # TODO? Add file pattern param
    train_x = []
    train_y = []

    for filename in os.listdir(path):
        if(not filename.endswith('.csv')):
            continue
        df = pd.read_csv(os.path.join(path, filename))
        trial_ids = df['trial'].unique()
        for id in trial_ids:
            trial = df[df['trial'] == id]
            train_x.append(trial.loc[:, ~df.columns.isin(['trial', 'label'])])
            train_y.append(trial['label'][trial.index[0]])

    train_x = np.dstack(train_x)
    train_x = np.moveaxis(train_x, -1, 0)
    train_y = np.asarray(train_y)

    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=test_size, random_state=42)

    logging.info('X_train.shape : {}, X_val.shape : {}, y_train.shape: {}, y_val.shape: {}'.format(X_train.shape, X_val.shape, y_train.shape, y_val.shape))

    return X_train, y_train, X_val, y_val
