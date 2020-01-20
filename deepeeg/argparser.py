import argparse

TRAIN_MODE = 'train'
PREDICT_MODE = 'predict'
CNN = 'cnn'
RNN = 'rnn'

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=[TRAIN_MODE, PREDICT_MODE], default='train', help='wether to train a new model or predict using a previously trained one')

    parser.add_argument('--model', '-m', type=str, choices=[CNN, RNN],
                        help='the type of the DL network to be used (cnn/rnn), default: \'cnn\'', default=CNN)

    parser.add_argument('--file', '-f', type=str, default='./data.csv', help='path to file with EEG data, default: \'./data.csv\'')

    parser.add_argument('--num_timesteps', '-nt', type=int, default=1000, help='number of values in the EEG timeseries per training sample')

    parser.add_argument('--num_sensors', '-ns', type=int, default=64, help='number of sensors per trial participant')

    return parser.parse_args()
