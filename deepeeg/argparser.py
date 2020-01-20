import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(['--model', '-M'], metavar='model', type=str,
                        help='the type of the DL network to be used (CNN/RNN)', default='CNN', required=False)

    parser.add_argument(['-f'], metavar='data file', type=str, default='./data.csv', help='path to file with EEG data',
                        required=False)

    return parser.parse_args()
