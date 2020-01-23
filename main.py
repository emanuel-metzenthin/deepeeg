import argparser
from deepeeg.dataloader import load_data_from_dir
from deepeeg.deepeeg import DeepEEG
import logging

def main():
    logging.getLogger().setLevel(logging.INFO)
    arguments = argparser.parse_arguments()

    MODE = arguments.mode

    X_train, y_train, X_val, y_val = load_data_from_dir('./data/deepeeg-format/raw')

    deepeeg = DeepEEG()

    if MODE == argparser.TRAIN_MODE:
        if arguments.model == argparser.CNN:
            model = deepeeg.train_cnn(X_train, y_train, X_val, y_val, save_model_to='./model.json', save_weights_to='./weights.hdf5')

        elif arguments.model == argparser.RNN:
            pass

    elif MODE == argparser.PREDICT_MODE:
        pass

if __name__ == '__main__':
    main()