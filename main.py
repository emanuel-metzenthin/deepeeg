import argparser
from deepeeg.dataloader import read_brainvis_from_directory
from deepeeg.deepeeg import DeepEEG
import logging

def main():
    logging.getLogger().setLevel(logging.INFO)
    arguments = argparser.parse_arguments()

    MODE = arguments.mode

    X_train, y_train, X_val, y_val = read_brainvis_from_directory('./data')

    logging.info('Shape : ', X_train.shape, y_train, X_val, y_val)

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