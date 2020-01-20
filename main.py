import argparser
import dataloader
import modelbuilder
import numpy as np

def main():
    arguments = argparser.parse_arguments()

    MODE = arguments.mode

    # data = dataloader.load_data(arguments.file)

    if MODE == argparser.TRAIN_MODE:
        input_shape = (arguments.num_timesteps, arguments.num_sensors)
        train_model(arguments.model, input_shape)
    elif MODE == argparser.PREDICT_MODE:
        pass

if __name__ == '__main__':
    main()