import argparser
import dataloader
import modelbuilder
import numpy as np

def train_model(model_type, input_shape):
    conv_layers = [{'filters': 25, 'pool_size': 3, 'kernel_size': 2, 'activation_func': 'relu'},
                   {'filters': 50, 'pool_size': 3, 'kernel_size': 2, 'activation_func': 'relu'},
                   {'filters': 100, 'pool_size': 2, 'kernel_size': 2, 'activation_func': 'relu'},
                   {'filters': 200, 'pool_size': 2, 'kernel_size': 2, 'activation_func': 'relu'}]
    dense_layers = [{'num_units': 1, 'activation_func': 'sigmoid'}]

    # https://www.researchgate.net/publication/309873852_Single-trial_EEG_classification_of_motor_imagery_using_deep_convolutional_neural_networks
    # https://www.researchgate.net/publication/315096373_Deep_learning_with_convolutional_neural_networks_for_brain_mapping_and_decoding_of_movement-related_information_from_the_human_EEG
    model = modelbuilder.build_cnn_model(conv_layers, dense_layers, False, input_shape)

    model.compile(optimizer='adam', loss='binary_crossentropy')

    print(model.summary())


def predict():
    pass

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