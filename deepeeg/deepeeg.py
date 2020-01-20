from deepeeg  import modelbuilder
import logging
from keras.models import model_from_json

class DeepEEG():
    # TODO make configurable
    # https://www.researchgate.net/publication/309873852_Single-trial_EEG_classification_of_motor_imagery_using_deep_convolutional_neural_networks
    # https://www.researchgate.net/publication/315096373_Deep_learning_with_convolutional_neural_networks_for_brain_mapping_and_decoding_of_movement-related_information_from_the_human_EEG
    conv_layers = [{'filters': 25, 'pool_size': 3, 'kernel_size': 2, 'activation_func': 'relu'},
                   {'filters': 50, 'pool_size': 3, 'kernel_size': 2, 'activation_func': 'relu'},
                   {'filters': 100, 'pool_size': 2, 'kernel_size': 2, 'activation_func': 'relu'},
                   {'filters': 200, 'pool_size': 2, 'kernel_size': 2, 'activation_func': 'relu'}]
    dense_layers = [{'num_units': 1, 'activation_func': 'sigmoid'}]

    def train_cnn(self, input_shape, save_model_to=None, save_weights_to=None):
        model = modelbuilder.build_cnn_model(self.conv_layers, self.dense_layers, input_shape)

        model.compile(optimizer='adam', loss='binary_crossentropy')

        logging.info(model.summary())

        if save_model_to and save_weights_to:
            model_json = model.to_json()
            with open(save_model_to, "w") as json_file:
                json_file.write(model_json)
                logging.info("Saved model to disk: {}".format(save_model_to))

            model.save_weights(save_weights_to)
            logging.info("Saved model weights to disk: {}".format(save_weights_to))

        self.model = model

        return model

    def load_model(self, model_file, weights_file):
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights_file)

        self.model = loaded_model

        return loaded_model