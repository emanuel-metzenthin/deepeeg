from deepeeg  import modelbuilder
import logging
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
import keras

class DeepEEG():
    # TODO make configurable
    # https://www.researchgate.net/publication/309873852_Single-trial_EEG_classification_of_motor_imagery_using_deep_convolutional_neural_networks
    # https://www.researchgate.net/publication/315096373_Deep_learning_with_convolutional_neural_networks_for_brain_mapping_and_decoding_of_movement-related_information_from_the_human_EEG
    conv_layers = [{'filters': 64, 'pool_size': 3, 'kernel_size': 3, 'activation_func': 'relu'},
                   {'filters': 128, 'pool_size': 3, 'kernel_size': 3, 'activation_func': 'relu'},
                   {'filters': 256, 'pool_size': 3, 'kernel_size': 3, 'activation_func': 'relu'},
                   {'filters': 512, 'pool_size': 3, 'kernel_size': 3, 'activation_func': 'relu'}]
    dense_layers = [{'num_units': 100, 'activation_func': 'relu'}, {'num_units': 1, 'activation_func': 'sigmoid'}]

    def train_cnn(self, X_train, y_train, X_val, y_val, save_model_to=None, save_weights_to=None):
        model = modelbuilder.build_cnn_model(self.conv_layers, self.dense_layers, (X_train.shape[1], X_train.shape[2]))

        keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        logging.info(model.summary())

        if save_model_to and save_weights_to:
            model_json = model.to_json()
            with open(save_model_to, "w") as json_file:
                json_file.write(model_json)
                logging.info("Saved model to disk: {}".format(save_model_to))

            model.save_weights(save_weights_to)
            logging.info("Saved model weights to disk: {}".format(save_weights_to))

        self.model = model

        hist = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), batch_size=5)

        print(hist.history['accuracy'])

        return model

    def load_model(self, model_file, weights_file):
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights_file)

        self.model = loaded_model

        return loaded_model