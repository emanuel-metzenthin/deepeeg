import logging
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from deepeeg import modelbuilder
from sklearn.metrics import classification_report


class DeepEEG():
    # https://www.researchgate.net/publication/309873852_Single-trial_EEG_classification_of_motor_imagery_using_deep_convolutional_neural_networks
    # https://www.researchgate.net/publication/315096373_Deep_learning_with_convolutional_neural_networks_for_brain_mapping_and_decoding_of_movement-related_information_from_the_human_EEG
    conv_layers = [{'filters': 64, 'pool_size': 3, 'kernel_size': 9, 'activation_func': 'relu'},
                   {'filters': 128, 'pool_size': 3, 'kernel_size': 9, 'activation_func': 'relu'},
                   {'filters': 256, 'pool_size': 3, 'kernel_size': 9, 'activation_func': 'relu'},
                   {'filters': 512, 'pool_size': 3, 'kernel_size': 9, 'activation_func': 'relu'}]
    dense_layers = [{'num_units': 100, 'activation_func': 'relu'}, {'num_units': 1, 'activation_func': 'sigmoid'}]

    rnn_layers = [{'num_units': 64}]#,
                  #{'num_units': 128},
                  #{'num_units': 256},
                  #{'num_units': 512}]
    dense_rnn_layers = [{'num_units': 100, 'activation_func': 'relu'}, {'num_units': 1, 'activation_func': 'sigmoid'}]

    def plot_training_history(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def init_cnn(self, data_shape, save_model_to=None):
        builder = modelbuilder.CNNBuilder()

        first = True

        for conv_layer in self.conv_layers:
            if first:
                builder.add_normalized_conv_layer(conv_layer['filters'], conv_layer['pool_size'],
                                                  conv_layer['kernel_size'],
                                                  conv_layer['activation_func'], data_shape)
                first = False
                continue
            builder.add_normalized_conv_layer(conv_layer['filters'], conv_layer['pool_size'], conv_layer['kernel_size'],
                                              conv_layer['activation_func'])

        builder.add_dropout_layer()

        builder.add_flatten_layer()

        for dense_layer in self.dense_layers:
            builder.add_dense_layer(dense_layer['num_units'], dense_layer['activation_func'])

        self.model = builder.build()

        if save_model_to:
            model_json = self.model.to_json()
            with open(save_model_to, "w") as json_file:
                json_file.write(model_json)
                logging.info("Saved model to disk: {}".format(save_model_to))

    def init_lstm(self, data_shape, save_model_to=None):
        builder = modelbuilder.LSTMBuilder()
        first = True
        for rnn_layer in self.rnn_layers:
            if first:
                if len(self.rnn_layers) == 1:
                    builder.add_LSTM_layer(rnn_layer['num_units'], input_shape=data_shape, return_sequences=False)
                    continue
                builder.add_LSTM_layer(rnn_layer['num_units'], input_shape=data_shape, return_sequences=True)
                first = False
                continue
            if rnn_layer == self.rnn_layers[len(self.rnn_layers) - 1]:
                builder.add_LSTM_layer(rnn_layer['num_units'], return_sequences=False)
                continue

            builder.add_LSTM_layer(rnn_layer['num_units'], return_sequences=True)
            builder.add_batch_normalization()

        for dense_layer in self.dense_rnn_layers:
            builder.add_dense_layer(dense_layer['num_units'], dense_layer['activation_func'])

        self.model = builder.build()

        if save_model_to:
            model_json = self.model.to_json()
            with open(save_model_to, "w") as json_file:
                json_file.write(model_json)
                logging.info("Saved model to disk: {}".format(save_model_to))

    def load_model(self, model_file, weights_file):
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights_file)

        self.model = loaded_model

        return loaded_model

    def train(self, X_train, y_train, X_val, y_val, epochs=25, batch_size=10, save_weights_to='.rnn-weights.hdf5'):
        checkpointer = ModelCheckpoint(filepath=save_weights_to, verbose=1, save_best_only=True)

        hist = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size,
                              callbacks=[checkpointer])

        self.plot_training_history(hist)

        self.model.load_weights(save_weights_to)

        logging.info('Training finished!')

        self.evaluate_model(X_val, y_val)

        return self.model

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test, batch_size=64, verbose=1)

        #y_pred_bool = np.argmax(y_pred, axis=1)
        y_pred_bool = np.around(y_pred, decimals=0)#
        print(y_pred_bool)

        print('Model evaluation:')
        print(classification_report(y_test, y_pred_bool))
