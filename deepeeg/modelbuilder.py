from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, BatchNormalization, Dropout, LSTM

class Builder():
    optimizer = 'adam'
    loss_function = 'binary_crossentropy'
    metrics = ['accuracy']

    def __init__(self):
        self.model = Sequential()

    def build(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)

        return self.model


class CNNBuilder(Builder):
    def with_optimizer(self, optimizer):
        self.optimizer = optimizer

    def with_loss_function(self, loss):
        self.loss_function = loss

    def add_normalized_conv_layer(self, num_filters, pool_size=2, kernel_size=2, activation_func='relu', input_shape=''):
        if input_shape != '':
            self.model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation=activation_func, input_shape=input_shape))
            self.model.add(MaxPooling1D(pool_size=pool_size))
            self.model.add(BatchNormalization())
            return

        self.model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,activation=activation_func))
        self.model.add(MaxPooling1D(pool_size=pool_size))
        self.model.add(BatchNormalization())

    def add_dropout_layer(self, rate=0.4):
        self.model.add(Dropout(rate))

    def add_flatten_layer(self):
        self.model.add(Flatten())

    def add_dense_layer(self, num_units, activation_func):
        self.model.add(Dense(num_units, activation=activation_func))

class LSTMBuilder(Builder):
    def add_LSTM_layer(self, num_units, input_shape=None, return_sequences=False):
        if input_shape:
            self.model.add(LSTM(num_units, input_shape=input_shape, return_sequences=return_sequences))
        else:
            self.model.add(LSTM(num_units, return_sequences=return_sequences))

    def add_dense_layer(self, num_units, activation_func):
        self.model.add(Dense(num_units, activation=activation_func))