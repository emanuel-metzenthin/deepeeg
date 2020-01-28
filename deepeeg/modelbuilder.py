from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, BatchNormalization, Dropout, LSTM


def add_conv1D_layer(model, num_filters, pool_size=2, kernel_size=2, activation_func='relu', input_shape=''):
    if input_shape != '':
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation=activation_func, input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(BatchNormalization())
        return
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,activation=activation_func))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(BatchNormalization())
    
def build_cnn_model(conv_layers, dense_layers, input_shape):
    model = Sequential()
    first = True

    for conv_layer in conv_layers:
        if first:
            add_conv1D_layer(model, conv_layer['filters'], conv_layer['pool_size'], conv_layer['kernel_size'], conv_layer['activation_func'], input_shape)
            first = False
            continue
        add_conv1D_layer(model, conv_layer['filters'], conv_layer['pool_size'], conv_layer['kernel_size'], conv_layer['activation_func'])

    model.add(Dropout(rate=0.4))

    model.add(Flatten())

    for dense_layer in dense_layers:
        model.add(Dense(dense_layer['num_units'], activation=dense_layer['activation_func']))

    return model


def build_rnn_model(rnn_layers,dense_layers, input_shape):
    model = Sequential()
    first = True

    for rnn_layer in rnn_layers:
        if first:
            if len(rnn_layers) == 1:
                model.add(LSTM(rnn_layer['num_units'], input_shape=input_shape, return_sequences=False))
                continue
            model.add(LSTM(rnn_layer['num_units'], input_shape=input_shape, return_sequences=True))
            first = False
            continue
        if rnn_layer == rnn_layers[len(rnn_layers)-1]:
            model.add(LSTM(rnn_layer['num_units'], return_sequences=False))
            continue
        model.add(LSTM(rnn_layer['num_units'], return_sequences=True))


    for dense_layer in dense_layers:
        model.add(Dense(dense_layer['num_units'], activation=dense_layer['activation_func']))

    return model