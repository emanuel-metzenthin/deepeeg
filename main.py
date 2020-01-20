import argparser
from deepeeg.dataloader import read_brainvis_from_directory
from deepeeg.deepeeg import DeepEEG

def main():
    arguments = argparser.parse_arguments()

    MODE = arguments.mode

    train, val = read_brainvis_from_directory('./data')

    deepeeg = DeepEEG()

    if MODE == argparser.TRAIN_MODE:
        input_shape = (arguments.num_timesteps, arguments.num_sensors)

        if arguments.model == argparser.CNN:
            model = deepeeg.train_cnn(input_shape=input_shape, save_model_to='./model.json', save_weights_to='./weights.hdf5')

        elif arguments.model == argparser.RNN:
            pass

    elif MODE == argparser.PREDICT_MODE:
        pass
    model.fit(train,val,epochs=1000)

if __name__ == '__main__':
    main()