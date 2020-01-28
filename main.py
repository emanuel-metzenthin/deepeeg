from deepeeg.dataloader import load_data_from_dir
from deepeeg.deepeeg import DeepEEG

def main():
    X_train, y_train, X_val, y_val = load_data_from_dir('./data/deepeeg-format/raw', test_size=0.217)

    deepeeg = DeepEEG()

    deepeeg.init_cnn((X_train.shape[1], X_train.shape[2]))

    deepeeg.train(X_train, y_train, X_val, y_val)

if __name__ == '__main__':
    main()