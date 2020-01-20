import pandas as pd

def load_data(filename):
    print(pd.read_csv(filename))