import pandas as pd
import numpy as np


def load_dataset(dataset_path):
    """
    Loads CSV dataset for evaluation.
    """

    df = pd.read_csv(dataset_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y

