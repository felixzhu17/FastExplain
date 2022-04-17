import os
from re import L

import pandas as pd

DF_PATH = os.path.dirname(os.path.abspath(__file__))
TITANIC_CSV_PATH = os.path.abspath(os.path.join(DF_PATH, "titanic.csv"))


def load_titanic_data():
    return pd.read_csv(TITANIC_CSV_PATH)
