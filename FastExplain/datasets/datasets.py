import os
from re import L

import pandas as pd

DF_PATH = os.path.dirname(os.path.abspath(__file__))
TITANIC_CSV_PATH = os.path.abspath(os.path.join(DF_PATH, "titanic.csv"))
TITANIC_URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"


def load_titanic_data():
    try:
        return pd.read_csv(TITANIC_CSV_PATH)
    except:
        return pd.read_csv(TITANIC_URL)

