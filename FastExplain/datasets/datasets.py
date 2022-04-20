import os

import pandas as pd

DF_PATH = os.path.dirname(os.path.abspath(__file__))
TITANIC_CSV_PATH = os.path.abspath(os.path.join(DF_PATH, "titanic.csv"))
TITANIC_URL = "https://raw.githubusercontent.com/felixzhu17/FastExplain/main/FastExplain/datasets/titanic.csv"


def load_titanic_data():
    try:
        return pd.read_csv(TITANIC_CSV_PATH)
    except:
        return pd.read_csv(TITANIC_URL)
