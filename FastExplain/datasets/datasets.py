import os

import pandas as pd

from FastExplain.clean.dates import add_datepart

DF_PATH = os.path.dirname(os.path.abspath(__file__))


def load_titanic_data():
    TITANIC_CSV_PATH = os.path.abspath(os.path.join(DF_PATH, "titanic.csv"))
    TITANIC_URL = "https://raw.githubusercontent.com/felixzhu17/FastExplain/main/FastExplain/datasets/titanic.csv"
    try:
        return pd.read_csv(TITANIC_CSV_PATH)
    except:
        return pd.read_csv(TITANIC_URL)


def load_nba_data():
    NBA_CSV_PATH = os.path.abspath(os.path.join(DF_PATH, "nba.csv"))
    NBA_URL = "https://raw.githubusercontent.com/felixzhu17/FastExplain/main/FastExplain/datasets/nba.csv"
    try:
        df = pd.read_csv(
            NBA_CSV_PATH, index_col=False, low_memory=False, parse_dates=["game_date"]
        )
    except:
        df = pd.read_csv(
            NBA_URL, index_col=False, low_memory=False, parse_dates=["game_date"]
        )

    df = df[~df["shot_made_flag"].isna()]
    df = add_datepart(df, "game_date")
    df = df[
        [
            "action_type",
            "game_event_id",
            "seconds_remaining",
            "loc_y",
            "loc_x",
            "lat",
            "lon",
            "game_id",
            "minutes_remaining",
            "shot_distance",
            "combined_shot_type",
            "season",
            "period",
            "shot_type",
            "playoffs",
            "shot_made_flag",
        ]
    ]
    return df
