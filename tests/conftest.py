import pandas as pd
import numpy as np
import pytest

from FastExplain import model_data
from FastExplain.datasets import load_titanic_data
from tests.params import *


@pytest.fixture(scope="session")
def test_csv():
    df = load_titanic_data()
    df["test"] = [np.nan for i in range(len(df))]
    return df


@pytest.fixture(scope="session")
def rf_class_object(test_csv):
    return model_data(
        test_csv,
        dep_var=CLASS_DEP_VAR,
        model="rf",
        perc_train=TRAIN_SPLIT,
    )


@pytest.fixture(scope="session")
def xgb_class_object(test_csv):
    return model_data(
        test_csv,
        dep_var=CLASS_DEP_VAR,
        model="xgb",
        perc_train=TRAIN_SPLIT,
        hypertune=True,
    )


@pytest.fixture(scope="session")
def ebm_class_object(test_csv):
    return model_data(
        test_csv,
        dep_var=CLASS_DEP_VAR,
        model="ebm",
        perc_train=TRAIN_SPLIT,
    )


@pytest.fixture(scope="session")
def rf_reg_object(test_csv):
    return model_data(
        test_csv,
        dep_var=REG_DEP_VAR,
        model="rf",
        perc_train=TRAIN_SPLIT,
        hypertune=True,
    )


@pytest.fixture(scope="session")
def xgb_reg_object(test_csv):
    return model_data(
        test_csv,
        dep_var=REG_DEP_VAR,
        model="xgb",
        perc_train=TRAIN_SPLIT,
    )


@pytest.fixture(scope="session")
def ebm_reg_object(test_csv):
    return model_data(
        test_csv,
        dep_var=REG_DEP_VAR,
        model="ebm",
        perc_train=TRAIN_SPLIT,
    )
