import pytest
import pandas as pd
from FastExplain import model_data
from tests.params import *


@pytest.fixture(scope="session")
def test_csv():
    return pd.read_csv(CLASS_DF_PATH)


@pytest.fixture(scope="session")
def rf_class_object(test_csv):
    return model_data(
        test_csv,
        cat_names=CAT_COLS,
        cont_names=CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        model="rf",
        perc_train=TRAIN_SPLIT,
        hypertune=True,
    )


@pytest.fixture(scope="session")
def xgb_class_object(test_csv):
    return model_data(
        test_csv,
        cat_names=CAT_COLS,
        cont_names=CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        model="xgb",
        perc_train=TRAIN_SPLIT,
        hypertune=True,
    )


@pytest.fixture(scope="session")
def ebm_class_object(test_csv):
    return model_data(
        test_csv,
        cat_names=CAT_COLS,
        cont_names=CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        model="ebm",
        perc_train=TRAIN_SPLIT,
    )


@pytest.fixture(scope="session")
def rf_reg_object(test_csv):
    return model_data(
        test_csv,
        cat_names=CAT_COLS,
        cont_names=CONT_COLS,
        dep_var=REG_DEP_VAR,
        model="rf",
        perc_train=TRAIN_SPLIT,
        hypertune=True,
    )


@pytest.fixture(scope="session")
def xgb_reg_object(test_csv):
    return model_data(
        test_csv,
        cat_names=CAT_COLS,
        cont_names=CONT_COLS,
        dep_var=REG_DEP_VAR,
        model="xgb",
        perc_train=TRAIN_SPLIT,
        hypertune=True,
    )


@pytest.fixture(scope="session")
def ebm_reg_object(test_csv):
    return model_data(
        test_csv,
        cat_names=CAT_COLS,
        cont_names=CONT_COLS,
        dep_var=REG_DEP_VAR,
        model="ebm",
        perc_train=TRAIN_SPLIT,
    )
