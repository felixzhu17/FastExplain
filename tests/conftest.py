import pytest
import pandas as pd
from modelflow import model_data
from .params import *


@pytest.fixture(scope="session")
def test_class_csv():
    return pd.read_csv(CLASS_DF_PATH)


@pytest.fixture(scope="session")
def rf_class_object(test_class_csv):
    return model_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        model="rf",
        perc_train=TRAIN_SPLIT,
    )


@pytest.fixture(scope="session")
def xgb_class_object(test_class_csv):
    return model_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        model="xgb",
        perc_train=TRAIN_SPLIT,
    )


@pytest.fixture(scope="session")
def ebm_class_object(test_class_csv):
    return model_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        model="ebm",
        perc_train=TRAIN_SPLIT,
    )
