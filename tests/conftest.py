import pytest
import pandas as pd
from model_helper.clean import prepare_data
from model_helper import model_data
from .params import *


@pytest.fixture(scope="session")
def test_class_csv():
    return pd.read_csv(CLASS_DF_PATH)


@pytest.fixture(scope="session")
def clean_class_object(test_class_csv):
    return prepare_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        return_class=True,
        perc_train=TRAIN_SPLIT,
    )


@pytest.fixture(scope="session")
def rf_class_object(test_class_csv):
    return model_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        type="rf",
    )


@pytest.fixture(scope="session")
def xgb_class_object(test_class_csv):
    return model_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        type="xgb",
    )


@pytest.fixture(scope="session")
def ebm_class_object(test_class_csv):
    return model_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        type="ebm",
    )
