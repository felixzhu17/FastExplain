from math import floor

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from FastExplain import model_data
from FastExplain.clean import prepare_data
from tests.params import (
    CAT_COLS,
    CLASS_DEP_VAR,
    CONT_COLS,
    STRATIFY_ERROR_MARGIN,
    TRAIN_SPLIT,
)


def test_split(test_csv, rf_class_object):
    assert (
        len(rf_class_object.data.train_xs)
        == len(rf_class_object.data.train_y)
        == floor(len(test_csv) * TRAIN_SPLIT)
    )


def test_stratify(rf_class_object):
    assert sum(rf_class_object.data.val_y == 1) / sum(
        rf_class_object.data.val_y == 0
    ) - sum(rf_class_object.data.train_y == 1) / sum(
        rf_class_object.data.train_y == 0
    ) == pytest.approx(
        0, abs=STRATIFY_ERROR_MARGIN
    )


def test_classification(rf_class_object):
    assert rf_class_object.data.classification == True


def test_regression(rf_reg_object):
    assert rf_reg_object.data.classification == False


def test_ordinal_encoding(rf_class_object):
    assert len(rf_class_object.data.cat_mapping) == 8
    assert "Cabin" in rf_class_object.data.cat_mapping


def test_one_hot_encoding(test_csv):
    one_hot_transform = prepare_data(
        test_csv,
        cat_names=CAT_COLS,
        cont_names=CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        return_class=True,
        perc_train=TRAIN_SPLIT,
        cat_strategy="one_hot",
    )

    assert len(one_hot_transform.cat_names) == 150
    assert "Sex_female" in one_hot_transform.cat_names


def test_fill_median(rf_class_object):
    assert "Age_na" in rf_class_object.data.cat_names
    assert rf_class_object.data.train_xs.Age.isna().sum() == 0


def test_custom_transformation(test_csv):
    min_max_transform = prepare_data(
        test_csv,
        cat_names=CAT_COLS,
        cont_names=CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        return_class=True,
        perc_train=TRAIN_SPLIT,
        cont_transformations=[MinMaxScaler],
    )
    assert min_max_transform.train_xs.Age.max() == 1
    assert min_max_transform.train_xs.Age.min() == 0


def test_custom_model(test_csv):
    custom_model = model_data(
        test_csv,
        dep_var=CLASS_DEP_VAR,
        model=RandomForestClassifier,
    )
    custom_model_2 = model_data(
        test_csv, dep_var=CLASS_DEP_VAR, model=custom_model.m, use_fitted_model=True
    )
    assert hasattr(custom_model, "m")
    assert hasattr(custom_model_2, "m")


def test_hypertune_defaults(test_csv):
    custom_model = model_data(
        test_csv,
        dep_var=CLASS_DEP_VAR,
        model="rf",
        hypertune=True,
        min_impurity_decrease=1,
    )
    assert custom_model.params["min_impurity_decrease"] == 1


def test_empty_clean(test_csv):
    prepare_data(test_csv)
    assert True
