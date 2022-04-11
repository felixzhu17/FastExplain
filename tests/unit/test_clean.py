from modelflow.clean import *
from tests.params import *
from math import floor
from sklearn.preprocessing import MinMaxScaler


def test_split(test_class_csv, rf_class_object):
    assert (
        len(rf_class_object.data.train_xs)
        == len(rf_class_object.data.train_y)
        == floor(len(test_class_csv) * TRAIN_SPLIT)
    )


def test_stratify(rf_class_object):
    assert (
        -STRATIFY_ERROR_MARGIN
        <= sum(rf_class_object.data.val_y == 1) / sum(rf_class_object.data.val_y == 0)
        - sum(rf_class_object.data.train_y == 1)
        / sum(rf_class_object.data.train_y == 0)
        <= STRATIFY_ERROR_MARGIN
    )


def test_classification(rf_class_object):
    assert rf_class_object.data.classification == True


def test_ordinal_encoding(rf_class_object):
    assert len(rf_class_object.data.cat_mapping) == 4
    assert "Cabin" in rf_class_object.data.cat_mapping


def test_one_hot_encoding(test_class_csv):
    one_hot_transform = prepare_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
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


def test_custom_transformation(test_class_csv):
    min_max_transform = prepare_data(
        test_class_csv,
        cat_names=CLASS_CAT_COLS,
        cont_names=CLASS_CONT_COLS,
        dep_var=CLASS_DEP_VAR,
        return_class=True,
        perc_train=TRAIN_SPLIT,
        cont_transformations=[MinMaxScaler],
    )
    assert min_max_transform.train_xs.Age.max() == 1
    assert min_max_transform.train_xs.Age.min() == 0
