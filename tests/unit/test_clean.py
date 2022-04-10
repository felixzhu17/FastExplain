from model_helper.clean import *
from tests.params import *
from math import floor
from sklearn.preprocessing import MinMaxScaler


def test_split(test_class_csv, clean_class_object):
    assert (
        len(clean_class_object.train_xs)
        == len(clean_class_object.train_y)
        == floor(len(test_class_csv) * TRAIN_SPLIT)
    )


def test_stratify(clean_class_object):
    assert (
        -STRATIFY_ERROR_MARGIN
        <= sum(clean_class_object.val_y == 1) / sum(clean_class_object.val_y == 0)
        - sum(clean_class_object.train_y == 1) / sum(clean_class_object.train_y == 0)
        <= STRATIFY_ERROR_MARGIN
    )


def test_classification(clean_class_object):
    assert clean_class_object.classification == True


def test_ordinal_encoding(clean_class_object):
    assert len(clean_class_object.cat_mapping) == 4
    assert "Cabin" in clean_class_object.cat_mapping


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


def test_fill_median(clean_class_object):
    assert "Age_na" in clean_class_object.cat_names
    assert clean_class_object.train_xs.Age.isna().sum() == 0


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
