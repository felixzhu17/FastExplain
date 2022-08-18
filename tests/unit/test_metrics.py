import pytest

from FastExplain.metrics import get_one_way_error
from tests.utils import check_dict_na


def test_rmse(rf_reg_object, xgb_reg_object, ebm_reg_object):
    assert check_dict_na(rf_reg_object.error["rmse"]["model"])
    assert check_dict_na(xgb_reg_object.error["rmse"]["model"])
    assert check_dict_na(ebm_reg_object.error["rmse"]["model"])
    assert check_dict_na(rf_reg_object.error["rmse"]["benchmark"])
    assert check_dict_na(xgb_reg_object.error["rmse"]["benchmark"])
    assert check_dict_na(ebm_reg_object.error["rmse"]["benchmark"])


def test_cross_entropy(rf_class_object, xgb_class_object, ebm_class_object):
    assert check_dict_na(rf_class_object.error["cross_entropy"]["model"])
    assert check_dict_na(xgb_class_object.error["cross_entropy"]["model"])
    assert check_dict_na(ebm_class_object.error["cross_entropy"]["model"])


def test_auc(rf_class_object, xgb_class_object, ebm_class_object):
    assert check_dict_na(rf_class_object.error["auc"]["model"])
    assert check_dict_na(xgb_class_object.error["auc"]["model"])
    assert check_dict_na(ebm_class_object.error["auc"]["model"])
    rf_class_object.plot_auc()
    xgb_class_object.plot_auc()
    ebm_class_object.plot_auc()
    assert True


def test_confusion_matrix(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.confusion_matrix()
    xgb_class_object.confusion_matrix()
    ebm_class_object.confusion_matrix()
    assert True


def test_one_way_class_error(rf_class_object, xgb_class_object, ebm_class_object):

    assert (
        len(
            get_one_way_error(
                rf_class_object.data.df,
                rf_class_object.raw_error["cross_entropy"]["model"]["overall"],
                "Sex",
            )
        )
        == 2
    )
    assert (
        len(
            get_one_way_error(
                xgb_class_object.data.df,
                xgb_class_object.raw_error["cross_entropy"]["model"]["overall"],
                "Sex",
            )
        )
        == 2
    )
    assert (
        len(
            get_one_way_error(
                ebm_class_object.data.df,
                ebm_class_object.raw_error["cross_entropy"]["model"]["overall"],
                "Sex",
            )
        )
        == 2
    )

    rf_class_object.plot_one_way_error("Age")
    xgb_class_object.plot_one_way_error("Age")
    ebm_class_object.plot_one_way_error("Age")
    assert True


def test_one_way_reg_error(rf_reg_object, xgb_reg_object, ebm_reg_object):
    rf_reg_object.plot_one_way_error()
    xgb_reg_object.plot_one_way_error("Age")
    ebm_reg_object.plot_one_way_error()
    assert True
