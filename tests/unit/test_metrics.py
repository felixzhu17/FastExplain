import pytest
from tests.utils import check_dict_na


def test_rmse(rf_reg_object, xgb_reg_object, ebm_reg_object):
    assert check_dict_na(rf_reg_object.rmse["model"])
    assert check_dict_na(xgb_reg_object.rmse["model"])
    assert check_dict_na(ebm_reg_object.rmse["model"])
    assert check_dict_na(rf_reg_object.rmse["benchmark"])
    assert check_dict_na(xgb_reg_object.rmse["benchmark"])
    assert check_dict_na(ebm_reg_object.rmse["benchmark"])


def test_cross_entropy(rf_class_object, xgb_class_object, ebm_class_object):
    assert check_dict_na(rf_class_object.cross_entropy["model"])
    assert check_dict_na(xgb_class_object.cross_entropy["model"])
    assert check_dict_na(ebm_class_object.cross_entropy["model"])


def test_auc(rf_class_object, xgb_class_object, ebm_class_object):
    assert check_dict_na(rf_class_object.auc["model"])
    assert check_dict_na(xgb_class_object.auc["model"])
    assert check_dict_na(ebm_class_object.auc["model"])
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
    rf_class_object.plot_one_way_cross_entropy("Age")
    xgb_class_object.plot_one_way_cross_entropy("Age")
    ebm_class_object.plot_one_way_cross_entropy("Age")
    assert True


def test_one_way_reg_error(rf_reg_object, xgb_reg_object, ebm_reg_object):
    rf_reg_object.plot_one_way_squared_error()
    xgb_reg_object.plot_one_way_squared_error("Age")
    ebm_reg_object.plot_one_way_squared_error()
    assert True
