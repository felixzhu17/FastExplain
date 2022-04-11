import pytest
from modelflow.metrics import *


def test_rmse(rf_reg_object, xgb_reg_object, ebm_reg_object):
    assert rf_reg_object.benchmark_rmse > rf_reg_object.model_rmse
    assert xgb_reg_object.benchmark_rmse > xgb_reg_object.model_rmse
    assert ebm_reg_object.benchmark_rmse > ebm_reg_object.model_rmse


def test_auc(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.auc()
    xgb_class_object.auc()
    ebm_class_object.auc()
    assert True


def test_confusion_matrix(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.confusion_matrix()
    xgb_class_object.confusion_matrix()
    ebm_class_object.confusion_matrix()
    assert True
