import pytest
from modelflow.explain import *


def test_multi_ale(rf_class_object, xgb_class_object):
    plot_ale(
        [rf_class_object.m, xgb_class_object.m],
        [rf_class_object.data.xs, xgb_class_object.data.xs],
        "Age",
    )
    assert True


def test_multi_ebm(ebm_class_object):
    plot_ale(
        [ebm_class_object.m, ebm_class_object.m],
        [ebm_class_object.data.xs, ebm_class_object.data.xs],
        "Age",
    )
    assert True


def test_ale(rf_class_object, xgb_class_object):
    rf_class_object.plot_ale("Age")
    xgb_class_object.plot_ale("Age")
    assert True


def test_ebm(ebm_class_object):
    ebm_class_object.plot_ebm_explain("Age")
    assert True


def test_feature_importance(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.plot_feature_importance()
    xgb_class_object.plot_feature_importance()
    ebm_class_object.plot_feature_importance()
    assert True


def test_one_way(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.plot_one_way_analysis("Age")
    xgb_class_object.plot_one_way_analysis("Age")
    ebm_class_object.plot_one_way_analysis("Age")
    assert True


def test_two_way(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.plot_two_way_analysis(["Age", "Fare"])
    xgb_class_object.plot_two_way_analysis(["Age", "Fare"])
    ebm_class_object.plot_two_way_analysis(["Age", "Fare"])
    assert True


def test_feature_correlation(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.feature_correlation()
    xgb_class_object.feature_correlation()
    ebm_class_object.feature_correlation()
    assert True


def test_shap_values(rf_reg_object, xgb_reg_object, rf_class_object, xgb_class_object):
    _ = rf_reg_object.shap_force_plot(query="Age < 30")
    _ = xgb_reg_object.shap_force_plot()
    _ = rf_class_object.shap_force_plot()
    _ = xgb_class_object.shap_force_plot(query="Age < 30")

    _ = rf_reg_object.shap_summary_plot(query="Age < 30")
    _ = xgb_reg_object.shap_summary_plot()
    _ = rf_class_object.shap_summary_plot()
    _ = xgb_class_object.shap_summary_plot(query="Age < 30")

    _ = rf_reg_object.shap_dependence_plot("Age", query="Age < 30")
    _ = xgb_reg_object.shap_dependence_plot("Age")
    _ = rf_class_object.shap_dependence_plot("Age")
    _ = xgb_class_object.shap_dependence_plot("Age", query="Age < 30")

    _ = rf_reg_object.shap_importance_plot()
    _ = xgb_reg_object.shap_importance_plot()
    _ = rf_class_object.shap_importance_plot()
    _ = xgb_class_object.shap_importance_plot()

    assert True


def test_sensitivity(rf_reg_object, xgb_reg_object, ebm_reg_object):
    rf_reg_object.sensitivity_test(
        replace_features=["Age"],
        replacement_conditions=["Sex == 0"],
        replacement_values=[10],
    )
    xgb_reg_object.sensitivity_test(
        replace_features=["Age"],
        replacement_conditions=["Sex == 0"],
        replacement_values=[10],
    )
    ebm_reg_object.sensitivity_test(
        replace_features=["Age"],
        replacement_conditions=["Sex == 0"],
        replacement_values=[10],
    )
    assert True
