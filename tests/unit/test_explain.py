import pytest

from FastExplain.explain import plot_ale, plot_one_way_analysis


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
    rf_class_object.plot_ale("Age", bins=[1, 2, 3, 4, 5])
    xgb_class_object.plot_ale(
        "Age",
        filter="Age < 30",
        title="Test",
        dep_name="Test",
        feature_name="Test",
    )
    assert True


def test_ale_2d(rf_class_object, xgb_class_object):
    rf_class_object.plot_ale_2d(["Age", "SibSp"])
    xgb_class_object.plot_ale_2d(
        ["Age", "SibSp"],
        filter="Age < 30",
        title="Test",
        dep_name="Test",
        feature_names=["Test", "Test"],
    )
    assert True


def test_ale_class(rf_class_object, xgb_class_object):
    assert len(rf_class_object.ale("Sex").index) == 2
    assert len(xgb_class_object.ale("Cabin", numeric=False).index) == 148
    rf_class_object.plot_ale("Sex", filter="Age < 30")
    xgb_class_object.plot_ale("Sex", title="Test", dep_name="Test", feature_name="Test")
    assert True


def test_ebm(ebm_class_object):
    ebm_class_object.plot_ebm_explain("Age", title="Test")
    ebm_class_object.plot_ebm_explain("Sex", title="Test")
    assert True


def test_feature_importance(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.plot_feature_importance()
    xgb_class_object.plot_feature_importance()
    ebm_class_object.plot_feature_importance()
    assert True


def test_one_way(test_csv, rf_class_object, xgb_class_object, ebm_class_object):
    plot_one_way_analysis(test_csv, "Sex", "Survived")
    rf_class_object.plot_one_way_analysis("Age", filter="Age < 30", sort=True)
    xgb_class_object.plot_one_way_analysis("Age")
    ebm_class_object.plot_one_way_analysis("Age", ["Survived", "SibSp"])
    ebm_class_object.plot_one_way_analysis(
        "Age", ["Survived", "SibSp"], dual_axis_y_col=True
    )
    assert True


def test_two_way(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.plot_two_way_analysis(["Age", "Fare"])
    xgb_class_object.plot_two_way_analysis(["Age", "Fare"])
    ebm_class_object.plot_two_way_analysis(["Age", "Fare"], filter="Age < 30")
    assert True


def test_feature_correlation(rf_class_object, xgb_class_object, ebm_class_object):
    rf_class_object.feature_correlation()
    xgb_class_object.feature_correlation()
    ebm_class_object.feature_correlation()
    rf_class_object.cluster_features()
    xgb_class_object.cluster_features()
    ebm_class_object.cluster_features()
    assert True


def test_shap_values(rf_reg_object, xgb_class_object):
    rf_reg_object.shap_force_plot()
    xgb_class_object.shap_force_plot(filter="Age < 30")

    rf_reg_object.shap_summary_plot(filter="Age < 30")
    xgb_class_object.shap_summary_plot()

    rf_reg_object.shap_dependence_plot("Age", filter="Age < 30")
    xgb_class_object.shap_dependence_plot("Age")

    rf_reg_object.shap_importance_plot()
    xgb_class_object.shap_importance_plot()
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


def test_ice(rf_reg_object, xgb_class_object, ebm_reg_object):
    rf_reg_object.plot_ice("Age", filter="Sex == 1")
    xgb_class_object.plot_ice("Age", filter="Sex == 1")
    ebm_reg_object.plot_ice("Age", sample=100)
    assert True


def test_frequency(rf_reg_object):
    rf_reg_object.plot_histogram("Age", filter="Sex == 1")
    assert True
