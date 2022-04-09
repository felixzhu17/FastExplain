from ...utils import Utils
from .one_way import OneWay, OneWayClassified
from .ale import Ale, AleClassified
from .ebm import EbmExplain, EbmExplainClassified
from .pdp import PDP, PDPClassified
from .shap import ShapExplain, ShapExplainClassified
from .importance import Importance, ImportanceClassified


class Explain(Ale, OneWay, PDP, EbmExplain, ShapExplain, Importance, Utils):
    ...


class ExplainClassified(
    OneWayClassified,
    AleClassified,
    PDPClassified,
    EbmExplainClassified,
    ShapExplainClassified,
    ImportanceClassified,
    Utils,
):
    def __init__(self, m, xs, df, dep_var):
        OneWayClassified.__init__(self, m, xs, df, dep_var)
        AleClassified.__init__(self, m, xs)
        PDPClassified.__init__(self, m, xs)
        EbmExplainClassified.__init__(self, m, xs)
        ShapExplainClassified.__init__(self, m, xs)
        ImportanceClassified.__init__(self, m, xs)
