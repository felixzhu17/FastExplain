from modelflow.explain.one_way import OneWay
from modelflow.explain.ale import Ale
from modelflow.explain.ebm import EbmExplain
from modelflow.explain.pdp import PDP
from modelflow.explain.shap import ShapExplain
from modelflow.explain.importance import Importance
from modelflow.explain.sensitivity import Sensitivity


class Explain(OneWay, Ale, PDP, EbmExplain, ShapExplain, Importance, Sensitivity):
    def __init__(self, m, xs, df=None, dep_var=None, m_cols=None):
        OneWay.__init__(self, m, xs, df, dep_var)
        Ale.__init__(self, m, xs)
        PDP.__init__(self, m, xs)
        EbmExplain.__init__(self, m, xs)
        ShapExplain.__init__(self, m, xs)
        Importance.__init__(self, m, xs)
        Sensitivity.__init__(self, m, xs, m_cols)
