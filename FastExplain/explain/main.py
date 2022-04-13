from FastExplain.explain.one_way import OneWay
from FastExplain.explain.ale import Ale
from FastExplain.explain.ebm import EbmExplain
from FastExplain.explain.pdp import PDP
from FastExplain.explain.shap import ShapExplain
from FastExplain.explain.importance import Importance
from FastExplain.explain.sensitivity import Sensitivity


class Explain(OneWay, Ale, PDP, EbmExplain, ShapExplain, Importance, Sensitivity):
    def __init__(self, m, xs, df=None, dep_var=None, m_cols=None):
        OneWay.__init__(self, m, xs, df, dep_var)
        Ale.__init__(self, m, xs)
        PDP.__init__(self, m, xs)
        EbmExplain.__init__(self, m, xs)
        ShapExplain.__init__(self, m, xs)
        Importance.__init__(self, m, xs)
        Sensitivity.__init__(self, m, xs, m_cols)
