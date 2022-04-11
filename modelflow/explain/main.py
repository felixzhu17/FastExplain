from .one_way import OneWay
from .ale import Ale
from .ebm import EbmExplain
from .pdp import PDP
from .shap import ShapExplain
from .importance import Importance
from .sensitivity import Sensitivity


class Explain(OneWay, Ale, PDP, EbmExplain, ShapExplain, Importance, Sensitivity):
    def __init__(self, m, xs, df=None, dep_var=None):
        OneWay.__init__(self, m, xs, df, dep_var)
        Ale.__init__(self, m, xs)
        PDP.__init__(self, m, xs)
        EbmExplain.__init__(self, m, xs)
        ShapExplain.__init__(self, m, xs)
        Importance.__init__(self, m, xs)
        Sensitivity.__init__(self, m, xs)
