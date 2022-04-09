import time
import functools
from .data import DataUtils
from .logic import LogicUtils
from .plot import PlotUtils
from .time import TimeUtils
from .logging import Logging
from .colours import ColourUtils

# Retries
def retry(retries, time_between_retries, func_on_fail=lambda *args: None):
    def actual_retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(e)
                    time.sleep(time_between_retries)
                    func_on_fail()
                    retry += 1
                    if retry >= retries:
                        raise e

        return wrapper

    return actual_retry


class Utils(DataUtils, LogicUtils, PlotUtils, TimeUtils, Logging, ColourUtils):
    def __init__(self):
        Logging.__init__(self)
        ColourUtils.__init__(self)
