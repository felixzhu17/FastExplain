from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def fit_regression(self):
        pass

    @abstractmethod
    def fit_classification(self):
        pass

    @abstractmethod
    def hypertune(self):
        pass
