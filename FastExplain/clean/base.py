from abc import ABC, abstractmethod


class Clean(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass

    @abstractmethod
    def transform(self):
        pass
