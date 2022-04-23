from abc import ABC, abstractmethod


class Clean(ABC):
    """Base for clean classes"""

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass

    @abstractmethod
    def transform(self):
        pass
