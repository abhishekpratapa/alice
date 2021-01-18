from abc import ABC, abstractmethod

class DataTemplate(ABC):
    @abstractmethod
    def has_next(self):
        pass

    @abstractmethod
    def next(self, has_label=True):
        pass
