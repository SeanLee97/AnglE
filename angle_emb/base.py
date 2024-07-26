from abc import ABCMeta, abstractmethod


class AngleBase(metaclass=ABCMeta):

    @abstractmethod
    def encode(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError
