from abc import abstractmethod

class Optimizer:
    def __init__(self, **kwargs):
        self.problem = kwargs.get('problem')
        pass

    @abstractmethod
    def solve(self, **kwargs):
        pass

    @abstractmethod
    def _next(self, **kwargs):
        pass