import numpy as np

class Regularizer:
    def __init__(self, rate=0.001):
        self.rate = rate
    
    def grad(self, weights):
        # Placeholder for the gradient computation method
        raise NotImplementedError("This method should be overridden by subclasses.")


class L1Regularizer(Regularizer):
    def grad(self, weights):
        return self.rate * np.sign(weights)
    
class L1Regularizer(Regularizer):
    def grad(self, weights):
        return self.rate * np.sign(weights)

