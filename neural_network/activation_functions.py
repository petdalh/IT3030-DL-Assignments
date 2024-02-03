import numpy as np

class ActivationFunction:
    def apply(self, x):
        pass

    def derivative(self, x):
        pass

class Softmax(ActivationFunction):
    def apply(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)

    def derivative(self, x):
        return self.apply(x) * (1 - self.apply(x))
    
class Sigmoid(ActivationFunction):
    def apply(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.apply(x) * (1 - self.apply(x))
    
class ReLU(ActivationFunction):
    def apply(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x <= 0, 0, 1)
    
