import numpy as np


class ActivationFunction:

    def apply(self, x):
        pass

    def derivative(self, x):
        pass

class Linear(ActivationFunction):

    def apply(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)
    
class Tanh(ActivationFunction):
    
        def apply(self, x):
            return np.tanh(x)
    
        def derivative(self, x):
            return 1 - np.tanh(x) ** 2


class Softmax(ActivationFunction):

    def apply(self, x):
        shiftx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=1, keepdims=True)

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
    

