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
