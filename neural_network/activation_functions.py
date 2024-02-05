import numpy as np


class ActivationFunction:

    def apply(self, x):
        pass

    def derivative(self, x):
        pass


class Softmax(ActivationFunction):

    def apply(self, x):
        # Subtract the max for numerical stability
        shiftx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, x):
        # Note: This simplistic derivative calculation for softmax is not used in practice
        # because the softmax derivative is typically combined with the cross-entropy loss derivative
        # which simplifies the calculation. You might not need this derivative for softmax if
        # you're using it with cross-entropy loss.
        softmax_output = self.apply(x)
        return softmax_output * (1 - softmax_output)


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
