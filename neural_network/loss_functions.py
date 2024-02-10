import numpy as np


class Loss:

    def compute_loss(self, predicted, true):
        """
        Computes the loss value.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def compute_gradient(self, predicted, true):
        """
        Computes the gradient of the loss function with respect to the output of the network.
        """
        raise NotImplementedError("Must be implemented by subclass.")


class CrossEntropyLoss(Loss):

    def compute_loss(self, predicted, true):
        epsilon = 1e-10  # A small constant to prevent log(0)
        predicted_clipped = np.clip(predicted, epsilon,
                                    1 - epsilon)  # Clip predicted values
        m = true.shape[0]
        log_likelihood = -np.log(predicted_clipped[range(m),
                                                   np.argmax(true, axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def compute_gradient(self, predicted, true):
        m = true.shape[0]
        predicted_clipped = np.clip(predicted, 1e-10,
                                    1 - 1e-10)  # Also clip here for safety
        dloss = predicted_clipped - true
        dloss /= m
        return dloss


class MSELoss(Loss):

    def compute_loss(self, predicted, true):
        loss = np.mean((predicted - true)**2)
        return loss

    def compute_gradient(self, predicted, true):
        dloss = 2 * (predicted - true) / true.shape[0]
        return dloss
