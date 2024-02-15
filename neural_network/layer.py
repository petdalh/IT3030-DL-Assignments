import numpy as np
from .activation_functions import Softmax, ReLU, ActivationFunction
from .regularization import L1Regularizer


class Layer:

    def __init__(self, input_size, output_size, activation, regularizer=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.input_size = input_size
        self.output_size = output_size
        self.biases = np.zeros(output_size)
        self.activation = activation
        self.regularizer = regularizer
        self.all_activations = []

    def forward_pass(self, inputs):
        """
        Computes forward pass of the layer
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        self.activations = self.output  # Current activations

        return self.activation.apply(self.output)

    def backward_pass(self, dvalues):
        """
        Computes backward pass of the layer
        """

        # Jacobian matrix for the activation function
        dN_dout = self.activation.derivative(self.output)

        # Apply derivative of the activation function
        self.dvalues = dvalues * dN_dout

        # Gradient on weights (dM/dW) and biases
        # dL/dW = dL/dN (transposed input matrix)
        self.dweights = np.dot(self.inputs.T, self.dvalues)

        self.dbiases = np.sum(
            self.dvalues, axis=0,
            keepdims=True).squeeze()  # This will also give a 1D array

        # If there is a regularizer, compute the gradient of the regularizer
        if self.regularizer:
            self.dweights += self.regularizer.grad(self.weights)

        # Gradient on the input values for next layer in backpropagation
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def save_activations(self):
        self.all_activations.append(self.activations.copy())

    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
