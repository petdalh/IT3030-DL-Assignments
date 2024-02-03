import numpy as np
from activation_functions import Softmax
from regularization import L1Regularizer


class Layer:

    def __init__(self, input_size, output_size, activation, regularizer=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.input_size = input_size
        self.output_size = output_size
        self.biases = np.zeros(output_size)
        self.activation = activation
        self.regularizer = regularizer

    def forward_pass(self, inputs):
        """
        Computes forward pass of the layer
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.activation.apply(self.output)

    def backward_pass(self, dvalues):
        """
        Computes backward pass of the layer
        """

        # Apply derivative of the activation function
        self.dvalues = dvalues * self.activation.derivative(self.output)

        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, self.dvalues)
        self.dbiases = np.sum(self.dvalues, axis=0, keepdims=True)

        # If there is a regularizer, compute the gradient of the regularizer
        if self.regularizer:
            self.dweights += self.regularizer.grad(self.weights)

        # Gradient on the input values for next layer in backpropagation
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
