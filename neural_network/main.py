from layer import Layer
from network import NeuralNetwork
from activation_functions import ReLU, Softmax
from regularization import L1Regularizer
import numpy as np

nn = NeuralNetwork()
nn.add_layer(2, 3, ReLU(), L1Regularizer(0.001))
nn.add_layer(3, 1, ReLU(), L1Regularizer(0.001))
nn.add_layer(1, 2, Softmax(), L1Regularizer(0.001))

input_data = np.array([[1, 2], [3, 4], [5, 6]])

output = nn.forward(input_data)

print(output)
