from layer import Layer
from network import NeuralNetwork
import numpy as np


nn = NeuralNetwork()
nn.add_layer(2, 3)
nn.add_layer(3, 1, activation='relu')

input_data = np.array([[1, 2], [3, 4], [5, 6]])

output = nn.forward(input_data)

print(output)