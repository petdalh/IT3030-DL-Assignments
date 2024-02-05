import sys
import os

script_dir = os.path.dirname(__file__)  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.append(parent_dir)

import numpy as np
from neural_network import NeuralNetwork
from neural_network import ReLU, Softmax
from neural_network import L1Regularizer
import os
import sys

# sys.path.append('/Users/petterdalhaug/Documents/IT3030/IT3030-DL-Assignments')







nn = NeuralNetwork()
nn.add_layer(2, 3, ReLU(), L1Regularizer(0.001))
nn.add_layer(3, 1, ReLU(), L1Regularizer(0.001))
nn.add_layer(1, 2, Softmax(), L1Regularizer(0.001))

input_data = np.array([[1, 2], [3, 4], [5, 6]])

output = nn.forward(input_data)

print(output)
