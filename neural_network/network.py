from layer import Layer


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, outp, activation="none", regularizer=None):
        layer = Layer(input_size, outp, activation)
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            layer.forward_pass(inputs)
            inputs = layer.output
        return inputs

    def backward(self, dvalues):
        for layer in reversed(self.layers):
            dvalues = layer.backward_pass(dvalues)
        return dvalues

    def update_params(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases
