from .layer import Layer
from .loss_functions import CrossEntropyLoss, MSELoss


class NeuralNetwork:

    def __init__(self, loss_function):
        self.layers = []
        self.loss_function = loss_function

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
    
    def fit(self, X_train, y_train, epochs, learning_rate, batch_size=None):
        n_samples = X_train.shape[0]
        if batch_size is None:
            batch_size = n_samples  # No batching, use full dataset
        
        for epoch in range(epochs):
            # Shuffle your training data here if desired
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                # Forward pass
                output = self.forward(X_batch)
                # Compute loss and its gradient
                loss, dloss = self.compute_loss_and_gradient(output, y_batch)
                # Backward pass
                self.backward(dloss)
                # Update parameters
                self.update_params(learning_rate)
            
            # Optional: Print epoch number and loss here
            print(f"Epoch {epoch + 1}, Loss: {loss}")


    def compute_loss_and_gradient(self, output, y_true):
        loss = self.loss_function.compute_loss(output, y_true)
        dloss = self.loss_function.compute_gradient(output, y_true)
        return loss, dloss

