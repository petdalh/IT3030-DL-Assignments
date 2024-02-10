from .layer import Layer
from .loss_functions import CrossEntropyLoss, MSELoss
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, loss_function):
        self.layers = []
        self.loss_function = loss_function
        self.losses = []
        self.gradients_norm = []
        self.weights_norm = []

    def add_layer(self, input_size, outp, activation="none", regularizer=None):
        layer = Layer(input_size, outp, activation)
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(
                inputs)  # Use the output of the forward pass
        return inputs

    def backward(self, dvalues):
        for layer in reversed(self.layers):
            dvalues = layer.backward_pass(dvalues)
        return dvalues

    def update_params(self, learning_rate):
        for layer in self.layers:
            layer.update_params(learning_rate)

    def fit(self,
            X_train,
            y_train,
            epochs,
            learning_rate,
            batch_size=None,
            verbose=False):
        n_samples = X_train.shape[0]
        if batch_size is None:
            batch_size = n_samples  # Use full dataset if no batch size specified

        for epoch in range(epochs):
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            epoch_losses = []  # To track loss over the epoch
            epoch_gradients_norm = []  # To track gradients norm over the epoch
            epoch_weights_norm = []  # To track weights norm over the epoch

            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                output = self.forward(X_batch)
                loss, dloss = self.compute_loss_and_gradient(output, y_batch)
                self.backward(dloss)
                self.update_params(learning_rate)

                epoch_losses.append(loss)
                # Aggregate gradients and weights norm for this batch
                batch_gradients_norm = sum(
                    np.linalg.norm(layer.dweights) for layer in self.layers)
                batch_weights_norm = sum(
                    np.linalg.norm(layer.weights) for layer in self.layers)
                epoch_gradients_norm.append(batch_gradients_norm)
                epoch_weights_norm.append(batch_weights_norm)

            # Compute the average loss, gradients norm, and weights norm for the epoch
            avg_epoch_loss = np.mean(epoch_losses)
            avg_gradients_norm = np.mean(epoch_gradients_norm)
            avg_weights_norm = np.mean(epoch_weights_norm)

            if verbose:
                print(
                    f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}, Avg Gradients Norm: {avg_gradients_norm}, Avg Weights Norm: {avg_weights_norm}"
                )
                #store the loss, gradients norm, and weights norm for this epoch
                self.losses.append(avg_epoch_loss)
                self.gradients_norm.append(avg_gradients_norm)
                self.weights_norm.append(avg_weights_norm)

    def plot_training_progress(self):
        import matplotlib.pyplot as plt
        plt.plot(self.losses, label="Loss")
        plt.plot(self.gradients_norm, label="Gradients Norm")
        plt.plot(self.weights_norm, label="Weights Norm")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    def compute_loss_and_gradient(self, output, y_true):
        loss = self.loss_function.compute_loss(output, y_true)
        dloss = self.loss_function.compute_gradient(output, y_true)
        return loss, dloss
