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
        self.val_losses = []

    def add_layer(self, input_size, outp, activation="none", regularizer=None):
        layer = Layer(input_size, outp, activation)
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(
                inputs)  # Use the output of the forward pass
            print(inputs)
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
            X_val,
            y_val,
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

            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                output = self.forward(X_batch)
                loss, dloss = self.compute_loss_and_gradient(output, y_batch)
                self.backward(dloss)
                self.update_params(learning_rate)

                epoch_losses.append(loss)

            avg_epoch_loss = np.mean(epoch_losses)

            # Validation step
            val_output = self.forward(X_val)
            val_loss, _ = self.compute_loss_and_gradient(
                val_output, y_val)  # No need for gradient

            if verbose:
                print(
                    f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}, Validation Loss: {val_loss}"
                )

            # Optionally, store the average loss and validation loss for later analysis
            self.losses.append(avg_epoch_loss)
            self.val_losses.append(val_loss)

    def plot_training_progress(self):
        plt.plot(self.losses, label="Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    def compute_loss_and_gradient(self, output, y_true):
        loss = self.loss_function.compute_loss(output, y_true)
        dloss = self.loss_function.compute_gradient(output, y_true)
        return loss, dloss
