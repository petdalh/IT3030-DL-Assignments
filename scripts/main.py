import sys
import os

script_dir = os.path.dirname(__file__)  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.append(parent_dir)

import numpy as np
from neural_network import NeuralNetwork
from neural_network import ReLU, Softmax
from neural_network import L1Regularizer
from neural_network import MSELoss, CrossEntropyLoss
from generator import ImageGenerator
import matplotlib.pyplot as plt

network = NeuralNetwork(loss_function=CrossEntropyLoss())


# Define a more complex network architecture
network.add_layer(input_size=1600, outp=256,
                  activation=ReLU())  # Increased size for the first layer
network.add_layer(input_size=256, outp=128,
                  activation=ReLU())  # Additional hidden layer
network.add_layer(input_size=128, outp=64,
                  activation=ReLU())  # Additional hidden layer

# If you want to include dropout for regularization (assuming dropout implementation is available)
# network.add_layer(input_size=128, outp=64, activation=ReLU(), dropout_rate=0.5)

network.add_layer(input_size=64, outp=4, activation=Softmax())  # Output layer

# Set up the image generator (parameters remain unchanged)
n = 40
noise = 0.01
generator = ImageGenerator(n, noise)

# Generate the datasets (parameters remain unchanged)
train_set, val_set, test_set = generator.generate_sets(
    num_images=1500,  # Increased number of images for more substantial training
    width_range=(10, 20),
    height_range=(10, 20),
    train_ratio=0.7,
    val_ratio=0.2)


# Preprocess function to separate images from labels
def preprocess_images_and_labels(image_label_pairs):
    images = np.array(
        [np.array(image).flatten() for image, _ in image_label_pairs])
    labels = [label for _, label in image_label_pairs]
    return images, labels


# Convert labels from names to one-hot encoded labels
def labels_to_one_hot(labels, label_to_index):
    numeric_labels = np.array([label_to_index[label] for label in labels])
    return np.eye(len(label_to_index))[numeric_labels]


# Mapping from label names to indices
label_to_index = {'Rectangle': 0, 'Triangle': 1, 'Circle': 2, 'Cross': 3}

# Preprocessing datasets
X_train, train_labels = preprocess_images_and_labels(train_set)
y_train = labels_to_one_hot(train_labels, label_to_index)

X_test, test_labels = preprocess_images_and_labels(test_set)
y_test = labels_to_one_hot(test_labels, label_to_index)

# Training the network
network.fit(X_train,
            y_train,
            epochs=50,
            learning_rate=0.08,
            batch_size=12,
            verbose=True)  # Increased epochs


# Predict function
def predict(network, X):
    predictions = []
    for i in range(X.shape[0]):
        output = network.forward(X[i].reshape(1, -1))
        predicted_class = np.argmax(output)
        predictions.append(predicted_class)
    return np.array(predictions)


# Predicting on the test set
test_predictions = predict(network, X_test)


# Plotting a sample of 10 images with their corresponding correct and predicted labels
def plot_sample(X, true_labels, predictions, label_names, sample_size=10):
    indices = np.random.choice(range(len(X)), sample_size, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        image = X[idx].reshape(n, n)
        true_label = label_names[np.argmax(true_labels[idx])]
        predicted_label = label_names[predictions[idx]]
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"True: {true_label}\nPredicted: {predicted_label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


# Inverse mapping from indices to label names for plotting
index_to_label = {v: k for k, v in label_to_index.items()}

# Plotting the sample
plot_sample(X_test, y_test, test_predictions, index_to_label)
