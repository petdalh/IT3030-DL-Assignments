import sys
import os

script_dir = os.path.dirname(__file__)  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.append(parent_dir)

import numpy as np
from neural_network import NeuralNetwork
from neural_network import ReLU, Softmax
from neural_network import L1Regularizer
from neural_network import MSELoss
from generator import ImageGenerator
import matplotlib.pyplot as plt


# function declaration

def predict(network, X):
    predictions = []
    for i in range(X.shape[0]):
        output = network.forward(X[i].reshape(1, -1))  # Reshape if your network expects a 2D array
        predicted_class = np.argmax(output)
        predictions.append(predicted_class)
    return np.array(predictions)

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    accuracy = correct / y_true.shape[0]
    return accuracy

def create_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)  # Updated dtype
    for actual, predicted in zip(y_true, y_pred):
        confusion_matrix[actual][predicted] += 1
    return confusion_matrix

def classification_report(confusion_mat):
    precision = np.diag(confusion_mat) / np.sum(confusion_mat, axis=0)
    recall = np.diag(confusion_mat) / np.sum(confusion_mat, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print("Class\tPrecision\tRecall\tF1-Score")
    for i in range(len(precision)):
        print(f"{i}\t{precision[i]:.2f}\t\t{recall[i]:.2f}\t{f1_score[i]:.2f}")


# Initialize the neural network with the Mean Squared Error loss function
network = NeuralNetwork(loss_function=MSELoss())

# Define the network architecture
network.add_layer(input_size=1600, outp=128, activation=ReLU())  # First layer with ReLU activation
network.add_layer(input_size=128, outp=4, activation=Softmax())  # Output layer with Softmax activation

# Set up the image generator
n = 40  # Image dimensions (n x n)
noise = 0.05  # Noise level
generator = ImageGenerator(n, noise)

# Define the width and height range for the shapes
width_range = (n // 4, n // 2)
height_range = (n // 4, n // 2)

# Set the ratio for training, validation, and testing sets
train_ratio = 0.7
val_ratio = 0.2

# Generate the datasets
train_set, val_set, test_set = generator.generate_sets(
    num_images=100,
    width_range=width_range,
    height_range=height_range,
    train_ratio=train_ratio,
    val_ratio=val_ratio
)

# Function to preprocess and flatten images
def preprocess_images(image_set):
    return np.array([np.array(image).flatten() for image in image_set])

# Function to create one-hot encoded labels
def create_labels(num_images, num_classes=4):
    labels = np.zeros((num_images, num_classes))
    for i in range(num_images):
        labels[i, i % num_classes] = 1
    return labels

# Preprocess the images and create labels for training, validation, and testing sets
X_train = preprocess_images(train_set)
y_train = create_labels(len(train_set))

X_val = preprocess_images(val_set)
y_val = create_labels(len(val_set))

X_test = preprocess_images(test_set)
y_test = create_labels(len(test_set))

# Train the network
network.fit(X_train, y_train, epochs=10, learning_rate=0.01, batch_size=32)

# Evaluate the network on the validation set
validation_output = network.forward(X_val)
validation_loss = network.loss_function.compute_loss(validation_output, y_val)
print(f"Validation Loss: {validation_loss}")

# Evaluate the network on the test set
test_output = network.forward(X_test)
test_loss = network.loss_function.compute_loss(test_output, y_test)
print(f"Test Loss: {test_loss}")


# Assuming y_test_indices is the true class indices and not one-hot encoded
y_test_indices = np.argmax(y_test, axis=1)  # Convert from one-hot if necessary

# Get predictions
test_predictions = predict(network, X_test)

# Calculate accuracy
accuracy = calculate_accuracy(y_test_indices, test_predictions)
print(f"Test Accuracy: {accuracy}")

# Create confusion matrix
conf_mat = create_confusion_matrix(y_test_indices, test_predictions, num_classes=4)
print("Confusion Matrix:")
print(conf_mat)

# Print classification report
print("Classification Report:")
classification_report(conf_mat)

label_names = {0: "Rectangle", 1: "Triangle", 2: "Circle", 3: "Cross"}


indices = np.random.choice(range(len(X_test)), 10, replace=False)

# Predict the class for these images
predicted_classes = predict(network, X_test[indices])

# Visualization
fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # Adjust the figure size as needed
axes = axes.flatten()

for i, idx in enumerate(indices):
    image = X_test[idx].reshape(n, n)  # Assuming n x n is the size of the images
    true_label = y_test_indices[idx]  # True label
    predicted_label = predicted_classes[i]  # Predicted label

    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"True: {label_names[true_label]}\nPredicted: {label_names[predicted_label]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()