import sys
import os
import yaml

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


def predict(network, X):
    predictions = []
    for i in range(X.shape[0]):
        output = network.forward(X[i].reshape(1, -1))
        predicted_class = np.argmax(output)
        predictions.append(predicted_class)
    return np.array(predictions)


# Plotting a sample of 10 images with their corresponding correct and predicted labels
def plot_sample(X, true_labels, predictions, label_names, n, sample_size=10):
    indices = np.random.choice(range(len(X)), sample_size, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        image = X[idx].reshape(
            n, n)  # Now 'n' is defined within the function's scope
        true_label = label_names[np.argmax(true_labels[idx])]
        predicted_label = label_names[predictions[idx]]
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"True: {true_label}\nPredicted: {predicted_label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)


def create_network_from_config(config):
    activation_map = {'relu': ReLU, 'softmax': Softmax}

    loss_function_map = {
        'cross_entropy': CrossEntropyLoss,
        # Add other loss functions as needed
    }

    regularizer_map = {
        'L1': L1Regularizer,
        # Add other regularizers as needed
    }

    loss_function = loss_function_map[config['globals']['loss_function']]()
    network = NeuralNetwork(loss_function=loss_function)

    for layer_conf in config['layers']:
        activation = activation_map[layer_conf['activation']]()

        regularizer = None
        if 'regularizer' in layer_conf:
            reg_type = layer_conf['regularizer']['type']
            reg_rate = layer_conf['regularizer']['rate']
            regularizer = regularizer_map[reg_type](rate=reg_rate)

        network.add_layer(input_size=layer_conf['input_size'],
                          outp=layer_conf['output_size'],
                          activation=activation,
                          regularizer=regularizer)

    return network


def main(config_path):
    # Load the configuration
    config = load_config(config_path)

    # Create the neural network
    network = create_network_from_config(config)

    # Set up the image generator
    n = 40
    noise = 0.01
    generator = ImageGenerator(n, noise)

    # Generate the datasets
    train_set, val_set, test_set = generator.generate_sets(num_images=1500,
                                                           width_range=(10,
                                                                        20),
                                                           height_range=(10,
                                                                         20),
                                                           train_ratio=0.7,
                                                           val_ratio=0.2)

    # Preprocessing
    X_train, train_labels = preprocess_images_and_labels(
        train_set)  # Corrected variable name from y_train to train_labels
    X_test, test_labels = preprocess_images_and_labels(
        test_set)  # Corrected variable name from y_test to test_labels
    label_to_index = {'Rectangle': 0, 'Triangle': 1, 'Circle': 2, 'Cross': 3}
    y_train = labels_to_one_hot(
        train_labels,
        label_to_index)  # Convert labels to one-hot encoded labels
    y_test = labels_to_one_hot(
        test_labels,
        label_to_index)  # Convert labels to one-hot encoded labels

    # Train the network
    network.fit(X_train,
                y_train,
                epochs=config['globals']['epochs'],
                learning_rate=config['globals']['learning_rate'],
                batch_size=config['globals']['batch_size'],
                verbose=True)

    # Predict and plot
    test_predictions = predict(network, X_test)
    plot_sample(X_test, y_test, test_predictions, {
        v: k
        for k, v in label_to_index.items()
    }, n)


if __name__ == '__main__':
    config_path = './scripts/config/neural_net_config.yml'
    main(config_path)
