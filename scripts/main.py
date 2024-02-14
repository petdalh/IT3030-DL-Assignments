import sys
import os
import yaml

script_dir = os.path.dirname(__file__)  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.append(parent_dir)

import numpy as np
from neural_network import NeuralNetwork
from neural_network import ReLU, Softmax, Linear, Sigmoid, Tanh
from neural_network import L1Regularizer, L2Regularizer
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
    activation_map = {
        'relu': ReLU,
        'softmax': Softmax,
        'linear': Linear,
        'sigmoid': Sigmoid,
        'tanh': Tanh
    }

    loss_function_map = {
        'cross_entropy': CrossEntropyLoss,
        'mse': MSELoss,
    }

    regularizer_map = {
        'L1': L1Regularizer,
        'L2': L2Regularizer,
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


def create_dataset_from_config(config):
    dataset_type = config['dataset']['type']
    dataset_params = config['dataset']['params']

    if dataset_type == 'image_generator':
        n = dataset_params['n']
        noise = dataset_params['noise']
        num_images = dataset_params['num_images']
        width_range = dataset_params['width_range']
        height_range = dataset_params['height_range']
        train_ratio = dataset_params['train_ratio']
        val_ratio = dataset_params['val_ratio']

        generator = ImageGenerator(n, noise)
        train_set, val_set, test_set = generator.generate_sets(
            num_images=num_images,
            width_range=width_range,
            height_range=height_range,
            train_ratio=train_ratio,
            val_ratio=val_ratio)

        return train_set, val_set, test_set
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def main(config_path):
    # Load the configuration
    config = load_config(config_path)
    n = config['dataset']['params']['n']

    # Create the neural network from the configuration
    network = create_network_from_config(config)

    # Generate the dataset based on the configuration
    train_set, val_set, test_set = create_dataset_from_config(config)

    # Preprocess the training, validation, and test sets
    X_train, train_labels = preprocess_images_and_labels(train_set)
    X_val, val_labels = preprocess_images_and_labels(
        val_set)  # Process validation set
    X_test, test_labels = preprocess_images_and_labels(test_set)

    # Convert labels to one-hot encoded labels
    label_to_index = {'Rectangle': 0, 'Triangle': 1, 'Circle': 2, 'Cross': 3}
    y_train = labels_to_one_hot(train_labels, label_to_index)
    y_val = labels_to_one_hot(val_labels,
                              label_to_index)  # Convert validation labels
    y_test = labels_to_one_hot(test_labels, label_to_index)

    # Train the network with validation data
    network.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=config['globals']['epochs'],
        learning_rate=config['globals']['learning_rate'],
        batch_size=config['globals']['batch_size'],
        verbose=True,
    )  # Pass validation data

    # Predict and plot results for the test set
    test_predictions = predict(network, X_test)
    plot_sample(X_test, y_test, test_predictions, {
        v: k
        for k, v in label_to_index.items()
    }, n)


    network.plot_training_progress()


if __name__ == '__main__':
    config_path = './scripts/config/config1.yml'
    # config_path = './scripts/config/config2.yml'
    # config_path = './scripts/config/config3.yml'
    main(config_path)
