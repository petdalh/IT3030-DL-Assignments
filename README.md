# Neural Network Framework made as a project for the course IT3030

This project implements a simple neural network framework from scratch using Python and NumPy. It is designed to classify synthetic images of geometric shapes (rectangles, triangles, circles, and crosses) into their respective categories. The framework includes a customizable neural network architecture, activation functions, loss functions, and a regularization option. Additionally, it features an image generator for creating a dataset of synthetic images with optional noise.

## Features

- **Neural Network Class**: Core model managing layers, forward/backward propagation, and parameter updates.
- **Layer Class**: Represents individual layers with support for different activations and regularizers.
- **Activation Functions**: Includes ReLU, Sigmoid, and Softmax for introducing non-linearities.
- **Loss Functions**: Implements CrossEntropyLoss and MSELoss for evaluating model performance.
- **Regularizers**: Supports L1 regularization to prevent overfitting.
- **Image Generator**: Utility to generate synthetic images for training and testing the model.
- **Configuration Driven**: Model architecture, training parameters, and dataset generation can be defined in a YAML configuration file.


## Usage

1. **Configure the Neural Network and Dataset**: Edit the `neural_net_config.yml` file to specify the network architecture, learning parameters, and dataset generation settings.

2. **Train the Model**: Run the main script to train the model on the generated dataset.

3. **Evaluation and Visualization**: The script will train the neural network and display sample images with their true and predicted labels after training.


## Components

- `neural_network.py`: Implements the neural network and training process.
- `layer.py`: Implements the layer used in the network
- `activation_functions.py`: Contains activation function classes.
- `loss_functions.py`: Defines loss function classes for model evaluation.
- `regularization.py`: Includes regularization classes.
- `generator.py`: Script for generating synthetic image datasets.
- `config/neural_net_config.yml`: Configuration file for setting up the neural network and dataset.

## Customization

You can customize the neural network by adjusting the configuration in `neural_net_config.yml`. Available options include layer sizes, activations, regularization, and training parameters like learning rate and epochs.


