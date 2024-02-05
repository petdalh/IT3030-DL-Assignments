from PIL import Image, ImageDraw
import numpy as np
import random
import matplotlib.pyplot as plt



class ImageGenerator:

    def __init__(self, n, noise):
        self.n = n
        self.noise = noise

    def generate_rectangle(self, width_range, height_range):
        width = np.random.randint(*width_range)
        height = np.random.randint(*height_range)
        x = np.random.randint(0, self.n - width)
        y = np.random.randint(0, self.n - height)

        image = Image.new('1', (self.n, self.n), 0)
        draw = ImageDraw.Draw(image)
        draw.rectangle([x, y, x + width, y + height], fill=1)

        # Add noise
        for _ in range(int(self.n * self.n * self.noise)):
            x_noise = np.random.randint(0, self.n)
            y_noise = np.random.randint(0, self.n)
            draw.point((x_noise, y_noise), fill=1)

        return image

    def generate_triangle(self, width_range, height_range):
        width = np.random.randint(*width_range)
        height = np.random.randint(*height_range)
        x = np.random.randint(0, self.n - width)
        y = np.random.randint(0, self.n - height)

        image = Image.new('1', (self.n, self.n), 0)
        draw = ImageDraw.Draw(image)
        draw.polygon([(x, y), (x + width, y), (x + width // 2, y + height)],
                     fill=1)

        # Add noise
        for _ in range(int(self.n * self.n * self.noise)):
            x_noise = np.random.randint(0, self.n)
            y_noise = np.random.randint(0, self.n)
            draw.point((x_noise, y_noise), fill=1)

        return image

    def generate_circle(self, width_range, height_range):
        width = np.random.randint(*width_range)
        height = np.random.randint(*height_range)
        x = np.random.randint(0, self.n - width)
        y = np.random.randint(0, self.n - height)

        image = Image.new('1', (self.n, self.n), 0)
        draw = ImageDraw.Draw(image)
        draw.ellipse([x, y, x + width, y + height], fill=1)

        # Add noise
        for _ in range(int(self.n * self.n * self.noise)):
            x_noise = np.random.randint(0, self.n)
            y_noise = np.random.randint(0, self.n)
            draw.point((x_noise, y_noise), fill=1)

        return image
    
    def generate_cross(self, width_range, height_range):
        width = np.random.randint(*width_range)
        height = np.random.randint(*height_range)
        x = np.random.randint(0, self.n - width)
        y = np.random.randint(0, self.n - height)

        image = Image.new('1', (self.n, self.n), 0)
        draw = ImageDraw.Draw(image)
        draw.line([x, y, x + width, y + height], fill=1)
        draw.line([x, y + height, x + width, y], fill=1)

        # Add noise
        for _ in range(int(self.n * self.n * self.noise)):
            x_noise = np.random.randint(0, self.n)
            y_noise = np.random.randint(0, self.n)
            draw.point((x_noise, y_noise), fill=1)

        return image

    def generate_sets(self, num_images, width_range, height_range, train_ratio, val_ratio):
        train_size = int(train_ratio * num_images)
        val_size = int(val_ratio * num_images)
        test_size = num_images - train_size - val_size

        training_set = []
        validation_set = []
        testing_set = []

        # Label assignments
        labels = ['Rectangle', 'Triangle', 'Circle', 'Cross']

        for i in range(num_images):
            shape_selector = i % 4
            label = labels[shape_selector]

            if shape_selector == 0:
                image = self.generate_rectangle(width_range, height_range)
            elif shape_selector == 1:
                image = self.generate_triangle(width_range, height_range)
            elif shape_selector == 2:
                image = self.generate_circle(width_range, height_range)
            else:  # shape_selector == 3
                image = self.generate_cross(width_range, height_range)

            image_label_pair = (image, label)

            if i < train_size:
                training_set.append(image_label_pair)
            elif i < train_size + val_size:
                validation_set.append(image_label_pair)
            else:
                testing_set.append(image_label_pair)

        return training_set, validation_set, testing_set


# Assume 'generator' is an instance of ImageGenerator with your desired parameters
generator = ImageGenerator(n=40, noise=0.05)  # Example parameters

# Generate a dataset
train_set, val_set, test_set = generator.generate_sets(num_images=100, width_range=(10, 20), height_range=(10, 20), train_ratio=0.7, val_ratio=0.2)

# Function to plot a random image along with its label
def plot_random_image_with_label(image_label_pairs):
    # Select a random image and label pair
    image, label = random.choice(image_label_pairs)
    
    # Convert image to numpy array for plotting
    image_array = np.array(image)
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap='gray')  # Plot the image
    ax.set_title(f'Label: {label}')  # Set the title to the label of the image
    ax.axis('off')  # Hide the axes ticks
    
    plt.show()  # Display the plot

# Example usage to plot a random image from the training set
plot_random_image_with_label(train_set)