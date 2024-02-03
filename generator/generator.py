from PIL import Image, ImageDraw
import numpy as np
import random


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

        for i in range(num_images):
            shape_selector = i % 4  # Cycle through 0, 1, 2, 3 for rectangle, triangle, circle, cross
            if shape_selector == 0:
                image = self.generate_rectangle(width_range, height_range)
            elif shape_selector == 1:
                image = self.generate_triangle(width_range, height_range)
            elif shape_selector == 2:
                image = self.generate_circle(width_range, height_range)
            else:  # shape_selector == 3
                image = self.generate_cross(width_range, height_range)

            if i < train_size:
                training_set.append(image)
            elif i < train_size + val_size:
                validation_set.append(image)
            else:
                testing_set.append(image)

        return training_set, validation_set, testing_set



# Example usage
n = 20  # Choose the image size (n x n)
noise = 0.05  # Adjust the noise parameter as needed
generator = ImageGenerator(n, noise)

width_range = (n // 4, n // 2)  # Adjust as desired
height_range = (n // 4, n // 2)  # Adjust as desired
train_ratio = 0.7  # Adjust as desired
val_ratio = 0.2  # Adjust as desired
test_ratio = 0.1  # Adjust as desired

train_set, val_set, test_set = generator.generate_sets(4*4, width_range,
                                                       height_range,
                                                       train_ratio, val_ratio)
random_training_image = random.choice(train_set)

# Resize the image to make it bigger before displaying
display_size = (200, 200)  # You can adjust this size as needed
image = random_training_image.resize(display_size, Image.NEAREST)

image.show()
