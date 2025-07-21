# First import all the libraries you will need
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Normalize the pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Setup training parameters
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("\nMODEL TRAINING:")
model.fit(training_images, training_labels, epochs=5)

# Evaluate on the test set
print("\nMODEL EVALUATION:")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f'test set accuracy: {test_accuracy}')
print(f'test set loss: {test_loss}')

# Convolutions are filters passed over images to change the underlying image
# For every pixel, take its value and its neighbors
# If filter is 3x3, you can look at its immediate neighbors ( -1 < x < 1, -1 < y < 1)
# To get new value for pixel, you multiply each neighbor value by corresponding filter value
# Changes image to emphasize certain features

# Pooling is a way of compressing an image
# Go over image 4 pixels at a time (x - 1, y - 1, x & y - 1)
# Of those 4, retain the largest value and discard the rest
# 16 pixels in grid -> 4 pixels in grid

# Define the model
model = tf.keras.models.Sequential([
    # Add convolutions and max pooling
    tf.keras.Input(shape=(28, 28, 1)), # 1 signifies single byte for color depth (greyscale)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # generate 64 filters, which are 3x3 pixels
    # activation = 'relu' so negative values are thrown away
    tf.keras.layers.MaxPooling2D(2, 2), # MaxPooling gets the maximum value, 2x2 pool
    # For every 2x2 pool; we get the largest value.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Add the same layers as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Print the model summary
model.summary()
# Allows us to inspect the layers of the model and see the journey of image through the convolutions
# Convolution (filter) row takes away the furthest right / left column and row due to 1 pixel margin not working on borders
# Max Pooling row will halve x and y as only the largest value is taken 2x2 -> 1x1, and will round downwards
# Flatten will multiply 25 pixels (5x5 after 2 convolution & maxpooling) by 64 filters = 1600

# Use the same settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("\nMODEL TRAINING:")
model.fit(training_images, training_labels, epochs=5)

# Evaluate on the test set
print("\nMODEL EVALUATION:")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f'test set accuracy: {test_accuracy}')
print(f'test set loss: {test_loss}')

print(f"First 100 labels:\n\n{test_labels[:100]}")

print(f"\nShoes: {[i for i in range(100) if test_labels[:100][i]==9]}")

################################
# VISUALIZATION OF IMAGE JOURNEY
################################

FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 1
layers_to_visualize = [tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D]

layer_outputs = [layer.output for layer in model.layers if type(layer) in layers_to_visualize]
activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

f, axarr = plt.subplots(3, len(layer_outputs))

for x in range(len(layer_outputs)):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1), verbose=False)[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)

    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1), verbose=False)[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)

    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1), verbose=False)[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2,x].grid(False)
