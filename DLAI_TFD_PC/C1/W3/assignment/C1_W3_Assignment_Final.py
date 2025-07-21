import os
import base64
import numpy as np
import tensorflow as tf

import unittests

# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/mnist.npz")

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")


# GRADED FUNCTION: reshape_and_normalize

def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """

    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

    # Normalize pixel values
    images = images / 255.0

    ### END CODE HERE ###

    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

# Test your code!
unittests.test_reshape_and_normalize(reshape_and_normalize)


# GRADED CLASS: EarlyStoppingCallback

### START CODE HERE ###

# Remember to inherit from the correct class
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy is greater or equal to 0.995
        if logs['accuracy'] >= 0.995:
            # Stop training once the above condition is met
            self.model.stop_training = True
            print("\nReached 99.5% accuracy so cancelling training!")

        ### END CODE HERE ###

# Test your code!
unittests.test_EarlyStoppingCallback(EarlyStoppingCallback)


# GRADED FUNCTION: convolutional_model

def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    ## START CODE HERE ###

    # Define the model
    model = tf.keras.models.Sequential([
        # Add convolutions and max pooling
        tf.keras.Input(shape=(28, 28, 1)),  # 1 signifies single byte for color depth (greyscale)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # generate 64 filters, which are 3x3 pixels
        # activation = 'relu' so negative values are thrown away
        tf.keras.layers.MaxPooling2D(2, 2),  # MaxPooling gets the maximum value, 2x2 pool
        # For every 2x2 pool; we get the largest value.
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    ### END CODE HERE ###

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Define your compiled (but untrained) model
model = convolutional_model()

# Check parameter count against a reference solution
unittests.parameter_count(model)

# Train your model (this can take up to 5 minutes)
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])

# Test your code!
unittests.test_training_history(training_history)