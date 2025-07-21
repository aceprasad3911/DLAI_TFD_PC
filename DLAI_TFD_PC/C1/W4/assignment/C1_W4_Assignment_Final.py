import os
import base64
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import unittests

BASE_DIR = "./data/"
happy_dir = os.path.join(BASE_DIR, "happy/")
sad_dir = os.path.join(BASE_DIR, "sad/")

fig, axs = plt.subplots(1, 2, figsize=(6, 6))
axs[0].imshow(tf.keras.utils.load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}"))
axs[0].set_title('Example happy Face')

axs[1].imshow(tf.keras.utils.load_img(f"{os.path.join(sad_dir, os.listdir(sad_dir)[0])}"))
axs[1].set_title('Example sad Face')

plt.tight_layout()

# Load the first example of a happy face
sample_image  = tf.keras.utils.load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}")

# Convert the image into its numpy array representation
sample_array = tf.keras.utils.img_to_array(sample_image)

print(f"Each image has shape: {sample_array.shape}")

print(f"The maximum pixel value used is: {np.max(sample_array)}")

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] >= 0.999:
            self.model.stop_training = True
            print("\nReached 99.9% accuracy so cancelling training!")


# GRADED FUNCTION: training_dataset

def training_dataset():
    """Creates the training dataset out of the training images. Pixel values should be normalized.

    Returns:
        tf.data.Dataset: The dataset including the images of happy and sad faces.
    """

    ### START CODE HERE ###

    # Specify the function to load images from a directory and pass in the appropriate arguments:
    # - directory: should be a relative path to the directory containing the data.
    #              You may hardcode this or use the previously defined global variable.
    # - image_size: set this equal to the resolution of each image (excluding the color dimension)
    # - batch_size: number of images the generator yields when asked for a next batch. Set this to 10.
    # - class_mode: How the labels are represented. Should be one of "binary", "categorical" or "int".
    #               Pick the one that better suits here given that the labels can only be two different values.
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        BASE_DIR,
        image_size=(150, 150),
        batch_size=10,
        label_mode='binary'
    )
    # Define the rescaling layer (passing in the appropriate parameters)
    rescale_layer = tf.keras.layers.Rescaling(1./255)
    # Apply the rescaling by using the map method and a lambda
    train_dataset_scaled = train_dataset.map(lambda x, y: (rescale_layer(x), y))
    ### END CODE HERE ###

    return train_dataset_scaled

# Save your generator in a variable
train_data = training_dataset()

for images, labels in train_data.take(1):
    print(f"Range for pixel values: {np.min(images[0]), np.max(images[0])}")

print(f"train_data is an instance of tf.data.Dataset: {isinstance(train_data, tf.data.Dataset)}")

# Test your code!
unittests.test_train_data(train_data)


# GRADED FUNCTION: create_and_compile_model

def create_and_compile_model():
    """Creates, compiles and trains the model to predict happy from sad faces.

    Returns:
        tf.keras.Model: The model that will be trained to predict predict happy and sad faces.
    """

    ### START CODE HERE ###
# Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )
    ### END CODE HERE ###

    return model

# Save untrained model in a variable
model = create_and_compile_model()

# Check parameter count against a reference solution
unittests.parameter_count(model)

# Get the first batch of images and labels
for images, labels in train_data.take(1):
	example_batch_images = images
	example_batch_labels = labels

try:
	model.evaluate(example_batch_images, example_batch_labels, verbose=False)
except:
	print("Your model is not compatible with the dataset you defined earlier. Check that the loss function, last layer and label_mode are compatible with one another.")
else:
	predictions = model.predict(example_batch_images, verbose=False)
	print(f"predictions have shape: {predictions.shape}")

# Test your code!
unittests.test_create_and_compile_model(create_and_compile_model)

# Get the training history from your model
training_history = model.fit(
	x=train_data,
    epochs=15,
    callbacks=[EarlyStoppingCallback()]
)

# Test your code!
unittests.test_training_history(training_history)







