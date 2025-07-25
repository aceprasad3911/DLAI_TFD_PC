import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import unittests

# Directory that holds the data
DATA_DIR = './PetImages'

# Subdirectories for each class
data_dir_dogs = os.path.join(DATA_DIR, 'Dog')
data_dir_cats = os.path.join(DATA_DIR, 'Cat')

# os.listdir returns a list containing all files under the given dir
print(f"There are {len(os.listdir(data_dir_dogs))} images of dogs.")
print(f"There are {len(os.listdir(data_dir_cats))} images of cats.")

# Get the filenames for cats and dogs images
cats_filenames = [os.path.join(data_dir_cats, filename) for filename in os.listdir(data_dir_cats)]
dogs_filenames = [os.path.join(data_dir_dogs, filename) for filename in os.listdir(data_dir_dogs)]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('Cat and Dog Images', fontsize=16)

# Plot the first 4 images of each class
for i, cat_image in enumerate(cats_filenames[:4]):
    img = tf.keras.utils.load_img(cat_image)
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Example Cat {i}')

for i, dog_image in enumerate(dogs_filenames[:4]):
    img = tf.keras.utils.load_img(dog_image)
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'Example Dog {i}')

plt.show()

# GRADED FUNCTION: train_val_datasets

def train_val_datasets():
    """Creates datasets for training and validation.

    Returns:
        (tf.data.Dataset, tf.data.Dataset): Training and validation datasets.
    """

    ### START CODE HERE ###

    training_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        image_size=(120, 120),
        batch_size=128,
        label_mode='binary',
        validation_split=0.15,
        subset='training',
        seed=42
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        image_size=(120, 120),
        batch_size=128,
        label_mode='binary',
        validation_split=0.15,
        subset='training',
        seed=42
    )

    ### END CODE HERE ###

    return training_dataset, validation_dataset

# Create the datasets
training_dataset, validation_dataset = train_val_datasets()

# Test your code!
unittests.test_train_val_datasets(train_val_datasets)

# Get the first batch of images and labels
for images, labels in training_dataset.take(1):
	example_batch_images = images
	example_batch_labels = labels

print(f"Maximum pixel value of images: {np.max(example_batch_images)}\n")
print(f"Shape of batch of images: {example_batch_images.shape}")
print(f"Shape of batch of labels: {example_batch_labels.shape}")


# GRADED FUNCTION: create_augmentation_model
def create_augmentation_model():
    """Creates a model (layers stacked on top of each other) for augmenting images of cats and dogs.

    Returns:
        tf.keras.Model: The model made up of the layers that will be used to augment the images of cats and dogs.
    """

    ### START CODE HERE ###

    # Create the augmentation model.
    augmentation_model = tf.keras.Sequential([
        tf.keras.Input(shape=(120, 120, 3)),  # Input layer with the shape of each image
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images
        tf.keras.layers.RandomRotation(0.2),  # Randomly rotate images by 20%
        tf.keras.layers.RandomTranslation(0.1, 0.1),  # Randomly translate images
        tf.keras.layers.RandomZoom(0.1)  # Randomly zoom into images
    ])

    ### END CODE HERE ###

    return augmentation_model

# Load your model for augmentation
data_augmentor = create_augmentation_model()

# Take a sample image
sample_image = tf.keras.utils.array_to_img(example_batch_images[0])

images = [sample_image]

# Apply random augmentation 3 times
for _ in range(3):
	image_aug = data_augmentor(tf.expand_dims(sample_image, axis=0))
	image_aug = tf.keras.utils.array_to_img(tf.squeeze(image_aug))
	images.append(image_aug)


fig, axs = plt.subplots(1, 4, figsize=(14, 7))
for ax, image, title in zip(axs, images, ['Original Image', 'Augmented 1', 'Augmented 2', 'Augmented 3']):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.show()

# Test your code!
unittests.test_create_augmentation_model(create_augmentation_model)


# GRADED FUNCTION: create_model

def create_model():
    """Creates the untrained model for classifying cats and dogs.

    Returns:
        tf.keras.Model: The model that will be trained to classify cats and dogs.
    """

    ### START CODE HERE ###

    # Get the augmentation layers (or model) from your earlier function
    augmentation_layers = create_augmentation_model()

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(120, 120, 3)),  # Input layer with the shape of each image
        augmentation_layers,  # Add the augmentation layers
        tf.keras.layers.Rescaling(1./255),  # Rescale pixel values to [0, 1]
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # First convolutional layer
        tf.keras.layers.MaxPooling2D(),  # First max pooling layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
        tf.keras.layers.MaxPooling2D(),  # Second max pooling layer
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
        tf.keras.layers.MaxPooling2D(),  # Third max pooling layer
        tf.keras.layers.Flatten(),  # Flatten the output
        tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(
        optimizer='adam',  # Adam optimizer
        loss='binary_crossentropy',  # Loss function for binary classification
        metrics=['accuracy']  # Metric to monitor
    )

    ### END CODE HERE ###

    return model


# Create the compiled but untrained model
model = create_model()

# Check parameter count against a reference solution
unittests.parameter_count(model)

try:
	model.evaluate(example_batch_images, example_batch_labels, verbose=False)
except:
	print("Your model is not compatible with the dataset you defined earlier. Check that the loss function, last layer and label_mode are compatible with one another.")
else:
	predictions = model.predict(example_batch_images, verbose=False)
	print(f"predictions have shape: {predictions.shape}")

# Test your code!
unittests.test_create_model(create_model)


# GRADED CLASS: EarlyStoppingCallback

### START CODE HERE ###
# Remember to inherit from the correct class

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy is greater or equal to 0.95 and validation accuracy is greater or equal to 0.8
        train_accuracy = logs.get('accuracy')  # Get training accuracy from logs
        val_accuracy = logs.get('val_accuracy')  # Get validation accuracy from logs

        if train_accuracy >= 0.8 and val_accuracy >= 0.8:
            self.model.stop_training = True  # Stop training
            print("\nReached 95% train accuracy and 80% validation accuracy, so cancelling training!")


### END CODE HERE ###

# Test your code!
unittests.test_EarlyStoppingCallback(EarlyStoppingCallback)

# Train the model and save the training history
# This may take up to 10-15 min so feel free to take a break! :P
history = model.fit(
	training_dataset,
	epochs=35,
	validation_data=validation_dataset,
	callbacks = [EarlyStoppingCallback()]
)

# Get training and validation accuracies
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Training and validation accuracy')

for i, (data, label) in enumerate(zip([(acc, val_acc), (loss, val_loss)], ["Accuracy", "Loss"])):
    ax[i].plot(epochs, data[0], 'r', label="Training " + label)
    ax[i].plot(epochs, data[1], 'b', label="Validation " + label)
    ax[i].legend()
    ax[i].set_xlabel('epochs')

plt.show()

# Test your code!
unittests.test_history(history)

with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)