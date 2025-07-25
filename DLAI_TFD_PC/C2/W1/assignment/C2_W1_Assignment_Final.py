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
        image_size=(150, 150),
        batch_size=128,           # Number of images to return in each batch
        label_mode='binary',     # Binary labels for two classes
        validation_split=0.1,    # Use 20% of data for validation
        subset='training',       # Specify this subset as training
        seed=42
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        image_size=(150, 150),
        batch_size=128,           # Number of images to return in each batch
        label_mode='binary',     # Binary labels for two classes
        validation_split=0.1,    # Use 20% of data for validation
        subset='validation',     # Specify this subset as validation
        seed=42
    )

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


# GRADED FUNCTION: create_model

def create_model():
    """Creates the untrained model for classifying cats and dogs.

    Returns:
        tf.keras.Model: The model that will be trained to classify cats and dogs.
    """

    ### START CODE HERE ###

    model = tf.keras.models.Sequential([
        # Rescale the image. Note the input shape is the desired size of the image: 150x150 with 3 bytes for color
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.Rescaling(1. / 255),  # Rescaling layer so map method is not needed
        # Convolution and Pooling layers
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),  # Convolution and Pooling loop
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for one class ('cats') and 1 for the other ('dogs')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
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
class EarlyStoppingCallback():

    # Remember to inherit from the correct class
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy is greater or equal to 0.95 and validation accuracy is greater or equal to 0.8
        if logs['accuracy'] >= 0.95 and logs['val_accuracy'] >= 0.8:
            self.model.stop_training = True  # Stop training
            print("\nReached 95% train accuracy and 80% validation accuracy, so cancelling training!")

### END CODE HERE ###

# Test your code!
unittests.test_EarlyStoppingCallback(EarlyStoppingCallback)

# Train the model and save the training history (this may take some time)
history = model.fit(
	training_dataset,
	epochs=15,
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

with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)