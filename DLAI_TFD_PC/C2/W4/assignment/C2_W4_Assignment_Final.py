import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import unittests

TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'

fig, axes = plt.subplots(1, 6, figsize=(14, 3))
fig.suptitle('Sign Language MNIST Images', fontsize=16)

# Plot one image from the first 4 letters
for ii, letter in enumerate(['A' , 'B', 'C', 'D', 'E', 'F']):
    dir = f'./data/train/{letter}'
    img = tf.keras.preprocessing.image.load_img(dir+'/'+os.listdir(dir)[0])
    axes[ii].imshow(img)
    axes[ii].set_title(f'Example of letter {letter}')

# Convert the image into its numpy array representation
sample_array = tf.keras.preprocessing.image.img_to_array(img)

print(f"Each image has shape: {sample_array.shape}")

sample_array[0,:5]


# GRADED FUNCTION: train_val_datasets
def train_val_datasets():
    """Create train and validation datasets

    Returns:
        (tf.data.Dataset, tf.data.Dataset): train and validation datasets
    """
    ### START CODE HERE ###
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        batch_size=32,
        image_size=(28, 28),
        color_mode='grayscale', # Use this argument to get just one color dimension, because it is greyscale
        label_mode='categorical'
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=VALIDATION_DIR,
        batch_size=32,
        image_size=(28, 28),
        color_mode='grayscale',  # Use this argument to get just one color dimension, because it imgs are greyscale
        label_mode='categorical'
    )
    ### END CODE HERE ###

    return train_dataset, validation_dataset

# Create train and validation datasets
train_dataset, validation_dataset = train_val_datasets()
print(f"Images of train dataset have shape: {train_dataset.element_spec[0].shape}")
print(f"Labels of train dataset have shape: {train_dataset.element_spec[1].shape}")
print(f"Images of validation dataset have shape: {validation_dataset.element_spec[0].shape}")
print(f"Labels of validation dataset have shape: {validation_dataset.element_spec[1].shape}")

# Test your function
unittests.test_train_val_datasets(train_val_datasets)


# GRADED FUNCTION: create_model
def create_model():
    """Create the classifier model

    Returns:
        tf.keras.model.Sequential: CNN for multi-class classification
    """
    ### START CODE HERE ###

    # Define the model
    # Use no more than 2 Conv2D and 2 MaxPooling2D
    model = tf.keras.models.Sequential([
        # Define an input layer
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),  # Set correct input size
        # Rescale images to [0, 1]
        tf.keras.layers.Rescaling(1. / 255),  # Rescale pixel values to be between 0 and 1
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 32 filters, 3x3 kernel
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 64 filters, 3x3 kernel
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer
        # Flatten the output
        tf.keras.layers.Flatten(),  # Flatten the 3D output to 1D
        # Fully connected layer
        tf.keras.layers.Dense(128, activation='relu'),  # Dense layer with 128 units
        # Output layer
        tf.keras.layers.Dense(24, activation='softmax')  # Output layer for 24 classes
    ])

    # Compile the model
    model.compile(optimizer='adam',  # Adam optimizer
                  loss='categorical_crossentropy',  # Suitable loss for multi-class classification
                  metrics=['accuracy'])  # Monitor accuracy

    ### END CODE HERE ###
    return model

# Create your model
model = create_model()

# Check parameter count against a reference solution
unittests.parameter_count(model)

print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')

model.summary()

for images, labels in train_dataset.take(1):
    example_batch_images = images
    example_batch_labels = labels

try:
    model.evaluate(example_batch_images, example_batch_labels, verbose=False)
except:
    print("Your model is not compatible with the dataset you defined earlier. Check that the loss function, last layer and label_mode are compatible with one another.")
else:
    predictions = model.predict(example_batch_images, verbose=False)
    print(f"predictions have shape: {predictions.shape}"

unittests.test_create_model(create_model)

# Train your model
history = model.fit(train_dataset,
                    epochs=15,
                    validation_data=validation_dataset)

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



